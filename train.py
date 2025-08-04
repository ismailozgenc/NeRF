import os
import torch
import torch.nn.functional as F
from datetime import datetime
from nerf.dataset import load_colmap_dataset
from nerf.encoding import positional_encoding
from nerf.model    import NeRF
from nerf.render   import get_rays, sample_points, volume_render, sample_pdf
from utils.config  import *
from utils.helpers import *
import numpy as np

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(CKPT_DIR, exist_ok=True)

    # data + models + optimizer
    dataset      = load_colmap_dataset(DATA_DIR, IMG_DIR)
    model_coarse = NeRF().to(device)
    model_fine   = NeRF().to(device)
    optimizer    = torch.optim.Adam(
        list(model_coarse.parameters()) + list(model_fine.parameters()),
        lr=LR
    )

    scaler = torch.cuda.amp.GradScaler()

    ckpt_path = f"{CKPT_DIR}/nerf_hierarchical.pth"
    start_iter, best_psnr = load_checkpoint(model_coarse, model_fine, optimizer, ckpt_path, device)
    for param in optimizer.param_groups: param['lr'] = LR  
    print(f"Best PSNR until now {best_psnr:.2f}")
    print(f"Current LR {optimizer.param_groups[0]['lr']}")
    for i in range(start_iter, N_ITERS+1):
        data = dataset[i % len(dataset)]
        img  = torch.from_numpy(data['image']).float().to(device) / 255.0
        H, W = img.shape[:2]
        qvec = torch.from_numpy(data['qvec']).float().to(device)
        tvec = torch.from_numpy(data['tvec']).float().to(device)

        rays_o, rays_d = get_rays(H, W, data['intrinsics'], qvec, tvec)
        rays_o = rays_o.reshape(-1,3)
        rays_d = rays_d.reshape(-1,3)
        idxs   = torch.randperm(rays_o.size(0), device=device)[:BATCH_RAYS]
        ro, rd = rays_o[idxs], rays_d[idxs]

        with torch.cuda.amp.autocast():
            # --- coarse pass ---
            pts_co, t_co = sample_points(ro, rd, NEAR, FAR, N_SAMPLES)
            pe_co  = positional_encoding(pts_co.reshape(-1,3), FREQ_POS)
            pe_dir = positional_encoding(
                rd.unsqueeze(1).expand(-1, N_SAMPLES, 3).reshape(-1,3),
                FREQ_DIR
            )
            rgb_co, sig_co = model_coarse(pe_co, pe_dir)
            rgb_co = rgb_co.view(-1, N_SAMPLES, 3)
            sig_co = sig_co.view(-1, N_SAMPLES, 1)
            comp_co, depth_co, weights = volume_render(rgb_co, sig_co, t_co, rd)

            # --- fine sampling ---
            bins   = t_co.unsqueeze(0).expand(weights.shape[0], -1)
            t_fine = sample_pdf(bins, weights, N_IMPORTANCE)
            t_all  = torch.sort(torch.cat([bins, t_fine], dim=-1), dim=-1)[0]

            # --- fine pass ---
            pts_fi = ro.unsqueeze(1) + rd.unsqueeze(1) * t_all[..., :, None]
            pe_fi = positional_encoding(pts_fi.reshape(-1,3), FREQ_POS)
            pe_dir_fi = positional_encoding(
                rd.unsqueeze(1).expand(-1, t_all.shape[-1], 3).reshape(-1,3),
                FREQ_DIR
            )
            rgb_fi, sig_fi = model_fine(pe_fi, pe_dir_fi)
            rgb_fi = rgb_fi.view(-1, t_all.shape[-1], 3)
            sig_fi = sig_fi.view(-1, t_all.shape[-1], 1)
            comp_fi, _, _ = volume_render(rgb_fi, sig_fi, t_all, rd)

            # --- loss ---
            target = img.reshape(-1,3)[idxs]
            loss_co = F.mse_loss(comp_co, target)
            loss_fi = F.mse_loss(comp_fi, target)
            loss    = loss_co + loss_fi

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        psnr_co = -10 * torch.log10(loss_co.detach())
        psnr_fi = -10 * torch.log10(loss_fi.detach())

        if psnr_fi > best_psnr:
            best_psnr = psnr_fi
            save_checkpoint(model_coarse, model_fine, optimizer, best_psnr, i, ckpt_path)
            print(f"Best @ iter={i}, psnr_co={psnr_co:.2f}, psnr_fi={psnr_fi:.2f}, time={datetime.now():%H:%M}")

        if i % LOG_INTERVAL == 0:
            print(f"iter={i}, psnr_co={psnr_co:.2f}, psnr_fi={psnr_fi:.2f}, time={datetime.now():%H:%M}")
