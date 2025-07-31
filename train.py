import os
import torch
import torch.nn.functional as F
from nerf.dataset import load_colmap_dataset
from nerf.render import get_rays, sample_points, volume_render
from nerf.encoding import positional_encoding
from nerf.model import NeRF
from utils.config import *
from utils.helpers import load_checkpoint, save_checkpoint
from datetime import datetime

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(CKPT_DIR, exist_ok=True)

dataset = load_colmap_dataset(DATA_DIR, IMG_DIR)
model   = NeRF().to(device)
opt     = torch.optim.Adam(model.parameters(), lr=LR)

def psnr(mse):
    return -10.0 * torch.log10(mse)

ckpt_path          = f"{CKPT_DIR}/nerf_best.pth"
start_iter, best_psnr = load_checkpoint(model, opt, ckpt_path, device)
print(f"Best PSNR up to now {best_psnr:.2f}")

for i in range(start_iter, N_ITERS + 1):
    model.train()
    img_data = dataset[i % len(dataset)]
    img      = torch.from_numpy(img_data["image"]).float().to(device)
    H, W     = img.shape[:2]

    qvec = torch.from_numpy(img_data["qvec"]).float().to(device)
    tvec = torch.from_numpy(img_data["tvec"]).float().to(device)
    rays_o, rays_d = get_rays(H, W, img_data["intrinsics"], qvec, tvec)
    rays_o = rays_o.reshape(-1, 3)
    rays_d = rays_d.reshape(-1, 3)

    idxs = torch.randperm(rays_o.size(0), device=device)[:BATCH_RAYS]
    ro, rd = rays_o[idxs], rays_d[idxs]

    pts, t_vals = sample_points(ro, rd, NEAR, FAR, N_SAMPLES)
    pts_flat    = pts.reshape(-1, 3)
    d_flat      = rd.unsqueeze(1).expand(-1, N_SAMPLES, -1).reshape(-1, 3)

    pe_pts = positional_encoding(pts_flat, FREQ_POS, True)
    pe_dir = positional_encoding(d_flat, FREQ_DIR, True)

    rgb_pred, sig_pred = model(pe_pts, pe_dir)
    rgb_pred = rgb_pred.view(BATCH_RAYS, N_SAMPLES, 3)
    sig_pred = sig_pred.view(BATCH_RAYS, N_SAMPLES, 1)

    comp_rgb, _ = volume_render(rgb_pred, sig_pred, t_vals, rd)
    comp_rgb    = comp_rgb.squeeze(0)
    target      = img.reshape(-1, 3)[idxs] / 255.0

    loss = F.mse_loss(comp_rgb, target)
    opt.zero_grad()
    loss.backward()
    opt.step()

    mse    = loss.detach()
    v_psnr = psnr(mse)
    if v_psnr > best_psnr:
        best_psnr = v_psnr
        save_checkpoint(model, opt, best_psnr, i, ckpt_path)
        print(f"BM Saved at iteration {i}, loss={mse:.4f}, psnr={v_psnr:.2f}, time={datetime.now().strftime('%H:%M')}")
         