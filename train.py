import os
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from nerf.dataset import load_colmap_dataset
from nerf.render import get_rays, sample_points, volume_render
from nerf.encoding import positional_encoding
from nerf.model import NeRF

DATA_DIR     = "data/south-building/sparse"
IMG_DIR      = "data/south-building/images"
N_ITERS      = 200000
LR           = 5e-4
BATCH_RAYS   = 1024
N_SAMPLES    = 64
NEAR, FAR    = 2.0, 6.0
FREQ_POS     = 10
FREQ_DIR     = 4
LOG_INTERVAL = 100
CKPT_DIR     = "checkpoints"

os.makedirs(CKPT_DIR, exist_ok=True)
writer = SummaryWriter("logs")

dataset = load_colmap_dataset(DATA_DIR, IMG_DIR)
device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model   = NeRF().to(device)
opt     = torch.optim.Adam(model.parameters(), lr=LR)

def psnr(mse):
    return -10.0 * torch.log10(mse)

best_psnr = float('-inf')

ckpt_path = os.path.join(CKPT_DIR, "nerf_best.pth")
if os.path.exists(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    opt.load_state_dict(ckpt["opt"])
    start_iter = ckpt["iter"] + 1
    best_psnr  = ckpt.get("best_psnr", float('-inf'))
    print(f"[Checkpoint Loaded] Resuming from iter {start_iter}, best_psnr={best_psnr:.2f}")
else:
    start_iter = 1
    best_psnr  = float('-inf')

for i in range(1, N_ITERS + 1):
    img_data = dataset[i % len(dataset)]
    img      = torch.from_numpy(img_data["image"]).float().to(device)
    H, W     = img.shape[:2]

    qvec = torch.from_numpy(img_data["qvec"]).float().to(device)
    tvec = torch.from_numpy(img_data["tvec"]).float().to(device)
    rays_o, rays_d = get_rays(H, W, img_data["intrinsics"], qvec, tvec)

    rays_o = rays_o.reshape(-1, 3)
    rays_d = rays_d.reshape(-1, 3)

    idxs = torch.randperm(rays_o.shape[0], device=device)[:BATCH_RAYS]
    ro, rd = rays_o[idxs], rays_d[idxs]

    pts, t_vals = sample_points(ro, rd, NEAR, FAR, N_SAMPLES)
    pts_flat    = pts.reshape(-1, 3)
    d_flat      = rd.unsqueeze(1).expand(-1, N_SAMPLES, -1).reshape(-1, 3)

    pe_pts = positional_encoding(pts_flat, FREQ_POS, True)
    pe_dir = positional_encoding(d_flat, FREQ_DIR, True)
    pe_dir = pe_dir

    rgb_pred, sig_pred = model(pe_pts, pe_dir)
    rgb_pred = rgb_pred.view(BATCH_RAYS, N_SAMPLES, 3)
    sig_pred = sig_pred.view(BATCH_RAYS, N_SAMPLES, 1)

    comp_rgb, _ = volume_render(rgb_pred, sig_pred, t_vals, rd)
    comp_rgb = comp_rgb.squeeze(0)   
    target = img.reshape(-1, 3)[idxs] / 255.0

    loss = F.mse_loss(comp_rgb, target)
    opt.zero_grad()
    loss.backward()
    opt.step()

    if i % LOG_INTERVAL == 0:
        mse    = loss.detach()
        v_psnr = psnr(mse)
        writer.add_scalar("train/mse", mse, i)
        writer.add_scalar("train/psnr", v_psnr, i)
        print(f"iter {i:06d}  loss={mse:.6f}  psnr={v_psnr:.2f}")

        if v_psnr > best_psnr:
            best_psnr = v_psnr
            ckpt = {
                "iter": i,
                "model": model.state_dict(),
                "opt":   opt.state_dict(),
                "best_psnr": best_psnr,
            }
            torch.save(ckpt, os.path.join(CKPT_DIR, "nerf_best.pth"))
            print(f"[Best Model Saved] iter {i:06d}, psnr={best_psnr:.2f}")

    if i % (LOG_INTERVAL * 50) == 0:
        ckpt = {
            "iter": i,
            "model": model.state_dict(),
            "opt":   opt.state_dict(),
            "best_psnr": best_psnr,
        }
        torch.save(ckpt, os.path.join(CKPT_DIR, f"nerf_ckpt_{i:06d}.pth"))
        print(f"[Checkpoint Saved] iter {i:06d}")
        
writer.close()
