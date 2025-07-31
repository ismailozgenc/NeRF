import torch
from PIL import Image
from nerf.render import get_rays, sample_points, volume_render
from nerf.encoding import positional_encoding
from nerf.model import NeRF
from utils.config import * 

# --- Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device.type}")

ckpt = torch.load("checkpoints/nerf_best.pth", map_location=device)
print(ckpt["best_psnr"])
model = NeRF().to(device)
model.load_state_dict(ckpt["model"])
model.eval()

H, W = 2304, 3072
intr = {'params': torch.tensor([800.0, 800.0, W/2, H/2], device=device)}
qvec = torch.tensor([0.9239, 0.0, 0.3827, 0.0], device=device)
tvec = torch.tensor([1.2, 0.5, -4.0], device=device)

# Generating rays
rays_o, rays_d = get_rays(H, W, intr, qvec, tvec)
rays_o = rays_o.reshape(-1, 3)
rays_d = rays_d.reshape(-1, 3)

# Batched rendering
total_rays = rays_d.shape[0]
batch_size = 4096 * 4
rendered_chunks = []
print(f"Rendering {total_rays} rays in batches of {batch_size}...")

for i in range(0, total_rays, batch_size):
    ro_batch = rays_o[i:i+batch_size].to(device)
    rd_batch = rays_d[i:i+batch_size].to(device)

    pts, t_vals = sample_points(ro_batch, rd_batch, NEAR, FAR, N_SAMPLES)
    pts_flat    = pts.reshape(-1, 3)

    pe_pts = positional_encoding(pts_flat, 10, True)
    pe_dir = positional_encoding(rd_batch, 4, True)

    with torch.no_grad():
        rgb, sig = model(pe_pts, pe_dir.repeat_interleave(64, dim=0))
        rgb = rgb.view(-1, 64, 3)
        sig = sig.view(-1, 64, 1)
        comp, _ = volume_render(rgb, sig, t_vals, rd_batch)
        rendered_chunks.append(comp.cpu())

print("Batch rendering complete.")

img_flat = torch.cat(rendered_chunks, dim=0)[:H*W]
img = img_flat.view(H, W, 3)
img_np = (img.clamp(0.0, 1.0).numpy() * 255).astype("uint8")
Image.fromarray(img_np).save(f"novel_view PSNR{ckpt["best_psnr"]:.1f}.png")
print("Saved novel_view.png")