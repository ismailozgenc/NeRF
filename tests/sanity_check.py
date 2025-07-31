import torch
from nerf.dataset import load_colmap_dataset
from nerf.render import get_rays, sample_points, volume_render
from nerf.encoding import positional_encoding
from nerf.model import NeRF
from utils.config import DATA_DIR, IMG_DIR, NEAR, FAR, N_SAMPLES, FREQ_POS, FREQ_DIR
import importlib, nerf.render
importlib.reload(nerf.render)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Load one example
data = load_colmap_dataset(DATA_DIR, IMG_DIR)
item = data[0]

# 2. Prepare rays
H, W = item["image"].shape[:2]
intr = item["intrinsics"]
qvec = torch.from_numpy(item["qvec"]).float().to(device)
tvec = torch.from_numpy(item["tvec"]).float().to(device)
rays_o, rays_d = get_rays(H, W, intr, qvec, tvec)
rays_o = rays_o.reshape(-1,3)[:256]
rays_d = rays_d.reshape(-1,3)[:256]

# 3. Sample points and encode
pts, t_vals = sample_points(rays_o, rays_d, NEAR, FAR, N_SAMPLES)
print("t_vals sample:", t_vals[:5].tolist(), "...", t_vals[-1].item())

pts_flat    = pts.reshape(-1, 3)
pe_pts      = positional_encoding(pts_flat, FREQ_POS, True)
pe_dir      = positional_encoding(rays_d, FREQ_DIR, True).repeat_interleave(N_SAMPLES, dim=0)

# 4. Instantiate model and load weights
model = NeRF().to(device).eval()
ckpt = torch.load("checkpoints/nerf_best.pth", map_location=device)
model.load_state_dict(ckpt["model"])
model.eval()

# 5. Forward + render
with torch.no_grad():
    rgb_pred, sig_pred = model(pe_pts, pe_dir)
    rgb_pred = rgb_pred.view(256, N_SAMPLES, 3)
    sig_pred = sig_pred.view(256, N_SAMPLES, 1)
    comp_rgb, depth = volume_render(rgb_pred, sig_pred, t_vals, rays_d)

print("comp_rgb:", comp_rgb.shape, "depth:", depth.shape)
print("RGB range:", comp_rgb.min().item(), "–", comp_rgb.max().item())
print("Depth range:", depth.min().item(), "–", depth.max().item())
