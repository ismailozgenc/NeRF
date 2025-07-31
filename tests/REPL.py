from nerf.dataset import load_colmap_dataset
DATA_DIR = "data/south-building/sparse"
IMG_DIR  = "data/south-building/images"

data = load_colmap_dataset(DATA_DIR, IMG_DIR)
print(f"Loaded {len(data)} images")

# Inspect first item
first = data[0]

print("\n -------TEST RAY GENERATION-------")
import torch
from nerf.render import get_rays
H, W = first["image"].shape[:2]
intr = first["intrinsics"]
qvec = torch.from_numpy(first["qvec"]).float()
tvec = torch.from_numpy(first["tvec"]).float()
rays_o, rays_d = get_rays(H, W, intr, qvec, tvec)
print("rays_o shape:", rays_o.shape)  # expect (H, W, 3)
print("rays_d shape:", rays_d.shape)  # expect (H, W, 3)
print("rays_d norm stats:", rays_d.norm(dim=-1).min(), rays_d.norm(dim=-1).max())
print("\n")


print("-------TEST POINT SAMPLING-------")
from nerf.render import sample_points
# Flatten
ro = rays_o.reshape(-1, 3)[:1024]   
rd = rays_d.reshape(-1, 3)[:1024]
pts, t_vals = sample_points(ro, rd, near=2.0, far=6.0, N_samples=64)
print("pts shape:", pts.shape)      # expect (1024, 64, 3)
print("t_vals shape:", t_vals.shape)  # expect (64,)
print("t_vals slice:", t_vals[:5], "...", t_vals[-1])
print("\n")

print("-------TEST POSITIONAL ENCODING-------")
from nerf.encoding import positional_encoding
pts_flat = pts.reshape(-1, 3)
pe_pts = positional_encoding(pts_flat, num_freqs=10, include_input=True)
print("pe_pts shape:", pe_pts.shape)
# expected channels = 3 * (1 + 2*10) = 63 → (1024*64, 63)
print("first row:", pe_pts[0])
print("\n")


print("-------TEST FORWARD PASS-------")
from nerf.model import NeRF
device = torch.device("cpu")
model = NeRF().to(device).eval()
# Generate a tiny test batch:
test_pts = torch.randn(128, pe_pts.shape[-1]).to(device)
test_dir = torch.randn(128, 3*(1+2*4)).to(device)
with torch.no_grad():
    rgb, sigma = model(test_pts, test_dir)
print("rgb shape:", rgb.shape)      # expect (128, 3)
print("sigma shape:", sigma.shape)  # expect (128, 1)
print("rgb range:", rgb.min().item(), rgb.max().item())
print("sigma stats:", sigma.mean().item(), sigma.std().item())
print("\n")


print("-------TEST VOLUME RENDERER-------")
from nerf.render import volume_render
batch = 64
rgb_dummy = torch.rand(batch, 64, 3)
sigma_dummy = torch.rand(batch, 64, 1)
t_vals = torch.linspace(2.0, 6.0, 64)
comp_rgb, depth = volume_render(rgb_dummy, sigma_dummy, t_vals, rd[:batch])
print("comp_rgb shape:", comp_rgb.shape)  # expect (batch, 3)
print("depth shape:", depth.shape)        # expect (batch,)
print("comp_rgb sample:", comp_rgb[0])
print("depth stats:", depth.min().item(), depth.max().item())
print("\n")
