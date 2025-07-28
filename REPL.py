import torch
from nerf.render import get_rays, sample_points, volume_render
from nerf.encoding import positional_encoding
from nerf.model import NeRF

H, W = 100, 150
intr = {'params': torch.tensor([150.0, 150.0, W/2, H/2])}
qvec = torch.tensor([1.0, 0.0, 0.0, 0.0])
tvec = torch.zeros(3)

rays_o, rays_d = get_rays(H, W, intr, qvec, tvec)
pts, t_vals = sample_points(rays_o, rays_d, 2.0, 6.0, N_samples=32)

net = NeRF()
N = pts.shape[2]

pe_pts = positional_encoding(pts.reshape(-1, 3), num_freqs=10, include_input=True)
pe_dirs = positional_encoding(rays_d.reshape(-1, 3), num_freqs=4, include_input=True)
pe_dirs = pe_dirs.repeat_interleave(N, dim=0)

rgb, sig = net(pe_pts, pe_dirs)
rgb = rgb.view(*([1, H, W, N, 3][1:])) if rgb.ndim == 5 else rgb.view(H, W, N, 3)
sig = sig.view(*([1, H, W, N, 1][1:])) if sig.ndim == 5 else sig.view(H, W, N, 1)

comp_rgb, depth = volume_render(rgb, sig, t_vals, rays_d)

if comp_rgb.dim() == 4:
    comp_rgb = comp_rgb.squeeze(0)
if depth.dim() == 3:
    depth = depth.squeeze(0)

print(comp_rgb.shape, depth.shape)
