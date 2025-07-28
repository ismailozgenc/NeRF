import torch

def get_rays(height: int,
             width: int,
             intrinsics: dict,
             qvec: torch.Tensor,
             tvec: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    dtype = qvec.dtype
    device = qvec.device
    intr = torch.tensor(intrinsics['params'], device=device, dtype=dtype)
    tvec = tvec.to(device=device, dtype=dtype)

    i, j = torch.meshgrid(
        torch.linspace(0, width - 1, width, device=device, dtype=dtype),
        torch.linspace(0, height - 1, height, device=device, dtype=dtype),
        indexing='xy'
    )

    dirs = torch.stack([
        (i - intr[2]) / intr[0],
        (j - intr[3]) / intr[1],
        torch.ones_like(i)
    ], dim=-1)

    qw, qx, qy, qz = qvec
    R = torch.tensor([
        [1 - 2*(qy**2 + qz**2),   2*(qx*qy - qz*qw),   2*(qx*qz + qy*qw)],
        [  2*(qx*qy + qz*qw), 1 - 2*(qx**2 + qz**2),   2*(qy*qz - qx*qw)],
        [  2*(qx*qz - qy*qw),   2*(qy*qz + qx*qw), 1 - 2*(qx**2 + qy**2)]
    ], device=device, dtype=dtype)

    rays_d = (dirs[..., None, :] @ R.T).squeeze(-2)
    rays_o = tvec.expand_as(rays_d)
    return rays_o, rays_d

def sample_points(rays_o: torch.Tensor,
                  rays_d: torch.Tensor,
                  near: float,
                  far: float,
                  N_samples: int) -> tuple[torch.Tensor, torch.Tensor]:
    t_vals = torch.linspace(near, far, N_samples, device=rays_o.device, dtype=rays_o.dtype)
    pts = rays_o[..., None, :] + rays_d[..., None, :] * t_vals[..., :, None]
    return pts, t_vals

def volume_render(rgb: torch.Tensor,
                  sigma: torch.Tensor,
                  t_vals: torch.Tensor,
                  dirs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    deltas = t_vals[1:] - t_vals[:-1]
    deltas = torch.cat([deltas, deltas[-1:]], dim=0)
    deltas = deltas.view(*((1,) * (rgb.ndim - 1)), -1)

    alpha = 1.0 - torch.exp(-sigma.squeeze(-1) * deltas)
    trans = torch.cumprod(torch.cat([
        torch.ones_like(alpha[..., :1]),
        1.0 - alpha + 1e-10
    ], dim=-1), dim=-1)[..., :-1]
    weights = alpha * trans

    comp_rgb = (weights.unsqueeze(-1) * rgb).sum(dim=-2)
    depth = (weights * t_vals).sum(dim=-1)
    return comp_rgb, depth
