import torch

def get_rays(height: int,
             width: int,
             intrinsics: dict,
             qvec: torch.Tensor,
             tvec: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    dtype, device = qvec.dtype, qvec.device
    intr = torch.tensor(intrinsics['params'], device=device, dtype=dtype)
    tvec = tvec.to(device=device, dtype=dtype)

    # build pixel grid of shape (height, width)
    j, i = torch.meshgrid(
        torch.linspace(0, height - 1, height, device=device, dtype=dtype),
        torch.linspace(0, width  - 1, width,  device=device, dtype=dtype),
        indexing='ij'
    )

    # camera-space directions
    dirs = torch.stack([
        (i - intr[2]) / intr[0],
        (j - intr[3]) / intr[1],
        torch.ones_like(i)
    ], dim=-1)  # (height, width, 3)

    # quaternion → rotation matrix
    qw, qx, qy, qz = qvec
    R = torch.tensor([
        [1 - 2*(qy**2 + qz**2),   2*(qx*qy - qz*qw),   2*(qx*qz + qy*qw)],
        [2*(qx*qy + qz*qw), 1 - 2*(qx**2 + qz**2),   2*(qy*qz - qx*qw)],
        [2*(qx*qz - qy*qw),   2*(qy*qz + qx*qw), 1 - 2*(qx**2 + qy**2)]
    ], device=device, dtype=dtype)

    # correct camera origin in world coordinates
    cam_center = -R.T @ tvec

    # rotate & normalize
    rays_d = (dirs[..., None, :] @ R.T).squeeze(-2)
    rays_d = rays_d / rays_d.norm(dim=-1, keepdim=True)

    # origins (same center for all rays)
    rays_o = cam_center.expand_as(rays_d)

    return rays_o, rays_d



def sample_points(rays_o: torch.Tensor,
                  rays_d: torch.Tensor,
                  near: float,
                  far: float,
                  N_samples: int) -> tuple[torch.Tensor, torch.Tensor]:
    # Linear samples between near and far for each ray
    t_vals = torch.linspace(near, far, N_samples,
                            device=rays_o.device, dtype=rays_o.dtype)
    pts = rays_o[..., None, :] + rays_d[..., None, :] * t_vals[..., :, None]
    return pts, t_vals


def volume_render(rgb, sigma, t_vals, rays_d):
    # move to correct device/dtype
    t_vals = t_vals.to(rgb.device)

    # ensure t_vals is (batch, N)
    if t_vals.dim() == 1:
        t = t_vals.unsqueeze(0).expand(rgb.shape[0], -1)
    else:
        t = t_vals

    if t_vals.dim() == 1:
        deltas = t_vals[1:] - t_vals[:-1]
        deltas = torch.cat([deltas, deltas[-1:]], dim=0).view(1, -1)
    else:
        deltas = t_vals[..., 1:] - t_vals[..., :-1]
        deltas = torch.cat([deltas, deltas[..., -1:]], dim=-1)

    alpha = 1.0 - torch.exp(-sigma.squeeze(-1) * deltas)  # (batch, N)

    trans = torch.cumprod(
        torch.cat([torch.ones_like(alpha[..., :1]), 1.0 - alpha + 1e-10], dim=-1),
        dim=-1
    )[..., :-1]  # (batch, N)
    weights = alpha * trans  # (batch, N)
    comp_rgb = (weights.unsqueeze(-1) * rgb).sum(dim=-2)  # (batch, 3)
    depth = (weights * t).sum(dim=-1)                    # (batch,)
    return comp_rgb, depth, weights


def sample_pdf(bins, weights, N_samples, det=False):
    """
    bins:   (R, N_bins) = typically t_co
    weights:(R, N_bins) = typically weights from volume_render
    returns: (R, N_samples)
    """
    # Ensure both are 2D
    if bins.dim() == 3:
        bins = bins.squeeze(0)
    if weights.dim() == 3:
        weights = weights.squeeze(0)

    assert bins.shape == weights.shape, f"bins {bins.shape}, weights {weights.shape} mismatch"

    R, B = bins.shape

    # PDF + CDF
    weights = weights + 1e-5
    pdf = weights / torch.sum(weights, dim=-1, keepdim=True)
    cdf = torch.cumsum(pdf, dim=-1)
    cdf = torch.cat([torch.zeros(R, 1, device=cdf.device), cdf], dim=-1)  # (R, B+1)

    # Draw uniform samples
    if det:
        u = torch.linspace(0., 1., N_samples, device=bins.device).unsqueeze(0).expand(R, N_samples)
    else:
        u = torch.rand(R, N_samples, device=bins.device)

    # Invert CDF
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.clamp(inds - 1, min=0)
    above = torch.clamp(inds, max=B)

    cdf_b = torch.gather(cdf, 1, below)
    cdf_a = torch.gather(cdf, 1, above)
    bins_b = torch.gather(bins, 1, below.clamp(max=B-1))
    bins_a = torch.gather(bins, 1, (above-1).clamp(max=B-1))

    denom = (cdf_a - cdf_b)
    denom[denom < 1e-5] = 1
    t = (u - cdf_b) / denom
    samples = bins_b + t * (bins_a - bins_b)

    return samples
