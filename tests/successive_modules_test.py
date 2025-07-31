import torch
import numpy as np
from nerf.encoding import positional_encoding
from nerf.render import get_rays, sample_points, volume_render
from nerf.dataset import load_colmap_dataset
from utils.config import DATA_DIR, IMG_DIR, NEAR, FAR, N_SAMPLES, BATCH_RAYS, FREQ_POS, FREQ_DIR


def check_positional_encoding_shapes():
    x = torch.randn(BATCH_RAYS * N_SAMPLES, 3)
    pe_pts = positional_encoding(x, FREQ_POS, True)
    pe_dir = positional_encoding(x, FREQ_DIR, True)
    assert pe_pts.shape == (BATCH_RAYS * N_SAMPLES, 3 * (1 + 2 * FREQ_POS)), \
        f"pe_pts shape {pe_pts.shape} != expected {(BATCH_RAYS * N_SAMPLES, 3 * (1 + 2 * FREQ_POS))}"
    assert pe_dir.shape == (BATCH_RAYS * N_SAMPLES, 3 * (1 + 2 * FREQ_DIR)), \
        f"pe_dir shape {pe_dir.shape} != expected {(BATCH_RAYS * N_SAMPLES, 3 * (1 + 2 * FREQ_DIR))}"


def check_rays():
    dataset = load_colmap_dataset(DATA_DIR, IMG_DIR)
    sample = dataset[0]
    img = sample["image"]
    H, W = img.shape[:2]
    qvec = torch.from_numpy(sample["qvec"]).float()
    tvec = torch.from_numpy(sample["tvec"]).float()
    rays_o, rays_d = get_rays(H, W, sample["intrinsics"], qvec, tvec)
    assert rays_o.shape == (H, W, 3), f"rays_o shape {rays_o.shape} != {(H, W, 3)}"
    assert rays_d.shape == (H, W, 3), f"rays_d shape {rays_d.shape} != {(H, W, 3)}"
    norms = rays_d.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5), "rays_d not normalized"


def check_t_values():
    dataset = load_colmap_dataset(DATA_DIR, IMG_DIR)
    sample = dataset[0]
    H, W = sample["image"].shape[:2]
    qvec = torch.from_numpy(sample["qvec"]).float()
    tvec = torch.from_numpy(sample["tvec"]).float()
    rays_o, rays_d = get_rays(H, W, sample["intrinsics"], qvec, tvec)
    ro = rays_o.reshape(-1, 3)[:BATCH_RAYS]
    rd = rays_d.reshape(-1, 3)[:BATCH_RAYS]
    pts, t_vals = sample_points(ro, rd, NEAR, FAR, N_SAMPLES)
    assert t_vals.ndim == 1 and t_vals.shape[0] == N_SAMPLES, \
        f"t_vals shape {t_vals.shape} incorrect"
    assert torch.all(t_vals >= NEAR) and torch.all(t_vals <= FAR), \
        "t_vals out of [NEAR, FAR] bounds"
    deltas = t_vals[1:] - t_vals[:-1]
    expected = (FAR - NEAR) / (N_SAMPLES - 1)
    assert np.isclose(deltas.mean().item(), expected, rtol=1e-3), \
        f"mean delta {deltas.mean().item()} != expected {expected}"
    assert deltas.std().item() < 1e-6, f"deltas not uniform, std={deltas.std().item()}"


def check_volume_render():
    # Simulate a batch
    ro = torch.randn(BATCH_RAYS, 3)
    rd = torch.randn(BATCH_RAYS, 3)
    pts, t_vals = sample_points(ro, rd, NEAR, FAR, N_SAMPLES)

    # Fake predictions
    rgb = torch.sigmoid(torch.randn(BATCH_RAYS, N_SAMPLES, 3))
    sigma = torch.relu(torch.randn(BATCH_RAYS, N_SAMPLES, 1))

    # Run volume_render
    comp_rgb, depth = volume_render(rgb, sigma, t_vals, rd)

    # Recompute deltas with padding like volume_render
    deltas = t_vals[1:] - t_vals[:-1]
    deltas = torch.cat([deltas, deltas[-1:]], dim=0)  # (N_SAMPLES,)

    # Recompute weights
    alpha = 1.0 - torch.exp(-sigma.squeeze(-1) * deltas.view(1, -1))  # (BATCH_RAYS, N_SAMPLES)
    trans = torch.cumprod(torch.cat([
        torch.ones((BATCH_RAYS, 1)),
        1.0 - alpha + 1e-10
    ], dim=1), dim=1)[..., :-1]  # (BATCH_RAYS, N_SAMPLES)
    weights = alpha * trans
    w_sum = weights.sum(dim=-1)

    assert torch.all(w_sum <= 1.0 + 1e-6), f"weights sum >1: {w_sum.max().item()}"
    assert torch.all(w_sum > 0), "all weights are zero"


def check_image_scaling():
    dataset = load_colmap_dataset(DATA_DIR, IMG_DIR)
    sample = dataset[0]
    img = torch.from_numpy(sample["image"]).float()
    target = img.reshape(-1, 3) / 255.0
    assert target.min() >= 0.0 and target.max() <= 1.0, \
        f"target values out of range [0,1]: min={target.min().item()}, max={target.max().item()}"

if __name__ == "__main__":
    check_positional_encoding_shapes()
    check_rays()
    check_t_values()
    check_volume_render()
    check_image_scaling()
    print("All sanity checks passed")