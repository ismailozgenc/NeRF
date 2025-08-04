import os
import torch

def load_checkpoint(model_coarse, model_fine, optimizer, ckpt_path, device):
    """Load checkpoint with both coarse and fine models."""
    if not os.path.exists(ckpt_path):
        return 0, float('-inf')
    ckpt = torch.load(ckpt_path, map_location=device)
    model_coarse.load_state_dict(ckpt["coarse"])
    model_fine.load_state_dict(ckpt["fine"])
    optimizer.load_state_dict(ckpt["opt"])
    start_iter = ckpt["iter"] + 1
    best_psnr  = ckpt.get("best_psnr", float('-inf'))
    return start_iter, best_psnr

def save_checkpoint(model_coarse, model_fine, optimizer, best_psnr, iteration, ckpt_path):
    """Save checkpoint with both coarse and fine models."""
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    torch.save({
        "iter":      iteration,
        "coarse":    model_coarse.state_dict(),
        "fine":      model_fine.state_dict(),
        "opt":       optimizer.state_dict(),
        "best_psnr": best_psnr,
    }, ckpt_path)
