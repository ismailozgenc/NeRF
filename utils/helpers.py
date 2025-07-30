import os
import torch

def load_checkpoint(model, optimizer,  ckpt_path, device):
    """Loads model, optimizer, scheduler (if present); returns start_iter & best_psnr."""
    if not os.path.exists(ckpt_path):
        return 1, float('-inf')
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["opt"])
    start_iter = ckpt["iter"] + 1
    best_psnr  = ckpt.get("best_psnr", float('-inf'))
    return start_iter, best_psnr

def save_checkpoint(model, optimizer,  best_psnr, iteration, ckpt_path):
    """Saves model, optimizer, best_psnr, and iteration."""
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    torch.save({
        "iter":      iteration,
        "model":     model.state_dict(),
        "opt":       optimizer.state_dict(),
        "best_psnr": best_psnr,
    }, ckpt_path)
