import os
import torch
from PIL import Image
import numpy as np
from tqdm import tqdm
from nerf.render import get_rays, sample_points, volume_render, sample_pdf
from nerf.encoding import positional_encoding
from nerf.model import NeRF
from utils.config import *
from utils.helpers import load_checkpoint
from nerf.dataset import load_colmap_dataset

def load_novel_view_data(data_dir, img_dir):
    dataset = load_colmap_dataset(data_dir, img_dir)
    poses = [(torch.from_numpy(d['qvec']).float(), torch.from_numpy(d['tvec']).float()) for d in dataset]

    cam = dataset[0]['intrinsics']
    if isinstance(cam, dict):
        if 'params' in cam:
            fx, fy, cx, cy = cam['params'][:4]
        else:
            fx, fy, cx, cy = cam.get('fx'), cam.get('fy'), cam.get('cx'), cam.get('cy')
        intrinsics = torch.tensor([[fx, 0, cx],[0, fy, cy],[0,0,1]], dtype=torch.float32)
    else:
        intrinsics = torch.from_numpy(np.array(cam, dtype=np.float32))

    H, W = dataset[0]['image'].shape[:2]
    return poses, intrinsics, H, W

def render_novel_view(model_coarse, model_fine, H, W, intrinsics, qvec, tvec, chunk_size=1024, device="cpu"):
    # Prepare COLMAP-style intrinsics dict
    fx, fy = intrinsics[0,0].item(), intrinsics[1,1].item()
    cx, cy = intrinsics[0,2].item(), intrinsics[1,2].item()
    cam_dict = {'params': [fx, fy, cx, cy]}

    # Generate rays
    rays_o, rays_d = get_rays(H, W, cam_dict, qvec.to(device), tvec.to(device))
    rays_o, rays_d = rays_o.reshape(-1,3), rays_d.reshape(-1,3)
    rgb_chunks = []

    total_chunks = (rays_o.shape[0] + chunk_size - 1) // chunk_size
    # No gradient needed for inference
    with torch.no_grad():
        for i in tqdm(range(0, rays_o.shape[0], chunk_size), desc="Rendering chunks", total=total_chunks):
            ro = rays_o[i:i+chunk_size].to(device)
            rd = rays_d[i:i+chunk_size].to(device)

            # Coarse pass
            pts_co, t_co = sample_points(ro, rd, NEAR, FAR, N_SAMPLES)
            pe_co  = positional_encoding(pts_co.reshape(-1,3), FREQ_POS)
            pe_dir = positional_encoding(rd.unsqueeze(1).expand(-1, N_SAMPLES,3).reshape(-1,3), FREQ_DIR)
            rgb_co, sig_co = model_coarse(pe_co, pe_dir)
            rgb_co = rgb_co.view(-1, N_SAMPLES,3); sig_co = sig_co.view(-1, N_SAMPLES,1)
            comp_co, _, weights = volume_render(rgb_co, sig_co, t_co, rd)

            # Fine sampling
            bins = t_co.unsqueeze(0).expand(weights.shape[0], -1)
            t_fine = sample_pdf(bins, weights, N_IMPORTANCE)
            t_all = torch.sort(torch.cat([bins, t_fine],dim=-1), dim=-1)[0]

            # Fine pass
            pts_fi = ro.unsqueeze(1) + rd.unsqueeze(1)*t_all[...,None]
            pe_fi   = positional_encoding(pts_fi.reshape(-1,3), FREQ_POS)
            pe_dir_fi = positional_encoding(rd.unsqueeze(1).expand(-1,t_all.shape[-1],3).reshape(-1,3), FREQ_DIR)
            rgb_fi, sig_fi = model_fine(pe_fi, pe_dir_fi)
            rgb_fi = rgb_fi.view(-1,t_all.shape[-1],3); sig_fi = sig_fi.view(-1,t_all.shape[-1],1)
            comp_fi, _, _ = volume_render(rgb_fi, sig_fi, t_all, rd)

            rgb_chunks.append(comp_fi.cpu())

    rgb_map = torch.cat(rgb_chunks,0).reshape(H,W,3).numpy()
    return rgb_map

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_coarse = NeRF().to(device); model_fine = NeRF().to(device)
    ckpt = torch.load(f"{CKPT_DIR}/nerf_hierarchical.pth", map_location=device)
    model_coarse.load_state_dict(ckpt['coarse']); model_fine.load_state_dict(ckpt['fine'])
    model_coarse.eval(); model_fine.eval()

    poses, intrinsics, H, W = load_novel_view_data(DATA_DIR, IMG_DIR)
    qvec, tvec = poses[0]

    img = render_novel_view(model_coarse, model_fine, H, W, intrinsics, qvec, tvec, device=device)
    img = (255 * np.clip(img,0,1)).astype(np.uint8)
    os.makedirs("renders", exist_ok=True)
    Image.fromarray(img).save("renders/view_single.png")
    print(f"Saved one view (chunk_size={1024}) to renders/view_single.png")
