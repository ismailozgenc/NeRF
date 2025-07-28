import torch
from PIL import Image
from nerf.render import get_rays, sample_points, volume_render
from nerf.encoding import positional_encoding
from nerf.model import NeRF

# --- Setup (Same as before) ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device.type}")

ckpt = torch.load("checkpoints/nerf_best.pth", map_location=device)
model = NeRF().to(device).eval()
model.load_state_dict(ckpt["model"])

H, W = 2304, 3072
intr = {'params': torch.tensor([800.0, 800.0, W/2, H/2], device=device)}
qvec = torch.tensor([0.9239, 0.0, 0.3827, 0.0], device=device)
tvec = torch.tensor([1.2, 0.5, -4.0], device=device)

# --- Modified Rendering with Batching ---

# 1. Get all rays for the image
rays_o, rays_d = get_rays(H, W, intr, qvec, tvec)
rays_o = rays_o.reshape(-1, 3) # Shape: (H*W, 3)
rays_d = rays_d.reshape(-1, 3) # Shape: (H*W, 3)

# 2. Define batch size and prepare for batched rendering
# Adjust batch_size based on your GPU memory. 4096 is a good starting point.
batch_size = 4096
rendered_chunks = []
num_rays = rays_d.shape[0]

print(f"Rendering {num_rays} rays in batches of {batch_size}...")

# 3. Process rays in a loop
for i in range(0, num_rays, batch_size):
    # Get the current batch of rays
    rays_o_batch = rays_o[i:i+batch_size].to(device)
    rays_d_batch = rays_d[i:i+batch_size].to(device)
    
    # Sample points along rays for the batch
    pts, t_vals = sample_points(rays_o_batch, rays_d_batch, near=2.0, far=6.0, N_samples=64)
    
    # Positional encoding for points and directions
    pe_pts = positional_encoding(pts, num_freqs=10, include_input=True)  # (batch_size*N_samples, input_ch_pts)
    pe_dir = positional_encoding(rays_d_batch, num_freqs=4, include_input=True)  # (batch_size, input_ch_dir)

    N_samples = pts.shape[1]  # 64

    # Expand and flatten pe_dir to match pe_pts shape
    pe_dir_expanded = pe_dir.unsqueeze(1).expand(-1, N_samples, -1)  # (batch_size, N_samples, input_ch_dir)
    pe_dir_flat = pe_dir_expanded.reshape(-1, pe_dir.shape[-1])       # (batch_size * N_samples, input_ch_dir)

    # Run the model forward pass on the batch
    with torch.no_grad():
        rgb, sig = model(pe_pts, pe_dir_flat)
        
        # Reshape for volume rendering
        num_rays_in_batch = rays_o_batch.shape[0]
        rgb = rgb.view(num_rays_in_batch, N_samples, 3)
        sig = sig.view(num_rays_in_batch, N_samples, 1)
        # Perform volume rendering on the batch
        comp_batch, _ = volume_render(rgb, sig, t_vals, rays_d_batch)

        # Ensure shape is [num_rays_in_batch, 3]
        comp_batch = comp_batch.reshape(-1, 3).cpu()
        rendered_chunks.append(comp_batch)

print("Batch rendering complete.")

# 4. Combine the rendered chunks and reshape to the final image
img_flat = torch.cat(rendered_chunks, dim=0)[:H*W]  # just in case overshoot
img = img_flat.view(H, W, 3)


# 5. Save the final image
img_np = (img.clamp(0.0, 1.0).numpy() * 255).astype("uint8")
Image.fromarray(img_np).save("novel_view.png")
print("Saved novel_view.png")