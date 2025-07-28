import torch
from PIL import Image
from nerf.render import get_rays, sample_points, volume_render
from nerf.encoding import positional_encoding
from nerf.model import NeRF

# --- Setup (Same as before) ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device.type}")

ckpt = torch.load("checkpoints/nerf_ckpt_020000.pth", map_location=device)
model = NeRF().to(device).eval()
model.load_state_dict(ckpt["model"])

H, W = 400, 600
intr = {'params': torch.tensor([800.0, 800.0, W/2, H/2], device=device)}
qvec = torch.tensor([0.9239, 0.0, 0.3827, 0.0], device=device)
tvec = torch.tensor([1.2, 0.5, -4.0], device=device)

# --- Modified Rendering with Batching ---

# --- Rendering Section ---

# 1. Get all rays for the image
rays_o, rays_d = get_rays(H, W, intr, qvec, tvec)
rays_o = rays_o.reshape(-1, 3)
rays_d = rays_d.reshape(-1, 3)

# 2. Define batch size and prepare for batched rendering
batch_size = 4096
rendered_chunks = []
num_rays_original = rays_d.shape[0]

### FIX STARTS HERE: PADDING THE INPUT ###
# =================================================================

# Calculate the number of padding rays needed to make the total a multiple of batch_size
remainder = num_rays_original % batch_size
if remainder != 0:
    num_padding = batch_size - remainder
    print(f"Padding input with {num_padding} rays to ensure all batches are full.")
    # Create padding by repeating the last valid ray
    padding_o = rays_o[-1:].repeat(num_padding, 1)
    padding_d = rays_d[-1:].repeat(num_padding, 1)
    # Add the padding to the original ray tensors
    rays_o = torch.cat([rays_o, padding_o], dim=0)
    rays_d = torch.cat([rays_d, padding_d], dim=0)

# =================================================================
### FIX ENDS HERE ###

num_rays_padded = rays_d.shape[0]
print(f"Rendering {num_rays_padded} total rays in batches of {batch_size}...")

# 3. Process rays in a loop (this loop is now guaranteed to only see full batches)
for i in range(0, num_rays_padded, batch_size):
    # Get the current batch of rays
    rays_o_batch = rays_o[i:i+batch_size].to(device)
    rays_d_batch = rays_d[i:i+batch_size].to(device)

    # Sample points along rays for the batch
    pts, t_vals = sample_points(rays_o_batch, rays_d_batch, near=2.0, far=6.0, N_samples=64)

    # Positional encoding for the batch
    pe_pts = positional_encoding(pts, num_freqs=10, include_input=True)
    pe_dir = positional_encoding(rays_d_batch, num_freqs=4, include_input=True)

    N_samples = pts.shape[1]
    pe_dir = pe_dir.repeat_interleave(N_samples, dim=0)
    
    # Flatten the points tensor to be 2D, matching the directions tensor
    pe_pts_flat = pe_pts.reshape(-1, pe_pts.shape[-1])

    # Run the model forward pass on the batch
    with torch.no_grad():
        rgb, sig = model(pe_pts_flat, pe_dir)
        num_rays_in_batch = rays_o_batch.shape[0]
        rgb = rgb.view(num_rays_in_batch, N_samples, 3)
        sig = sig.view(num_rays_in_batch, N_samples, 1)
        comp_batch, _ = volume_render(rgb, sig, t_vals, rays_d_batch)
        rendered_chunks.append(comp_batch.cpu())

print("Batch rendering complete.")

# 4. Combine the rendered chunks
img_flat_padded = torch.cat(rendered_chunks, dim=0)

# 5. IMPORTANT: Truncate the tensor to remove the padded results
# The variable 'num_rays_original' should hold the original ray count (240000)
img_flat = img_flat_padded[:num_rays_original]

# 6. Now, reshape the correctly-sized tensor and save the image
img = img_flat.view(H, W, 3)
img_np = (img.clamp(0.0, 1.0).numpy() * 255).astype("uint8")
Image.fromarray(img_np).save("novel_view.png")

print("Saved novel_view.png")