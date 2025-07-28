import torch

def positional_encoding(x: torch.Tensor,
                        num_freqs: int,
                        include_input: bool = True) -> torch.Tensor:
    dims = x.shape[-1]
    freqs = 2.0 ** torch.arange(num_freqs, device=x.device).float()
    parts = [x] if include_input else []
    for f in freqs:
        parts.append(torch.sin(x * f))
        parts.append(torch.cos(x * f))
    return torch.cat(parts, dim=-1)
