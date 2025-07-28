import torch
import torch.nn as nn
import torch.nn.functional as F

class NeRF(nn.Module):
    def __init__(self,
                 D: int = 8,
                 W: int = 256,
                 input_ch: int = 3 * (1 + 2 * 10),
                 input_ch_dir: int = 3 * (1 + 2 * 4),
                 skips: list[int] = [4]):
        super().__init__()
        self.D = D
        self.W = W
        self.skips = skips

        layers = []
        for i in range(D):
            in_ch = input_ch if i == 0 else W
            if i in skips:
                in_ch += input_ch
            layers.append(nn.Linear(in_ch, W))
            layers.append(nn.ReLU(True))
        self.pts_linears = nn.Sequential(*layers)

        self.sigma_layer = nn.Linear(W, 1)
        self.feature_layer = nn.Linear(W, W)

        self.dir_linear = nn.Linear(input_ch_dir + W, W // 2)
        self.rgb_layer = nn.Linear(W // 2, 3)

    def forward(self, x: torch.Tensor, d: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = x
        for i, layer in enumerate(self.pts_linears):
            if isinstance(layer, nn.Linear) and i // 2 in self.skips and i % 2 == 0:
                h = torch.cat([h, x], dim=-1)
            h = layer(h)

        sigma = self.sigma_layer(h)

        features = self.feature_layer(h)
        if features.dim() == 3:
            features = features.view(-1, features.shape[-1])  # flatten batch and samples dims

        # now d and features both (B*N, feature_dim)
        h_dir = torch.cat([features, d], dim=-1)
        h_dir = self.dir_linear(h_dir)
        h_dir = F.relu(h_dir)
        rgb = self.rgb_layer(h_dir)

        return rgb, sigma

