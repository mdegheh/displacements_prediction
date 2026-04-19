import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


class LocalMLP(nn.Module):
    def __init__(self, in_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_dim, hidden_dim, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, 2, kernel_size=1),
        )

    def forward(self, x):
        """
        x: (B, N, C)
        returns: (B, N, 2)
        """
        x = x.permute(0, 2, 1)   # (B, C, N)
        out = self.net(x)        # (B, 2, N)
        return out.permute(0, 2, 1)  # (B, N, 2)


class LandmarkTransformerRegressor(nn.Module):
    def __init__(
        self,
        in_dim,
        embed_dim=256,
        num_heads=8,
        num_layers=4,
        ff_dim=512,
        dropout=0.1,
    ):
        super().__init__()

        self.feature_proj = nn.Linear(in_dim, embed_dim)
        self.coord_proj = nn.Linear(2, embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, 2),
        )

    def forward(self, sampled_features, landmark_coords_norm):
        """
        sampled_features: (B, N, C)
        landmark_coords_norm: (B, N, 2) in [-1, 1]
        """
        feat_tokens = self.feature_proj(sampled_features)   # (B, N, D)
        coord_tokens = self.coord_proj(landmark_coords_norm) # (B, N, D)

        tokens = feat_tokens + coord_tokens
        tokens = self.transformer(tokens)                  # (B, N, D)

        displacement = self.head(tokens)                   # (B, N, 2)
        return displacement


class SineLayer(nn.Module):
    def __init__(self, in_features, out_features, is_first=False, omega_0=30.0):
        super().__init__()
        self.in_features = in_features
        self.is_first = is_first
        self.omega_0 = omega_0
        self.linear = nn.Linear(in_features, out_features)
        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            if self.is_first:
                bound = 1 / self.in_features
            else:
                bound = math.sqrt(6 / self.in_features) / self.omega_0
            self.linear.weight.uniform_(-bound, bound)
            self.linear.bias.uniform_(-bound, bound)

    def forward(self, x):
        return torch.sin(self.omega_0 * self.linear(x))


class ConditionalSIRENRegressor(nn.Module):
    """
    Learns a continuous displacement function:
        f(x, y | z) -> (dx, dy)

    where:
        - (x, y): landmark coordinate
        - z: global conditioning vector
    """

    def __init__(
        self,
        coord_dim=2,
        cond_dim=768,
        hidden_dim=256,
        num_hidden_layers=3,
        first_omega_0=30.0,
        hidden_omega_0=30.0,
    ):
        super().__init__()

        input_dim = coord_dim + cond_dim

        layers = [
            SineLayer(
                in_features=input_dim,
                out_features=hidden_dim,
                is_first=True,
                omega_0=first_omega_0,
            )
        ]

        for _ in range(num_hidden_layers):
            layers.append(
                SineLayer(
                    in_features=hidden_dim,
                    out_features=hidden_dim,
                    is_first=False,
                    omega_0=hidden_omega_0,
                )
            )

        self.siren = nn.Sequential(*layers)
        self.head = nn.Linear(hidden_dim, 2)

    def forward(self, landmark_coords_norm, global_condition):
        """
        landmark_coords_norm: (B, N, 2)
        global_condition:     (B, C)

        returns:
            displacement: (B, N, 2)
        """
        B, N, _ = landmark_coords_norm.shape

        cond = global_condition.unsqueeze(1).expand(-1, N, -1)   # (B, N, C)
        x = torch.cat([landmark_coords_norm, cond], dim=-1)      # (B, N, 2+C)

        x = self.siren(x)                                         # (B, N, H)
        displacement = self.head(x)                               # (B, N, 2)
        return displacement