import torch
import torch.nn as nn
import torch.nn.functional as F


def normalize_coords(coords, H, W):
    """
    returns: (B, N, 2) in [-1, 1]
    """
    x = coords[..., 0] / (W - 1) * 2 - 1
    y = coords[..., 1] / (H - 1) * 2 - 1
    return torch.stack([x, y], dim=-1)

def sample_features_at_landmarks(feature_map, landmark_coords_norm):
    """
    feature_map: (B, C, Hf, Wf)
    landmark_coords_norm: (B, N, 2) in [-1, 1]
    """
    B, N, _ = landmark_coords_norm.shape

    # grid_sample expects (B, H_out, W_out, 2)
    grid = landmark_coords_norm.unsqueeze(2)  # (B, N, 1, 2)

    sampled = F.grid_sample(
        feature_map,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )  # (B, C, N, 1)

    sampled = sampled.squeeze(-1).permute(0, 2, 1)  # (B, N, C)
    return sampled