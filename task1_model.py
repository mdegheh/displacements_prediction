import torch
import torch.nn as nn
import torch.nn.functional as F

from features import DINOFeatureExtractor
from sampling import normalize_coords, sample_features_at_landmarks
from models import LocalMLP


class Task1ShapeRegressor(nn.Module):
    def __init__(self, hidden_dim=128):
        super().__init__()
        self.feature_extractor = DINOFeatureExtractor(model_name="dinov2_vitb14", freeze=True)
        self.mlp = LocalMLP(in_dim=768, hidden_dim=hidden_dim)

        self.backbone_input_size = 252  # divisible by 14

    def forward(self, images, init_landmarks_px):
        """
        images: (B, 1, H, W)
        init_landmarks_px: (B, N, 2) in original pixel coordinates
        """
        B, _, H, W = images.shape

        # Resize images for DINO
        images_resized = F.interpolate(
            images,
            size=(self.backbone_input_size, self.backbone_input_size),
            mode="bilinear",
            align_corners=False,
        )

        # Scale landmarks to resized image coordinates
        init_landmarks_resized = init_landmarks_px.clone()
        init_landmarks_resized[..., 0] = init_landmarks_resized[..., 0] * (self.backbone_input_size / W)
        init_landmarks_resized[..., 1] = init_landmarks_resized[..., 1] * (self.backbone_input_size / H)

        # Extract dense features
        feature_map = self.feature_extractor(images_resized)  # (B, C, Hf, Wf)
        _, C, Hf, Wf = feature_map.shape

        # Map resized-image landmarks to feature-map coordinates
        init_landmarks_feat = init_landmarks_resized.clone()
        init_landmarks_feat[..., 0] = init_landmarks_feat[..., 0] * (Wf / self.backbone_input_size)
        init_landmarks_feat[..., 1] = init_landmarks_feat[..., 1] * (Hf / self.backbone_input_size)

        # Normalize for grid_sample
        init_landmarks_norm = normalize_coords(init_landmarks_feat, Hf, Wf)

        # Sample features at landmark positions
        sampled_features = sample_features_at_landmarks(feature_map, init_landmarks_norm)

        # Predict displacement in feature-map scale
        pred_disp_feat = self.mlp(sampled_features)

        # Convert displacement back to original image pixel scale
        pred_disp_px = pred_disp_feat.clone()
        pred_disp_px[..., 0] = pred_disp_px[..., 0] * (W / Wf)
        pred_disp_px[..., 1] = pred_disp_px[..., 1] * (H / Hf)

        pred_landmarks_px = init_landmarks_px + pred_disp_px
        return pred_landmarks_px, pred_disp_px