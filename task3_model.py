import torch
import torch.nn as nn
import torch.nn.functional as F

from features import DINOFeatureExtractor
from sampling import normalize_coords, sample_features_at_landmarks
from models import ConditionalSIRENRegressor


class Task3ShapeRegressor(nn.Module):
    """
    Task 3:
    - extract dense DINO features
    - sample local features at initial landmarks
    - mean-pool sampled features to get global conditioning vector z
    - use conditional SIREN:
          f(x, y | z) -> displacement
    """

    def __init__(
        self,
        feature_dim=768,
        hidden_dim=256,
        num_hidden_layers=3,
        backbone_input_size=252,
        first_omega_0=30.0,
        hidden_omega_0=30.0,
    ):
        super().__init__()

        self.feature_extractor = DINOFeatureExtractor(
            model_name="dinov2_vitb14",
            freeze=True,
        )

        self.regressor = ConditionalSIRENRegressor(
            coord_dim=2,
            cond_dim=feature_dim,
            hidden_dim=hidden_dim,
            num_hidden_layers=num_hidden_layers,
            first_omega_0=first_omega_0,
            hidden_omega_0=hidden_omega_0,
        )

        self.backbone_input_size = backbone_input_size

    def forward(self, images, init_landmarks_px):
        """
        images: (B, 1, H, W)
        init_landmarks_px: (B, N, 2) in original image pixel coordinates

        returns:
            pred_landmarks_px: (B, N, 2)
            pred_disp_px:      (B, N, 2)
        """
        B, _, H, W = images.shape

        # Resize image for DINO compatibility (multiple of 14)
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

        # Dense DINO features
        feature_map = self.feature_extractor(images_resized)   # (B, C, Hf, Wf)
        _, C, Hf, Wf = feature_map.shape

        # Convert resized-image coords to feature-map coords
        init_landmarks_feat = init_landmarks_resized.clone()
        init_landmarks_feat[..., 0] = init_landmarks_feat[..., 0] * (Wf / self.backbone_input_size)
        init_landmarks_feat[..., 1] = init_landmarks_feat[..., 1] * (Hf / self.backbone_input_size)

        # Normalize coordinates for grid_sample and SIREN input
        init_landmarks_norm = normalize_coords(init_landmarks_feat, Hf, Wf)  # (B, N, 2)

        # Sample local DINO features at landmark positions
        sampled_features = sample_features_at_landmarks(feature_map, init_landmarks_norm)  # (B, N, C)

        # Global conditioning vector z from mean pooled landmark features
        global_condition = sampled_features.mean(dim=1)  # (B, C)

        # Predict displacement in feature-map coordinate scale
        pred_disp_feat = self.regressor(init_landmarks_norm, global_condition)  # (B, N, 2)

        # Convert displacement back to original image pixel scale
        pred_disp_px = pred_disp_feat.clone()
        pred_disp_px[..., 0] = pred_disp_px[..., 0] * (W / Wf)
        pred_disp_px[..., 1] = pred_disp_px[..., 1] * (H / Hf)

        # Refine shape
        pred_landmarks_px = init_landmarks_px + pred_disp_px

        return pred_landmarks_px, pred_disp_px