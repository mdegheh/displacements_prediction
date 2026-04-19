import torch
import torch.nn as nn
import torch.nn.functional as F
from features import DINOFeatureExtractor
from sampling import normalize_coords, sample_features_at_landmarks
from models import LandmarkTransformerRegressor


class Task2ShapeRegressor(nn.Module):
    def __init__(
        self,
        feature_dim=768,
        embed_dim=256,
        num_heads=8,
        num_layers=4,
        ff_dim=512,
        dropout=0.1,
        backbone_input_size=252,
    ):
        super().__init__()

        self.feature_extractor = DINOFeatureExtractor(
            model_name="dinov2_vitb14",
            freeze=True,
        )

        self.regressor = LandmarkTransformerRegressor(
            in_dim=feature_dim,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            ff_dim=ff_dim,
            dropout=dropout,
        )

        self.backbone_input_size = backbone_input_size

    def forward(self, images, init_landmarks_px):
        """
        images: (B, 1, H, W)
        init_landmarks_px: (B, N, 2) in original image pixel coordinates
        """
        B, _, H, W = images.shape

        # Resize image for DINO (patch size 14 compatibility)
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

        # feature extraction
        feature_map = self.feature_extractor(images_resized)   # (B, C, Hf, Wf)
        _, C, Hf, Wf = feature_map.shape

        init_landmarks_feat = init_landmarks_resized.clone()
        init_landmarks_feat[..., 0] = init_landmarks_feat[..., 0] * (Wf / self.backbone_input_size)
        init_landmarks_feat[..., 1] = init_landmarks_feat[..., 1] * (Hf / self.backbone_input_size)

        # Normalize for grid_sample
        init_landmarks_norm = normalize_coords(init_landmarks_feat, Hf, Wf)

        # Sample per-landmark features
        sampled_features = sample_features_at_landmarks(feature_map, init_landmarks_norm)  # (B, N, C)

        # Context-aware displacement prediction
        pred_disp_feat = self.regressor(sampled_features, init_landmarks_norm)  # (B, N, 2)

        # Convert feature-map displacement back to original pixel scale
        pred_disp_px = pred_disp_feat.clone()
        pred_disp_px[..., 0] = pred_disp_px[..., 0] * (W / Wf)
        pred_disp_px[..., 1] = pred_disp_px[..., 1] * (H / Hf)

        pred_landmarks_px = init_landmarks_px + pred_disp_px
        return pred_landmarks_px, pred_disp_px