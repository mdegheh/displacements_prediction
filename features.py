import torch
import torch.nn as nn
import torch.nn.functional as F


class DINOFeatureExtractor(nn.Module):
    def __init__(self, model_name="dinov2_vitb14", freeze=True):
        super().__init__()
        self.backbone = torch.hub.load("facebookresearch/dinov2", model_name)

        if freeze:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, x):
        """
        x: (B, 1, H, W) or (B, 3, H, W)

        returns:
            dense feature map: (B, C, Hf, Wf)
        """
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)  # grayscale -> RGB

        # DINOv2 patch tokens
        feats = self.backbone.forward_features(x)
        patch_tokens = feats["x_norm_patchtokens"]

        B, P, C = patch_tokens.shape

        # For 256x256 image and patch size 14, token grid may be 18x18
        side = int(P ** 0.5)
        assert side * side == P, f"Patch token count {P} is not square."

        feature_map = patch_tokens.transpose(1, 2).reshape(B, C, side, side)
        return feature_map