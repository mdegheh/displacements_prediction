import torch
from torch.utils.data import Dataset


def generate_init_shape(gt_landmarks, height, width, noise_std=5.0):
    """
    Generate an initial shape close to the ground truth.
    """
    noise = torch.randn_like(gt_landmarks) * noise_std
    init_shape = gt_landmarks + noise

    init_shape[..., 0] = init_shape[..., 0].clamp(0, width - 1)   # x
    init_shape[..., 1] = init_shape[..., 1].clamp(0, height - 1)  # y
    return init_shape


def compute_displacement_target(gt_landmarks, init_landmarks):
    return gt_landmarks - init_landmarks


def tre(pred_landmarks, gt_landmarks):
    return torch.norm(pred_landmarks - gt_landmarks, dim=-1).mean()

def shape_smoothness(landmarks):
    """
    landmarks: (B, N, 2) or (N, 2)

    Measures contour regularity using second differences.
    Lower = smoother / more regular shape.
    """
    if landmarks.ndim == 2:
        landmarks = landmarks.unsqueeze(0)  # (1, N, 2)

    second_diff = landmarks[:, 2:, :] - 2 * landmarks[:, 1:-1, :] + landmarks[:, :-2, :]
    smoothness = torch.norm(second_diff, dim=-1).mean()
    return smoothness


def load_dataset(path):
    data = torch.load(path, map_location="cpu")
    images = data["JSRT_img1"]   # (247, 256, 256)
    landmarks = data["JSRT_lms"] # (247, 166, 2)

    if images.ndim == 3:
        images = images.unsqueeze(1)  # (N, 1, H, W)

    return images.float(), landmarks.float()


def make_splits(num_samples=247):
    torch.manual_seed(0)
    idx_train, idx_val = torch.randperm(num_samples).split_with_sizes([215, 32])
    return idx_train, idx_val


class ShapeRegressionDataset(Dataset):
    def __init__(self, images, landmarks, indices, noise_std=5.0):
        self.images = images[indices]
        self.landmarks = landmarks[indices]
        self.noise_std = noise_std

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]             # (1, H, W)
        gt_landmarks = self.landmarks[idx]   # (166, 2)

        if image.ndim == 2:
            image = image.unsqueeze(0)

        _, H, W = image.shape

        init_landmarks = generate_init_shape(
            gt_landmarks, H, W, noise_std=self.noise_std
        )

        target_displacement = compute_displacement_target(
            gt_landmarks, init_landmarks
        )

        return {
            "image": image,
            "gt_landmarks": gt_landmarks,
            "init_landmarks": init_landmarks,
            "target_displacement": target_displacement,
        }