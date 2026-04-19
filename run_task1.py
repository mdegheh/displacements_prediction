import torch
from torch.utils.data import DataLoader
from preprocess import load_dataset, make_splits, ShapeRegressionDataset
from task1_model import Task1ShapeRegressor
from train import train_one_epoch, validate, evaluate_final_metrics

batch_size = 8
lr = 5e-4
num_epochs = 25


dataset_images, dataset_landmarks = load_dataset("data/JSRT_img_lms.pth")
idx_train, idx_val = make_splits(num_samples=len(dataset_images))

train_dataset = ShapeRegressionDataset(dataset_images, dataset_landmarks, idx_train, noise_std=5.0)
val_dataset = ShapeRegressionDataset(dataset_images, dataset_landmarks, idx_val, noise_std=5.0)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Task1ShapeRegressor(hidden_dim=128).to(device)

# Only MLP trains if DINO is frozen
optimizer = torch.optim.Adam(model.mlp.parameters(), lr=lr)

for epoch in range(num_epochs):
    train_loss, train_tre = train_one_epoch(model, train_loader, optimizer, device)
    val_loss, val_tre = validate(model, val_loader, device)

    print(
        f"Epoch {epoch+1:02d} | "
        f"train_loss={train_loss:.4f} train_TRE={train_tre:.4f} | "
        f"val_loss={val_loss:.4f} val_TRE={val_tre:.4f}"
    )

final_metrics = evaluate_final_metrics(model, val_loader, device)

print("\nFinal validation metrics:")
print(f"TRE: {final_metrics['tre']:.4f}")
print(f"Pred smoothness: {final_metrics['pred_smoothness']:.4f}")
print(f"GT smoothness: {final_metrics['gt_smoothness']:.4f}")