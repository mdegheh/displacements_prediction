import torch
from torch.utils.data import DataLoader

from preprocess import load_dataset, make_splits, ShapeRegressionDataset
from task3_model import Task3ShapeRegressor
from train import train_one_epoch, validate, evaluate_final_metrics

batch_size = 8
lr = 2e-5
num_epochs = 50

images, landmarks = load_dataset("data/JSRT_img_lms.pth")
idx_train, idx_val = make_splits(num_samples=len(images))

train_dataset = ShapeRegressionDataset(images, landmarks, idx_train, noise_std=5.0)
val_dataset = ShapeRegressionDataset(images, landmarks, idx_val, noise_std=5.0)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Task3ShapeRegressor(
    feature_dim=768,
    hidden_dim=128,
    num_hidden_layers=4,
    backbone_input_size=252,
    first_omega_0=30.0,
    hidden_omega_0=30.0,
).to(device)

# DINO stays frozen; only SIREN regressor trains
optimizer = torch.optim.Adam(model.regressor.parameters(), lr=lr)

# Optional forward sanity check
batch = next(iter(train_loader))
batch_images = batch["image"].to(device)
batch_init_landmarks = batch["init_landmarks"].to(device)

pred_landmarks, pred_disp = model(batch_images, batch_init_landmarks)

print("images:", batch_images.shape)
print("init_landmarks:", batch_init_landmarks.shape)
print("pred_landmarks:", pred_landmarks.shape)
print("pred_disp:", pred_disp.shape)

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