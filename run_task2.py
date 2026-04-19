import torch
from torch.utils.data import DataLoader

from preprocess import load_dataset, make_splits, ShapeRegressionDataset
from task2_model import Task2ShapeRegressor
from train import train_one_epoch, validate, evaluate_final_metrics

batch_size = 8
lr = 15e-5
num_epochs = 50

images, landmarks = load_dataset("data/JSRT_img_lms.pth")
idx_train, idx_val = make_splits(num_samples=len(images))

train_dataset = ShapeRegressionDataset(images, landmarks, idx_train, noise_std=5.0)
val_dataset = ShapeRegressionDataset(images, landmarks, idx_val, noise_std=5.0)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Task2ShapeRegressor(
    feature_dim=768,
    embed_dim=256,
    num_heads=8,
    num_layers=4,
    ff_dim=512,
    dropout=0.1,
    backbone_input_size=252,
).to(device)

optimizer = torch.optim.Adam(model.regressor.parameters(), lr=lr)

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