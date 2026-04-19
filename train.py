import torch
import torch.nn as nn
import torch.nn.functional as F
from preprocess import tre, shape_smoothness


def train_one_epoch(model, loader, optimizer, device):

    model.train()
    total_loss = 0.0
    total_tre = 0.0

    for batch in loader:
        images = batch["image"].to(device)                         # (B, 1, 256, 256)
        gt_landmarks = batch["gt_landmarks"].to(device)            # (B, 166, 2)
        init_landmarks = batch["init_landmarks"].to(device)        # (B, 166, 2)
        target_displacement = batch["target_displacement"].to(device)

        optimizer.zero_grad()

        pred_landmarks, pred_disp = model(images, init_landmarks)

        loss = F.mse_loss(pred_disp, target_displacement)
        metric = tre(pred_landmarks, gt_landmarks)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_tre += metric.item()

    return total_loss / len(loader), total_tre / len(loader)

@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    total_loss = 0.0
    total_tre = 0.0

    for batch in loader:
        images = batch["image"].to(device)
        gt_landmarks = batch["gt_landmarks"].to(device)
        init_landmarks = batch["init_landmarks"].to(device)
        target_displacement = batch["target_displacement"].to(device)

        pred_landmarks, pred_disp = model(images, init_landmarks)

        loss = F.mse_loss(pred_disp, target_displacement)
        metric = tre(pred_landmarks, gt_landmarks)

        total_loss += loss.item()
        total_tre += metric.item()

    return total_loss / len(loader), total_tre / len(loader)

@torch.no_grad()
def evaluate_final_metrics(model, loader, device):
    model.eval()

    total_tre = 0.0
    total_smoothness = 0.0
    total_gt_smoothness = 0.0

    for batch in loader:
        images = batch["image"].to(device)
        gt_landmarks = batch["gt_landmarks"].to(device)
        init_landmarks = batch["init_landmarks"].to(device)

        pred_landmarks, pred_disp = model(images, init_landmarks)

        total_tre += tre(pred_landmarks, gt_landmarks).item()
        total_smoothness += shape_smoothness(pred_landmarks).item()
        total_gt_smoothness += shape_smoothness(gt_landmarks).item()

    n = len(loader)
    return {
        "tre": total_tre / n,
        "pred_smoothness": total_smoothness / n,
        "gt_smoothness": total_gt_smoothness / n,
    }