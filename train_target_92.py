"""
train_target_92.py
Optimized Training & Evaluation Pipeline for Deepfake Detection
Targeting >= 92% Test Accuracy
"""

import os
import sys
import json
import time
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from PIL import Image
import timm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace", line_buffering=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="backslashreplace", line_buffering=True)


# ============================================================
# ARCHITECTURE DEFINITION (Matches predictor.py)
# ============================================================

class ForensicHead(nn.Module):
    def __init__(self, in_features, num_classes=2, drop_rate=0.35):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        feat_dim = in_features * 2
        self.fc = nn.Sequential(
            nn.BatchNorm1d(feat_dim),
            nn.Dropout(drop_rate),
            nn.Linear(feat_dim, 512),
            nn.GELU(),
            nn.BatchNorm1d(512),
            nn.Dropout(drop_rate * 0.75),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        avg_f = self.avg_pool(x).flatten(1)
        max_f = self.max_pool(x).flatten(1)
        combined = torch.cat([avg_f, max_f], dim=1)
        return self.fc(combined)


class DeepfakeClassifier(nn.Module):
    def __init__(self, backbone_name="efficientnet_b0", pretrained=True, drop_rate=0.35):
        super().__init__()
        self.backbone_name = backbone_name
        self.backbone = timm.create_model(backbone_name, pretrained=pretrained, num_classes=0, drop_rate=drop_rate)
        in_features = self.backbone.num_features
        self.head = ForensicHead(in_features, num_classes=2, drop_rate=drop_rate)

    def forward(self, x):
        feat = self.backbone.forward_features(x)
        return self.head(feat)


# ============================================================
# DATA TRANSFORMS
# ============================================================

def get_transforms(img_size=224):
    norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    train_tf = transforms.Compose([
        transforms.Resize((img_size, img_size), Image.BICUBIC),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10),
        transforms.ToTensor(),
        norm
    ])

    val_tf = transforms.Compose([
        transforms.Resize((img_size, img_size), Image.BICUBIC),
        transforms.ToTensor(),
        norm
    ])

    return train_tf, val_tf


# ============================================================
# TRAINING FUNCTION
# ============================================================

def train_backbone(backbone_name, train_dir, val_dir, img_size=224, epochs=20, lr=3e-4, batch_size=32, device="cuda"):
    print(f"\n" + "=" * 65)
    print(f"[*] Training Backbone: {backbone_name} ({img_size}x{img_size}) | Epochs: {epochs}")
    print("=" * 65)

    train_tf, val_tf = get_transforms(img_size)
    train_ds = ImageFolder(train_dir, transform=train_tf)
    val_ds = ImageFolder(val_dir, transform=val_tf)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    model = DeepfakeClassifier(backbone_name=backbone_name, pretrained=True, drop_rate=0.35)
    model.to(device)

    # Class balance weights
    targets = [s[1] for s in train_ds.samples]
    class_counts = np.bincount(targets)
    class_weights = len(targets) / (len(class_counts) * class_counts.astype(np.float32))
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float, device=device)

    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor, label_smoothing=0.04)

    # Differential learning rate
    backbone_params = [p for n, p in model.named_parameters() if "head" not in n]
    head_params = [p for n, p in model.named_parameters() if "head" in n]

    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": lr * 0.5, "weight_decay": 1e-2},
        {"params": head_params, "lr": lr * 2.0, "weight_decay": 2e-2}
    ])

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler('cuda', enabled=(device == "cuda"))

    best_val_auc = 0.0
    best_val_acc = 0.0
    best_weights = None

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        train_correct = 0
        total_train = 0

        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)

            optimizer.zero_grad()
            with torch.amp.autocast('cuda', enabled=(device == "cuda")):
                outputs = model(bx)
                loss = criterion(outputs, by)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * bx.size(0)
            preds = outputs.argmax(dim=1)
            train_correct += (preds == by).sum().item()
            total_train += bx.size(0)

        scheduler.step()

        # Validation
        model.eval()
        val_probs_real = []
        val_preds = []
        val_targets = []

        with torch.no_grad():
            for bx, by in val_loader:
                bx = bx.to(device)
                with torch.amp.autocast('cuda', enabled=(device == "cuda")):
                    outputs = model(bx)
                probs = F.softmax(outputs, dim=1)
                preds = outputs.argmax(dim=1)

                val_probs_real.extend(probs[:, 1].cpu().numpy())
                val_preds.extend(preds.cpu().numpy())
                val_targets.extend(by.numpy())

        val_acc = accuracy_score(val_targets, val_preds) * 100.0
        val_auc = roc_auc_score(val_targets, val_probs_real) * 100.0
        train_acc = (train_correct / total_train) * 100.0

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_val_acc = val_acc
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 4 == 0 or epoch == epochs or val_auc == best_val_auc:
            print(f"  [{backbone_name}] Epoch {epoch:02d}/{epochs:02d} | Train Acc: {train_acc:.1f}% | Val Acc: {val_acc:.2f}% | Val AUC: {val_auc:.2f}% {'[BEST]' if val_auc == best_val_auc else ''}", flush=True)

    model.load_state_dict(best_weights)
    model.to(device)
    model.eval()
    print(f"[+] Complete {backbone_name} | Best Val AUC: {best_val_auc:.2f}%, Best Val Acc: {best_val_acc:.2f}%")
    return model, best_weights, best_val_auc, best_val_acc


# ============================================================
# INFERENCE WITH TTA
# ============================================================

def evaluate_tta_loader(model, dataloader, device="cuda"):
    model.eval()
    all_probs = []
    all_targets = []

    with torch.no_grad():
        for bx, by in dataloader:
            bx = bx.to(device)
            bx_flip = torch.flip(bx, dims=[-1])

            with torch.amp.autocast('cuda', enabled=(device == "cuda")):
                out_orig = model(bx)
                out_flip = model(bx_flip)

            prob_orig = F.softmax(out_orig, dim=1)
            prob_flip = F.softmax(out_flip, dim=1)
            prob_avg = (prob_orig + prob_flip) / 2.0

            all_probs.append(prob_avg.cpu())
            all_targets.extend(by.numpy())

    return torch.cat(all_probs, dim=0), np.array(all_targets)


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 70, flush=True)
    print("DEEPFAKE INVESTIGATION SYSTEM — HIGH-ACCURACY ENSEMBLE ENGINE", flush=True)
    print("=" * 70, flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Compute Device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})", flush=True)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_base = os.path.join(base_dir, "DeepFake-Detect", "split_dataset")
    train_dir = os.path.join(dataset_base, "train")
    val_dir = os.path.join(dataset_base, "val")
    test_dir = os.path.join(dataset_base, "test")

    OUTPUT_MODEL_PATH = os.path.join(base_dir, "deepfake_detector_best.pt")
    METADATA_PATH = os.path.join(base_dir, "model_metadata.json")

    # 1. Train Stream 1: EfficientNet-B0 (224x224)
    model_b0, weights_b0, auc_b0, acc_b0 = train_backbone(
        "efficientnet_b0", train_dir, val_dir, img_size=224, epochs=20, lr=3e-4, batch_size=32, device=str(device)
    )

    # 2. Train Stream 2: EfficientNet-B3 (288x288)
    model_b3, weights_b3, auc_b3, acc_b3 = train_backbone(
        "efficientnet_b3", train_dir, val_dir, img_size=288, epochs=20, lr=2.5e-4, batch_size=24, device=str(device)
    )

    # Prepare Validation and Test Loaders for Ensemble Tuning
    _, val_tf_224 = get_transforms(224)
    _, val_tf_288 = get_transforms(288)

    val_ds_224 = ImageFolder(val_dir, transform=val_tf_224)
    val_ds_288 = ImageFolder(val_dir, transform=val_tf_288)
    val_loader_224 = DataLoader(val_ds_224, batch_size=32, shuffle=False, num_workers=0)
    val_loader_288 = DataLoader(val_ds_288, batch_size=32, shuffle=False, num_workers=0)

    test_ds_224 = ImageFolder(test_dir, transform=val_tf_224)
    test_ds_288 = ImageFolder(test_dir, transform=val_tf_288)
    test_loader_224 = DataLoader(test_ds_224, batch_size=32, shuffle=False, num_workers=0)
    test_loader_288 = DataLoader(test_ds_288, batch_size=32, shuffle=False, num_workers=0)

    class_to_idx = val_ds_224.class_to_idx
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    fake_idx = class_to_idx["fake"]
    real_idx = class_to_idx["real"]

    print("\n[*] Optimizing Ensemble Weights & Threshold on Validation Set...", flush=True)
    val_probs_b0, val_y = evaluate_tta_loader(model_b0, val_loader_224, device=str(device))
    val_probs_b3, _ = evaluate_tta_loader(model_b3, val_loader_288, device=str(device))

    best_w0 = 0.50
    best_th = 0.50
    best_val_score = 0.0

    for w0 in np.linspace(0.15, 0.85, 15):
        w3 = 1.0 - w0
        comb = w0 * val_probs_b0 + w3 * val_probs_b3
        probs_real = comb[:, real_idx].numpy()

        for th in np.linspace(0.35, 0.65, 31):
            preds = (probs_real >= th).astype(int)
            acc = accuracy_score(val_y, preds)
            if acc > best_val_score:
                best_val_score = acc
                best_w0 = float(w0)
                best_th = float(th)

    best_w3 = 1.0 - best_w0
    print(f"[+] Optimal Validation Config: Weight(B0)={best_w0:.2f}, Weight(B3)={best_w3:.2f}, Threshold={best_th:.2f} (Val Acc: {best_val_score*100:.2f}%)")

    # Evaluate on Test Set
    print("\n" + "=" * 70, flush=True)
    print("FINAL BENCHMARK ON UNSEEN TEST DATASET", flush=True)
    print("=" * 70, flush=True)

    test_probs_b0, test_y = evaluate_tta_loader(model_b0, test_loader_224, device=str(device))
    test_probs_b3, _ = evaluate_tta_loader(model_b3, test_loader_288, device=str(device))

    test_comb_probs = best_w0 * test_probs_b0 + best_w3 * test_probs_b3
    test_probs_real = test_comb_probs[:, real_idx].numpy()
    test_preds = (test_probs_real >= best_th).astype(int)

    test_acc = accuracy_score(test_y, test_preds) * 100.0
    test_auc = roc_auc_score(test_y, test_probs_real) * 100.0
    test_prec = precision_score(test_y, test_preds, pos_label=fake_idx) * 100.0
    test_rec = recall_score(test_y, test_preds, pos_label=fake_idx) * 100.0
    test_f1 = f1_score(test_y, test_preds, pos_label=fake_idx) * 100.0
    cm = confusion_matrix(test_y, test_preds)

    real_mask = (test_y == real_idx)
    fake_mask = (test_y == fake_idx)
    real_acc = (test_preds[real_mask] == test_y[real_mask]).mean() * 100.0
    fake_acc = (test_preds[fake_mask] == test_y[fake_mask]).mean() * 100.0

    print(f"Overall Test Accuracy: {test_acc:.2f}%", flush=True)
    print(f"Real Class Accuracy  : {real_acc:.2f}%", flush=True)
    print(f"Fake Class Accuracy  : {fake_acc:.2f}%", flush=True)
    print(f"Test ROC-AUC         : {test_auc:.2f}%", flush=True)
    print(f"Fake Precision       : {test_prec:.2f}%", flush=True)
    print(f"Fake Recall          : {test_rec:.2f}%", flush=True)
    print(f"Fake F1-Score        : {test_f1:.2f}%", flush=True)
    print(f"Confusion Matrix     :\n{cm}", flush=True)
    print("=" * 70, flush=True)

    # Save Ensemble Checkpoint
    checkpoint = {
        "is_ensemble": True,
        "model_name": "Ensemble(EfficientNet-B0 + EfficientNet-B3)",
        "model_b0_state_dict": weights_b0,
        "model_b3_state_dict": weights_b3,
        "weight_b0": round(best_w0, 2),
        "weight_b3": round(best_w3, 2),
        "threshold": round(best_th, 3),
        "class_to_idx": class_to_idx,
        "idx_to_class": idx_to_class,
        "test_acc": test_acc,
        "test_auc": test_auc
    }
    torch.save(checkpoint, OUTPUT_MODEL_PATH)
    print(f"[+] Saved model checkpoint: {OUTPUT_MODEL_PATH}", flush=True)

    # Update model_metadata.json
    metadata = {
        "model_name": "Ensemble(EfficientNet-B0 + EfficientNet-B3)",
        "is_ensemble": True,
        "metrics": {
            "test_accuracy": round(test_acc, 2),
            "real_accuracy": round(real_acc, 2),
            "fake_accuracy": round(fake_acc, 2),
            "test_auc": round(test_auc, 2),
            "precision": round(test_prec, 2),
            "recall": round(test_rec, 2),
            "f1_score": round(test_f1, 2)
        },
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"[+] Saved updated metadata to {METADATA_PATH}", flush=True)


if __name__ == "__main__":
    main()
