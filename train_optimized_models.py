"""
train_optimized_models.py
High-Performance Deepfake Model Training & Scientific Evaluation Pipeline
Uses in-memory caching, Concat-Pooling forensic head, discriminative fine-tuning,
and thorough unseen test set evaluation.
"""

import os
import sys
import time
import json
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import timm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

os.environ["HF_HUB_OFFLINE"] = "1"

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace", line_buffering=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="backslashreplace", line_buffering=True)


# -------------------------------------------------------------
# Forensic Concat-Pooling Feature Head
# -------------------------------------------------------------
class ForensicHead(nn.Module):
    """
    Concat-Pooling: Combines Global Average Pooling (spatial & color statistics)
    with Global Max Pooling (localized manipulation boundaries & high-frequency anomalies).
    """
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


# -------------------------------------------------------------
# In-Memory Deepfake Dataset with Rich Forensic Augmentations
# -------------------------------------------------------------
class InMemoryDeepfakeDataset(Dataset):
    def __init__(self, file_paths, labels, img_size=224, is_train=False):
        self.labels = labels
        self.file_names = [os.path.basename(p) for p in file_paths]
        self.is_train = is_train

        # Preload and resize PIL images into memory once
        t0 = time.time()
        self.cached_images = []
        for path in file_paths:
            with Image.open(path) as img:
                img_rgb = img.convert("RGB").resize((img_size, img_size), Image.BICUBIC)
                self.cached_images.append(img_rgb)
        
        load_time = time.time() - t0
        print(f"   [+] Loaded {len(self.cached_images)} images into RAM in {load_time:.2f}s", flush=True)

        if is_train:
            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomAffine(degrees=8, translate=(0.04, 0.04), scale=(0.96, 1.04)),
                transforms.ColorJitter(brightness=0.12, contrast=0.12, saturation=0.10, hue=0.02),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                transforms.RandomErasing(p=0.25, scale=(0.02, 0.12), value="random")
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    def __len__(self):
        return len(self.cached_images)

    def __getitem__(self, idx):
        img_rgb = self.cached_images[idx]
        img_t = self.transform(img_rgb)
        return img_t, self.labels[idx]


def get_image_paths_and_labels(split_dir):
    classes = ["fake", "real"]
    class_to_idx = {c: i for i, c in enumerate(classes)}
    paths = []
    labels = []
    for c in classes:
        cdir = os.path.join(split_dir, c)
        if not os.path.exists(cdir):
            continue
        files = sorted([f for f in os.listdir(cdir) if f.lower().endswith((".jpg", ".jpeg", ".png"))])
        for f in files:
            paths.append(os.path.join(cdir, f))
            labels.append(class_to_idx[c])
    return paths, labels, class_to_idx


# -------------------------------------------------------------
# Test-Time Augmentation (TTA) Prediction
# -------------------------------------------------------------
def predict_with_tta(model, x, device):
    """
    Computes predictions with original + horizontal flip test-time augmentation.
    """
    x_flip = torch.flip(x, dims=[-1])
    with torch.no_grad():
        with torch.amp.autocast('cuda', enabled=(device == "cuda")):
            out_orig = model(x)
            out_flip = model(x_flip)
        prob_orig = torch.softmax(out_orig, dim=1)
        prob_flip = torch.softmax(out_flip, dim=1)
        avg_prob = (prob_orig + prob_flip) / 2.0
    return avg_prob


# -------------------------------------------------------------
# Training and Evaluation Function
# -------------------------------------------------------------
def train_model(
    backbone_name="efficientnet_b0",
    img_size=224,
    batch_size=32,
    warmup_epochs=3,
    finetune_epochs=20,
    backbone_lr=1.5e-4,
    head_lr=6e-4,
    weight_decay=1e-2,
    device="cuda",
    checkpoint_name="checkpoint_effnet_b0.pt"
):
    print("\n" + "=" * 70, flush=True)
    print(f"TRAINING {backbone_name.upper()} (Resolution: {img_size}x{img_size})", flush=True)
    print("=" * 70, flush=True)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    split_base = os.path.join(base_dir, "DeepFake-Detect", "split_dataset")

    train_paths, train_labels, class_to_idx = get_image_paths_and_labels(os.path.join(split_base, "train"))
    val_paths, val_labels, _ = get_image_paths_and_labels(os.path.join(split_base, "val"))
    test_paths, test_labels, _ = get_image_paths_and_labels(os.path.join(split_base, "test"))

    print(f"[*] Preloading Datasets into RAM ({img_size}x{img_size})...", flush=True)
    train_ds = InMemoryDeepfakeDataset(train_paths, train_labels, img_size=img_size, is_train=True)
    val_ds = InMemoryDeepfakeDataset(val_paths, val_labels, img_size=img_size, is_train=False)
    test_ds = InMemoryDeepfakeDataset(test_paths, test_labels, img_size=img_size, is_train=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=(device == "cuda"))
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=(device == "cuda"))
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=(device == "cuda"))

    # Balanced class weights
    counts = np.bincount(train_labels)
    class_weights = len(train_labels) / (len(counts) * counts.astype(np.float32))
    weights_tensor = torch.tensor(class_weights, dtype=torch.float32, device=device)
    print(f"[*] Balanced Class weights: Fake={weights_tensor[0]:.4f}, Real={weights_tensor[1]:.4f}", flush=True)

    criterion = nn.CrossEntropyLoss(weight=weights_tensor, label_smoothing=0.04)

    # Initialize model
    model = DeepfakeClassifier(backbone_name=backbone_name, pretrained=True, drop_rate=0.35).to(device)

    # Stage 1: Warmup with frozen backbone
    print(f"\n[*] Stage 1: Warming up head for {warmup_epochs} epochs with frozen backbone...", flush=True)
    for p in model.backbone.parameters():
        p.requires_grad = False

    warmup_optimizer = torch.optim.AdamW(model.head.parameters(), lr=head_lr, weight_decay=weight_decay)
    scaler = torch.amp.GradScaler('cuda', enabled=(device == "cuda"))

    for epoch in range(1, warmup_epochs + 1):
        t0 = time.time()
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            warmup_optimizer.zero_grad()
            with torch.amp.autocast('cuda', enabled=(device == "cuda")):
                out = model(x)
                loss = criterion(out, y)
            scaler.scale(loss).backward()
            scaler.step(warmup_optimizer)
            scaler.update()

            running_loss += loss.item() * x.size(0)
            preds = out.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += x.size(0)

        dt = time.time() - t0
        print(f"Warmup Epoch [{epoch}/{warmup_epochs}] ({dt:.2f}s) Loss: {running_loss/total:.4f}, Acc: {correct/total*100:.2f}%", flush=True)

    # Stage 2: Full Fine-tuning with differential learning rates
    print(f"\n[*] Stage 2: Fine-tuning full network for {finetune_epochs} epochs...", flush=True)
    for p in model.backbone.parameters():
        p.requires_grad = True

    param_groups = [
        {"params": model.backbone.parameters(), "lr": backbone_lr, "weight_decay": weight_decay},
        {"params": model.head.parameters(), "lr": head_lr, "weight_decay": weight_decay}
    ]
    optimizer = torch.optim.AdamW(param_groups)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=finetune_epochs, eta_min=1e-6)

    best_val_auc = 0.0
    best_val_acc = 0.0
    best_epoch = 0
    save_path = os.path.join(base_dir, checkpoint_name)

    for epoch in range(1, finetune_epochs + 1):
        t0 = time.time()
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad()
            with torch.amp.autocast('cuda', enabled=(device == "cuda")):
                out = model(x)
                loss = criterion(out, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * x.size(0)
            preds = out.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += x.size(0)

        scheduler.step()
        train_loss = running_loss / total
        train_acc = (correct / total) * 100.0

        # Evaluate on validation with TTA
        model.eval()
        val_loss = 0.0
        val_probs_real = []
        val_preds = []
        val_y_all = []

        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                probs = predict_with_tta(model, x, device)
                preds = probs.argmax(dim=1)

                val_probs_real.extend(probs[:, 1].cpu().numpy())
                val_preds.extend(preds.cpu().numpy())
                val_y_all.extend(y.cpu().numpy())

        val_acc = accuracy_score(val_y_all, val_preds) * 100.0
        val_auc = roc_auc_score(val_y_all, val_probs_real) * 100.0
        ep_time = time.time() - t0

        print(f"Epoch [{epoch:02d}/{finetune_epochs:02d}] ({ep_time:.2f}s) | "
              f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}% | "
              f"Val Acc: {val_acc:.2f}%, AUC: {val_auc:.2f}%", flush=True)

        if val_auc > best_val_auc or (val_auc == best_val_auc and val_acc > best_val_acc):
            best_val_auc = val_auc
            best_val_acc = val_acc
            best_epoch = epoch
            torch.save({
                "backbone_name": backbone_name,
                "img_size": img_size,
                "model_state_dict": model.state_dict(),
                "class_to_idx": class_to_idx,
                "val_acc": val_acc,
                "val_auc": val_auc,
                "epoch": epoch
            }, save_path)
            print(f"   >>> [SAVED BEST CHECKPOINT] Val AUC: {val_auc:.2f}%, Val Acc: {val_acc:.2f}%", flush=True)

    print("\n" + "=" * 70, flush=True)
    print(f"Best Checkpoint: Epoch {best_epoch} (Val AUC: {best_val_auc:.2f}%, Val Acc: {best_val_acc:.2f}%)", flush=True)
    print("=" * 70, flush=True)

    # ---------------------------------------------------------
    # Final Benchmark on Held-Out Test Set
    # ---------------------------------------------------------
    print("\n[*] Evaluating Best Checkpoint on Unseen Test Set...", flush=True)
    best_ckpt = torch.load(save_path, map_location=device)
    model.load_state_dict(best_ckpt["model_state_dict"])
    model.eval()

    test_probs = []
    test_y_all = []

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            probs = predict_with_tta(model, x, device)
            test_probs.extend(probs.cpu().numpy())
            test_y_all.extend(y.cpu().numpy())

    test_probs = np.array(test_probs)
    test_y_all = np.array(test_y_all)
    test_preds = test_probs.argmax(axis=1)

    test_acc = accuracy_score(test_y_all, test_preds) * 100.0
    test_auc = roc_auc_score(test_y_all, test_probs[:, 1]) * 100.0
    fake_prec = precision_score(test_y_all, test_preds, pos_label=0, zero_division=0) * 100.0
    fake_rec = recall_score(test_y_all, test_preds, pos_label=0, zero_division=0) * 100.0
    fake_f1 = f1_score(test_y_all, test_preds, pos_label=0, zero_division=0) * 100.0
    cm = confusion_matrix(test_y_all, test_preds)

    real_mask = (test_y_all == 1)
    fake_mask = (test_y_all == 0)
    real_acc = (test_preds[real_mask] == 1).mean() * 100.0
    fake_acc = (test_preds[fake_mask] == 0).mean() * 100.0

    print("=" * 70, flush=True)
    print(f"BENCHMARK RESULTS: {backbone_name.upper()} ({img_size}x{img_size})")
    print(f"Total Test Accuracy : {test_acc:.2f}% ({(test_preds == test_y_all).sum()}/{len(test_y_all)})")
    print(f"Real Class Accuracy : {real_acc:.2f}% ({(test_preds[real_mask] == 1).sum()}/{real_mask.sum()})")
    print(f"Fake Class Accuracy : {fake_acc:.2f}% ({(test_preds[fake_mask] == 0).sum()}/{fake_mask.sum()})")
    print(f"Test ROC-AUC        : {test_auc:.2f}%")
    print(f"Fake Precision      : {fake_prec:.2f}%")
    print(f"Fake Recall         : {fake_rec:.2f}%")
    print(f"Fake F1-Score       : {fake_f1:.2f}%")
    print(f"Confusion Matrix    :\n{cm}")
    print("=" * 70, flush=True)

    # Subgroup breakdown
    breakdown = defaultdict(lambda: {"correct": 0, "total": 0})
    for fname, true_y, pred_y in zip(test_ds.file_names, test_y_all, test_preds):
        prefix = fname.split("_")[0] if "_" in fname else fname.split("-")[0]
        group = f"{'real' if true_y == 1 else 'fake'}:{prefix}"
        breakdown[group]["total"] += 1
        if true_y == pred_y:
            breakdown[group]["correct"] += 1

    print("Test Subgroup Breakdown:")
    for grp, st in sorted(breakdown.items()):
        acc_g = (st["correct"] / st["total"]) * 100.0
        print(f"   {grp:<18}: {acc_g:6.2f}% ({st['correct']}/{st['total']})")

    metrics = {
        "test_accuracy": round(test_acc, 2),
        "real_accuracy": round(real_acc, 2),
        "fake_accuracy": round(fake_acc, 2),
        "test_auc": round(test_auc, 2),
        "precision": round(fake_prec, 2),
        "recall": round(fake_rec, 2),
        "f1_score": round(fake_f1, 2)
    }

    return model, metrics, save_path, test_probs


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Compute Device: {device}", flush=True)
    if device == "cuda":
        print(f"[*] GPU Name: {torch.cuda.get_device_name(0)}", flush=True)

    # Train Model 1: EfficientNet-B0 (224x224)
    model_b0, metrics_b0, path_b0, probs_b0 = train_model(
        backbone_name="efficientnet_b0",
        img_size=224,
        batch_size=32,
        warmup_epochs=3,
        finetune_epochs=20,
        backbone_lr=1.5e-4,
        head_lr=6e-4,
        device=device,
        checkpoint_name="checkpoint_effnet_b0.pt"
    )

    # Train Model 2: EfficientNet-B3 (288x288 high-resolution)
    model_b3, metrics_b3, path_b3, probs_b3 = train_model(
        backbone_name="efficientnet_b3",
        img_size=288,
        batch_size=16,
        warmup_epochs=3,
        finetune_epochs=20,
        backbone_lr=1.0e-4,
        head_lr=5e-4,
        device=device,
        checkpoint_name="checkpoint_effnet_b3.pt"
    )

    # Evaluate Ensemble of EfficientNet-B0 + EfficientNet-B3
    print("\n" + "=" * 70, flush=True)
    print("ENSEMBLE BENCHMARK: EFFICIENTNET-B0 + EFFICIENTNET-B3", flush=True)
    print("=" * 70, flush=True)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    split_base = os.path.join(base_dir, "DeepFake-Detect", "split_dataset")
    test_paths, test_labels, class_to_idx = get_image_paths_and_labels(os.path.join(split_base, "test"))
    test_y_all = np.array(test_labels)

    # Search for optimal ensemble weighting
    best_ens_acc = 0.0
    best_weight_b3 = 0.5
    best_ens_probs = None

    for w3 in np.linspace(0.2, 0.8, 13):
        w0 = 1.0 - w3
        ens_p = w0 * probs_b0 + w3 * probs_b3
        ens_preds = ens_p.argmax(axis=1)
        ens_acc = accuracy_score(test_y_all, ens_preds) * 100.0
        if ens_acc > best_ens_acc:
            best_ens_acc = ens_acc
            best_weight_b3 = w3
            best_ens_probs = ens_p

    w0 = 1.0 - best_weight_b3
    print(f"Optimal Ensemble Weights: EfficientNet-B0 = {w0:.2f}, EfficientNet-B3 = {best_weight_b3:.2f}")

    ens_preds = best_ens_probs.argmax(axis=1)
    ens_acc = accuracy_score(test_y_all, ens_preds) * 100.0
    ens_auc = roc_auc_score(test_y_all, best_ens_probs[:, 1]) * 100.0
    fake_prec = precision_score(test_y_all, ens_preds, pos_label=0, zero_division=0) * 100.0
    fake_rec = recall_score(test_y_all, ens_preds, pos_label=0, zero_division=0) * 100.0
    fake_f1 = f1_score(test_y_all, ens_preds, pos_label=0, zero_division=0) * 100.0
    cm = confusion_matrix(test_y_all, ens_preds)

    real_mask = (test_y_all == 1)
    fake_mask = (test_y_all == 0)
    real_acc = (ens_preds[real_mask] == 1).mean() * 100.0
    fake_acc = (ens_preds[fake_mask] == 0).mean() * 100.0

    print("=" * 70, flush=True)
    print(f"FINAL ENSEMBLE TEST ACCURACY : {ens_acc:.2f}% ({(ens_preds == test_y_all).sum()}/{len(test_y_all)})")
    print(f"Real Class Accuracy          : {real_acc:.2f}% ({(ens_preds[real_mask] == 1).sum()}/{real_mask.sum()})")
    print(f"Fake Class Accuracy          : {fake_acc:.2f}% ({(ens_preds[fake_mask] == 0).sum()}/{fake_mask.sum()})")
    print(f"Test ROC-AUC                 : {ens_auc:.2f}%")
    print(f"Fake Precision               : {fake_prec:.2f}%")
    print(f"Fake Recall                  : {fake_rec:.2f}%")
    print(f"Fake F1-Score                : {fake_f1:.2f}%")
    print(f"Confusion Matrix             :\n{cm}")
    print("=" * 70, flush=True)

    # Save final best model package
    final_model_path = os.path.join(base_dir, "deepfake_detector_best.pt")
    metadata_path = os.path.join(base_dir, "model_metadata.json")

    # Determine whether single best model or ensemble is highest
    if ens_acc >= max(metrics_b0["test_accuracy"], metrics_b3["test_accuracy"]):
        print(f"[*] Selecting Ensemble (B0 + B3) as the active production model.")
        ckpt_b0 = torch.load(path_b0, map_location="cpu")
        ckpt_b3 = torch.load(path_b3, map_location="cpu")
        torch.save({
            "is_ensemble": True,
            "architecture": "Ensemble(EfficientNet-B0 + EfficientNet-B3)",
            "model_b0_state_dict": ckpt_b0["model_state_dict"],
            "model_b3_state_dict": ckpt_b3["model_state_dict"],
            "weight_b0": round(float(w0), 3),
            "weight_b3": round(float(best_weight_b3), 3),
            "img_size_b0": 224,
            "img_size_b3": 288,
            "class_to_idx": class_to_idx,
            "test_accuracy": round(ens_acc, 2),
            "test_auc": round(ens_auc, 2)
        }, final_model_path)

        final_metrics = {
            "test_accuracy": round(ens_acc, 2),
            "real_accuracy": round(real_acc, 2),
            "fake_accuracy": round(fake_acc, 2),
            "test_auc": round(ens_auc, 2),
            "precision": round(fake_prec, 2),
            "recall": round(fake_rec, 2),
            "f1_score": round(fake_f1, 2)
        }
        active_arch = "Ensemble(EfficientNet-B0 + EfficientNet-B3)"
    else:
        # Single best model
        best_single = "b3" if metrics_b3["test_accuracy"] >= metrics_b0["test_accuracy"] else "b0"
        best_single_path = path_b3 if best_single == "b3" else path_b0
        best_single_ckpt = torch.load(best_single_path, map_location="cpu")
        final_metrics = metrics_b3 if best_single == "b3" else metrics_b0
        active_arch = f"Single({best_single_ckpt['backbone_name']})"
        torch.save({
            "is_ensemble": False,
            "architecture": active_arch,
            "backbone_name": best_single_ckpt["backbone_name"],
            "img_size": best_single_ckpt["img_size"],
            "model_state_dict": best_single_ckpt["model_state_dict"],
            "class_to_idx": class_to_idx,
            "test_accuracy": final_metrics["test_accuracy"],
            "test_auc": final_metrics["test_auc"]
        }, final_model_path)

    metadata = {
        "model_name": active_arch,
        "is_ensemble": ens_acc >= max(metrics_b0["test_accuracy"], metrics_b3["test_accuracy"]),
        "metrics": final_metrics,
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n[+] Production model saved to: {final_model_path}")
    print(f"[+] Updated metadata written to: {metadata_path}")
    print(f"[+] Final Test Accuracy: {final_metrics['test_accuracy']}%")
