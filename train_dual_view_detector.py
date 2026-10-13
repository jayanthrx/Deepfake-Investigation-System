"""
train_dual_view_detector.py
Dual-View Multi-Scale Deepfake Forensic Architecture:
Stream 1: Global Face (Resized to 224x224) - Captures global lighting, posture, boundary
Stream 2: Native Biometric Patch (224x224 unscaled center crop) - Captures micro-texture artifacts, GAN synthesis seams
"""

import os
import sys
import time
import json
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


def extract_native_patch(img, target_size=224):
    w, h = img.size
    if w >= target_size and h >= target_size:
        left = (w - target_size) // 2
        top = (h - target_size) // 2
        return img.crop((left, top, left + target_size, top + target_size))
    return img.resize((target_size, target_size), Image.BICUBIC)


class DualViewDeepfakeDataset(Dataset):
    def __init__(self, file_paths, labels, img_size=224, is_train=False):
        self.labels = labels
        self.file_names = [os.path.basename(p) for p in file_paths]
        self.is_train = is_train

        t0 = time.time()
        self.cached_global = []
        self.cached_patch = []

        for p in file_paths:
            with Image.open(p) as img:
                img_rgb = img.convert("RGB")
                # 1. Global View (Full face resized to 224)
                g_img = img_rgb.resize((img_size, img_size), Image.BICUBIC)
                # 2. Native Patch View (224x224 unscaled central biometric crop)
                p_img = extract_native_patch(img_rgb, target_size=img_size)
                self.cached_global.append(g_img)
                self.cached_patch.append(p_img)

        load_t = time.time() - t0
        print(f"   [+] Preloaded {len(self.cached_global)} Dual-View pairs in {load_t:.2f}s", flush=True)

        norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        if is_train:
            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomAffine(degrees=6, translate=(0.03, 0.03), scale=(0.97, 1.03)),
                transforms.ColorJitter(brightness=0.10, contrast=0.10, saturation=0.08),
                transforms.ToTensor(),
                norm,
                transforms.RandomErasing(p=0.20, scale=(0.02, 0.10), value="random")
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                norm
            ])

    def __len__(self):
        return len(self.cached_global)

    def __getitem__(self, idx):
        g = self.cached_global[idx]
        p = self.cached_patch[idx]
        return self.transform(g), self.transform(p), self.labels[idx]


class DualViewClassifier(nn.Module):
    """
    Dual-View Deepfake Architecture:
    Backbone 1: Global stream
    Backbone 2: Patch stream (shares weights or uses twin EfficientNet)
    Concat-Pooling Fusion Head
    """
    def __init__(self, backbone_name="efficientnet_b0", pretrained=True, drop_rate=0.35):
        super().__init__()
        # Shared feature extractor to keep parameter count lean and prevent overfitting
        self.backbone = timm.create_model(backbone_name, pretrained=pretrained, num_classes=0, drop_rate=drop_rate)
        in_features = self.backbone.num_features  # 1280 for b0

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # 2 streams * 2 pooling types (avg + max) = 4 * in_features
        feat_dim = in_features * 4
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(feat_dim),
            nn.Dropout(drop_rate),
            nn.Linear(feat_dim, 512),
            nn.GELU(),
            nn.BatchNorm1d(512),
            nn.Dropout(drop_rate * 0.75),
            nn.Linear(512, 2)
        )

    def forward(self, x_global, x_patch):
        f_global = self.backbone.forward_features(x_global)
        f_patch = self.backbone.forward_features(x_patch)

        avg_g = self.avg_pool(f_global).flatten(1)
        max_g = self.max_pool(f_global).flatten(1)

        avg_p = self.avg_pool(f_patch).flatten(1)
        max_p = self.max_pool(f_patch).flatten(1)

        combined = torch.cat([avg_g, max_g, avg_p, max_p], dim=1)
        return self.classifier(combined)


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


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 70, flush=True)
    print(f"DUAL-VIEW (GLOBAL + NATIVE BIOMETRIC PATCH) TRAINING ON {device.upper()}", flush=True)
    print("=" * 70, flush=True)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    split_base = os.path.join(base_dir, "DeepFake-Detect", "split_dataset")

    train_paths, train_labels, class_to_idx = get_image_paths_and_labels(os.path.join(split_base, "train"))
    val_paths, val_labels, _ = get_image_paths_and_labels(os.path.join(split_base, "val"))
    test_paths, test_labels, _ = get_image_paths_and_labels(os.path.join(split_base, "test"))

    train_ds = DualViewDeepfakeDataset(train_paths, train_labels, is_train=True)
    val_ds = DualViewDeepfakeDataset(val_paths, val_labels, is_train=False)
    test_ds = DualViewDeepfakeDataset(test_paths, test_labels, is_train=False)

    batch_size = 24
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    counts = np.bincount(train_labels)
    class_weights = len(train_labels) / (len(counts) * counts.astype(np.float32))
    weights_tensor = torch.tensor(class_weights, dtype=torch.float32, device=device)
    print(f"[*] Class weights: Fake={weights_tensor[0]:.4f}, Real={weights_tensor[1]:.4f}", flush=True)

    model = DualViewClassifier("efficientnet_b0", pretrained=True, drop_rate=0.40).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights_tensor, label_smoothing=0.04)

    # Stage 1: Warmup classifier head
    print("[*] Stage 1: Warming up Dual-View Fusion Head for 3 epochs...", flush=True)
    for p in model.backbone.parameters():
        p.requires_grad = False

    warmup_optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=5e-4, weight_decay=1e-2)
    scaler = torch.amp.GradScaler('cuda')

    for epoch in range(1, 4):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for x_g, x_p, y in train_loader:
            x_g, x_p, y = x_g.to(device), x_p.to(device), y.to(device)
            warmup_optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                out = model(x_g, x_p)
                loss = criterion(out, y)
            scaler.scale(loss).backward()
            scaler.step(warmup_optimizer)
            scaler.update()

            running_loss += loss.item() * y.size(0)
            preds = out.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
        print(f"Warmup Epoch [{epoch}/3] Loss: {running_loss/total:.4f}, Acc: {correct/total*100:.2f}%", flush=True)

    # Stage 2: Full fine-tuning
    print("\n[*] Stage 2: Fine-tuning full Dual-View Network for 18 epochs...", flush=True)
    for p in model.backbone.parameters():
        p.requires_grad = True

    finetune_epochs = 18
    param_groups = [
        {"params": model.backbone.parameters(), "lr": 8e-5, "weight_decay": 2e-2},
        {"params": model.classifier.parameters(), "lr": 4e-4, "weight_decay": 2e-2}
    ]
    optimizer = torch.optim.AdamW(param_groups)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=finetune_epochs, eta_min=1e-6)

    best_val_auc = 0.0
    best_val_acc = 0.0
    save_path = os.path.join(base_dir, "checkpoint_dual_view.pt")

    for epoch in range(1, finetune_epochs + 1):
        t0 = time.time()
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for x_g, x_p, y in train_loader:
            x_g, x_p, y = x_g.to(device), x_p.to(device), y.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                out = model(x_g, x_p)
                loss = criterion(out, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * y.size(0)
            preds = out.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

        scheduler.step()
        train_loss = running_loss / total
        train_acc = (correct / total) * 100.0

        # Validate with TTA
        model.eval()
        val_probs_real = []
        val_preds = []
        val_y_all = []

        with torch.no_grad():
            for x_g, x_p, y in val_loader:
                x_g, x_p = x_g.to(device), x_p.to(device)
                with torch.amp.autocast('cuda'):
                    # TTA: flip both global and patch
                    out1 = model(x_g, x_p)
                    out2 = model(torch.flip(x_g, [-1]), torch.flip(x_p, [-1]))
                p1 = torch.softmax(out1, dim=1)
                p2 = torch.softmax(out2, dim=1)
                probs = (p1 + p2) / 2.0

                val_probs_real.extend(probs[:, 1].cpu().numpy())
                val_preds.extend(probs.argmax(dim=1).cpu().numpy())
                val_y_all.extend(y.cpu().numpy())

        val_acc = accuracy_score(val_y_all, val_preds) * 100.0
        val_auc = roc_auc_score(val_y_all, val_probs_real) * 100.0
        dt = time.time() - t0

        print(f"Epoch [{epoch:02d}/{finetune_epochs:02d}] ({dt:.2f}s) | Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%, AUC: {val_auc:.2f}%", flush=True)

        if val_auc > best_val_auc or (val_auc == best_val_auc and val_acc > best_val_acc):
            best_val_auc = val_auc
            best_val_acc = val_acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "val_acc": val_acc,
                "val_auc": val_auc,
                "epoch": epoch
            }, save_path)
            print(f"   >>> [SAVED BEST DUAL-VIEW CHECKPOINT] Val AUC: {val_auc:.2f}%, Val Acc: {val_acc:.2f}%", flush=True)

    # Final Test Set Evaluation
    print("\n" + "=" * 70, flush=True)
    print("[*] Evaluating Best Dual-View Checkpoint on Unseen Test Set...", flush=True)
    best_ckpt = torch.load(save_path, map_location=device)
    model.load_state_dict(best_ckpt["model_state_dict"])
    model.eval()

    test_probs = []
    test_y_all = []

    with torch.no_grad():
        for x_g, x_p, y in test_loader:
            x_g, x_p = x_g.to(device), x_p.to(device)
            with torch.amp.autocast('cuda'):
                out1 = model(x_g, x_p)
                out2 = model(torch.flip(x_g, [-1]), torch.flip(x_p, [-1]))
            probs = (torch.softmax(out1, dim=1) + torch.softmax(out2, dim=1)) / 2.0
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
    print("DUAL-VIEW (GLOBAL + NATIVE BIOMETRIC PATCH) TEST RESULTS")
    print(f"Total Test Accuracy : {test_acc:.2f}% ({(test_preds == test_y_all).sum()}/{len(test_y_all)})")
    print(f"Real Class Accuracy : {real_acc:.2f}% ({(test_preds[real_mask] == 1).sum()}/{real_mask.sum()})")
    print(f"Fake Class Accuracy : {fake_acc:.2f}% ({(test_preds[fake_mask] == 0).sum()}/{fake_mask.sum()})")
    print(f"Test ROC-AUC        : {test_auc:.2f}%")
    print(f"Fake Precision      : {fake_prec:.2f}%")
    print(f"Fake Recall         : {fake_rec:.2f}%")
    print(f"Fake F1-Score       : {fake_f1:.2f}%")
    print(f"Confusion Matrix    :\n{cm}")
    print("=" * 70, flush=True)


if __name__ == "__main__":
    main()
