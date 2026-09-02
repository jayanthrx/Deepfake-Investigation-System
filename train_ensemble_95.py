"""
train_ensemble_95.py
Trains a High-Accuracy Forensic Ensemble:
Stream 1: Dual-Stream EfficientNet-B0 (Spatial + Frequency Residuals)
Stream 2: Dual-Stream ResNet-34 / ConvNeXt (Spatial + Frequency Residuals)
Ensemble Decision Fusion + Multi-Scale Test-Time Augmentation (TTA)
Targeting 93% - 95%+ Accuracy.
"""

import os
import sys
import json
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import timm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace", line_buffering=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="backslashreplace", line_buffering=True)


# ============================================================
# HIGH-PASS FREQUENCY FILTER
# ============================================================

def compute_highpass_residual(tensors):
    kernel = torch.tensor([
        [1, 4, 6, 4, 1],
        [4, 16, 24, 16, 4],
        [6, 24, 36, 24, 6],
        [4, 16, 24, 16, 4],
        [1, 4, 6, 4, 1]
    ], dtype=torch.float32, device=tensors.device) / 256.0

    kernel = kernel.repeat(3, 1, 1, 1)
    blurred = F.conv2d(tensors, kernel, padding=2, groups=3)
    return tensors - blurred


# ============================================================
# DUAL STREAM MODEL
# ============================================================

class DualStreamModel(nn.Module):
    def __init__(self, backbone_name="efficientnet_b0", num_classes=2, drop_rate=0.40):
        super().__init__()
        self.rgb_backbone = timm.create_model(backbone_name, pretrained=True, num_classes=0, drop_rate=drop_rate)
        self.res_backbone = timm.create_model(backbone_name, pretrained=True, num_classes=0, drop_rate=drop_rate)

        num_features = self.rgb_backbone.num_features * 2
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(num_features),
            nn.Dropout(drop_rate),
            nn.Linear(num_features, 256),
            nn.GELU(),
            nn.BatchNorm1d(256),
            nn.Dropout(drop_rate),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        res = compute_highpass_residual(x)
        feat_rgb = self.rgb_backbone(x)
        feat_res = self.res_backbone(res)
        combined = torch.cat([feat_rgb, feat_res], dim=1)
        return self.classifier(combined)


# ============================================================
# ENSEMBLE DETECTOR
# ============================================================

class ForensicEnsembleDetector(nn.Module):
    def __init__(self, model_a, model_b, weight_a=0.55, weight_b=0.45):
        super().__init__()
        self.model_a = model_a
        self.model_b = model_b
        self.weight_a = weight_a
        self.weight_b = weight_b

    def forward(self, x):
        out_a = self.model_a(x)
        out_b = self.model_b(x)
        prob_a = F.softmax(out_a, dim=1)
        prob_b = F.softmax(out_b, dim=1)
        return self.weight_a * prob_a + self.weight_b * prob_b


# ============================================================
# DATASET UTILS
# ============================================================

def load_dataset(root_dir, img_size=224):
    classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}

    to_tensor = transforms.ToTensor()
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    tensor_list = []
    label_list = []

    for cls_name in classes:
        cls_dir = os.path.join(root_dir, cls_name)
        label = class_to_idx[cls_name]
        files = [f for f in os.listdir(cls_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        for fname in files:
            fpath = os.path.join(cls_dir, fname)
            try:
                with Image.open(fpath) as img:
                    img_rgb = img.convert("RGB").resize((img_size, img_size), Image.BICUBIC)
                    t = normalize(to_tensor(img_rgb))
                    tensor_list.append(t)
                    label_list.append(label)
            except Exception:
                pass

    return torch.stack(tensor_list), torch.tensor(label_list, dtype=torch.long), class_to_idx


def apply_gpu_augmentations(batch_x):
    # Random horizontal flip
    flip_mask = torch.rand(batch_x.size(0), device=batch_x.device) > 0.5
    if flip_mask.any():
        batch_x[flip_mask] = torch.flip(batch_x[flip_mask], dims=[-1])

    # Random scale/contrast
    scale = 1.0 + (torch.rand((batch_x.size(0), 1, 1, 1), device=batch_x.device) - 0.5) * 0.15
    batch_x = batch_x * scale
    return batch_x


def train_single_model(model_name, train_x, train_y, val_x, val_y, class_weights, num_epochs=20, lr=2.5e-4, device="cuda"):
    print(f"\n[*] Training Sub-Model Backbone: {model_name}...", flush=True)
    model = DualStreamModel(backbone_name=model_name, num_classes=2, drop_rate=0.40)
    model.to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.03)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=2e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler('cuda', enabled=(device == "cuda"))

    best_val_auc = 0.0
    best_weights = None
    N_train = train_x.size(0)
    N_val = val_x.size(0)
    BATCH_SIZE = 32

    for epoch in range(1, num_epochs + 1):
        model.train()
        perm = torch.randperm(N_train, device=device)
        running_loss = 0.0

        for i in range(0, N_train, BATCH_SIZE):
            batch_indices = perm[i:i + BATCH_SIZE]
            bx = train_x[batch_indices].clone()
            by = train_y[batch_indices]

            bx = apply_gpu_augmentations(bx)

            optimizer.zero_grad()
            with torch.amp.autocast('cuda', enabled=(device == "cuda")):
                outputs = model(bx)
                loss = criterion(outputs, by)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item() * bx.size(0)

        scheduler.step()

        # Validation
        model.eval()
        val_probs_real = []
        with torch.no_grad():
            for i in range(0, N_val, BATCH_SIZE):
                bx = val_x[i:i + BATCH_SIZE]
                with torch.amp.autocast('cuda', enabled=(device == "cuda")):
                    outputs = model(bx)
                probs = F.softmax(outputs, dim=1)
                val_probs_real.extend(probs[:, 1].cpu().numpy())

        val_auc = roc_auc_score(val_y.cpu().numpy(), val_probs_real) * 100.0
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 5 == 0 or epoch == num_epochs:
            print(f"  [{model_name}] Epoch [{epoch:02d}/{num_epochs:02d}] Loss: {running_loss/N_train:.4f} | Val AUC: {val_auc:.2f}% (Best: {best_val_auc:.2f}%)", flush=True)

    model.load_state_dict(best_weights)
    model.to(device)
    model.eval()
    return model, best_val_auc


def main():
    print("=" * 70, flush=True)
    print("FORENSIC DUAL-STREAM ENSEMBLE TRAINING (TARGET: 93% - 95%+ ACCURACY)", flush=True)
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

    print("[*] Preloading datasets into memory...", flush=True)
    t0 = time.time()
    train_x, train_y, class_to_idx = load_dataset(train_dir, img_size=224)
    val_x, val_y, _ = load_dataset(val_dir, img_size=224)
    test_x, test_y, _ = load_dataset(test_dir, img_size=224)
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    print(f"[*] Dataset loaded in {time.time() - t0:.2f}s | Train: {len(train_x)}, Val: {len(val_x)}, Test: {len(test_x)}", flush=True)

    train_x = train_x.to(device)
    train_y = train_y.to(device)
    val_x = val_x.to(device)
    val_y = val_y.to(device)
    test_x = test_x.to(device)
    test_y = test_y.to(device)

    class_counts = np.bincount(train_y.cpu().numpy())
    class_weights = len(train_y) / (len(class_counts) * class_counts.astype(np.float32))
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float, device=device)

    # 1. Train Backbone A: EfficientNet-B0 Dual Stream
    model_a, auc_a = train_single_model("efficientnet_b0", train_x, train_y, val_x, val_y, class_weights_tensor, num_epochs=22, lr=2.5e-4, device=str(device))

    # 2. Train Backbone B: ResNet-34 Dual Stream
    model_b, auc_b = train_single_model("resnet34", train_x, train_y, val_x, val_y, class_weights_tensor, num_epochs=20, lr=2.0e-4, device=str(device))

    print(f"\n[+] Backbone A (EfficientNet-B0) Val AUC: {auc_a:.2f}%")
    print(f"[+] Backbone B (ResNet-34) Val AUC:       {auc_b:.2f}%")

    # 3. Build Ensemble & Evaluate on Test Set with Multi-Scale TTA
    print("\n[*] Assembling Dual-Stream Ensemble with Multi-Scale TTA...", flush=True)
    ensemble = ForensicEnsembleDetector(model_a, model_b, weight_a=0.55, weight_b=0.45)
    ensemble.eval()

    test_preds_list = []
    test_probs_fake_list = []
    N_test = test_x.size(0)
    fake_idx = class_to_idx["fake"]
    real_idx = class_to_idx["real"]

    with torch.no_grad():
        for i in range(0, N_test, 32):
            bx = test_x[i:i + 32]
            bx_flip = torch.flip(bx, dims=[-1])

            with torch.amp.autocast('cuda', enabled=(device.type == "cuda")):
                p1 = ensemble(bx)
                p2 = ensemble(bx_flip)
                probs = (p1 + p2) / 2.0

            preds = probs.argmax(dim=1)
            test_probs_fake_list.extend(probs[:, fake_idx].cpu().numpy())
            test_preds_list.extend(preds.cpu().numpy())

    test_y_cpu = test_y.cpu().numpy()
    test_acc = accuracy_score(test_y_cpu, test_preds_list) * 100.0
    test_auc = roc_auc_score(test_y_cpu, test_probs_fake_list) * 100.0
    test_prec = precision_score(test_y_cpu, test_preds_list, pos_label=fake_idx) * 100.0
    test_rec = recall_score(test_y_cpu, test_preds_list, pos_label=fake_idx) * 100.0
    test_f1 = f1_score(test_y_cpu, test_preds_list, pos_label=fake_idx) * 100.0
    cm = confusion_matrix(test_y_cpu, test_preds_list)

    real_mask = (test_y_cpu == real_idx)
    fake_mask = (test_y_cpu == fake_idx)
    real_acc = (np.array(test_preds_list)[real_mask] == test_y_cpu[real_mask]).mean() * 100.0
    fake_acc = (np.array(test_preds_list)[fake_mask] == test_y_cpu[fake_mask]).mean() * 100.0

    print("\n" + "=" * 70, flush=True)
    print("FINAL ENSEMBLE TEST SET RESULTS", flush=True)
    print("=" * 70, flush=True)
    print(f"Total Test Accuracy : {test_acc:.2f}%", flush=True)
    print(f"Real Class Accuracy : {real_acc:.2f}%", flush=True)
    print(f"Fake Class Accuracy : {fake_acc:.2f}%", flush=True)
    print(f"Test ROC-AUC        : {test_auc:.2f}%", flush=True)
    print(f"Fake Precision      : {test_prec:.2f}%", flush=True)
    print(f"Fake Recall         : {test_rec:.2f}%", flush=True)
    print(f"Fake F1-Score       : {test_f1:.2f}%", flush=True)
    print(f"Confusion Matrix    :\n{cm}", flush=True)
    print("=" * 70, flush=True)

    # Save Ensemble Checkpoint
    torch.save({
        "is_ensemble": True,
        "is_dual_stream": True,
        "model_a_name": "efficientnet_b0",
        "model_b_name": "resnet34",
        "model_a_state_dict": model_a.state_dict(),
        "model_b_state_dict": model_b.state_dict(),
        "weight_a": 0.55,
        "weight_b": 0.45,
        "img_size": 224,
        "class_to_idx": class_to_idx,
        "test_acc": test_acc,
        "test_auc": test_auc
    }, OUTPUT_MODEL_PATH)
    print(f"[*] Saved Forensic Ensemble Checkpoint to {OUTPUT_MODEL_PATH}", flush=True)

    metadata = {
        "model_name": "forensic_dual_stream_ensemble (efficientnet_b0 + resnet34)",
        "is_ensemble": True,
        "is_dual_stream": True,
        "img_size": 224,
        "class_to_idx": class_to_idx,
        "idx_to_class": idx_to_class,
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
    print(f"[*] Saved metadata to {METADATA_PATH}", flush=True)


if __name__ == "__main__":
    main()
