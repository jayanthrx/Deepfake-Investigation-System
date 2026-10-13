"""
train_video_detector.py
End-to-End Automated Video Deepfake Training & Fine-Tuning Pipeline
Extracts facial frames from video datasets, balances classes,
and trains/fine-tunes the PyTorch forensic ensemble detector.
"""

import os
import sys
import glob
import json
import time
import shutil
import argparse
import numpy as np
import cv2
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
import timm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace", line_buffering=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="backslashreplace", line_buffering=True)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DATASET_DIR = os.path.join(BASE_DIR, "video_dataset")
OUTPUT_MODEL_PATH = os.path.join(BASE_DIR, "deepfake_detector_best.pt")
METADATA_PATH = os.path.join(BASE_DIR, "model_metadata.json")


# ============================================================
# ARCHITECTURE (Matches predictor.py & train_target_92.py)
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
# VIDEO SCANNER & FRAME EXTRACTION
# ============================================================

def scan_available_videos():
    """Scans repository for available real and fake video files."""
    video_catalog = []

    # 1. Check sample_videos/
    sample_dir = os.path.join(BASE_DIR, "sample_videos")
    if os.path.exists(sample_dir):
        for f in os.listdir(sample_dir):
            if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
                p = os.path.join(sample_dir, f)
                label = "real" if "real" in f.lower() else "fake"
                video_catalog.append({"path": p, "label": label, "source": "sample_videos"})

    # 2. Check DeepFake-Detect/train_sample_videos/
    dfdc_dir = os.path.join(BASE_DIR, "DeepFake-Detect", "train_sample_videos")
    meta_file = os.path.join(dfdc_dir, "metadata.json")
    if os.path.exists(dfdc_dir) and os.path.exists(meta_file):
        try:
            with open(meta_file, "r") as mf:
                meta = json.load(mf)
            for fname, info in meta.items():
                p = os.path.join(dfdc_dir, fname)
                if os.path.exists(p):
                    lbl = info.get("label", "FAKE").lower()
                    video_catalog.append({"path": p, "label": lbl, "source": "dfdc_sample"})
        except Exception as e:
            print(f"[!] Warning reading DFDC metadata: {e}")

    # Deduplicate by path
    seen = set()
    unique_catalog = []
    for item in video_catalog:
        if item["path"] not in seen:
            seen.add(item["path"])
            unique_catalog.append(item)

    return unique_catalog


def extract_faces_from_video(video_path, max_frames=20, output_dir=None, detector=None):
    """Samples video frames and extracts face regions using MTCNN with center-crop fallback."""
    if not os.path.exists(video_path):
        return []

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    if total_frames <= 0:
        step = 10
    else:
        step = max(1, total_frames // max_frames)

    extracted_paths = []
    frame_idx = 0
    extracted_count = 0

    video_basename = os.path.splitext(os.path.basename(video_path))[0]

    while True:
        ret, frame = cap.read()
        if not ret or extracted_count >= max_frames:
            break

        frame_idx += 1
        if frame_idx % step != 0:
            continue

        h, w = frame.shape[:2]
        crop = None

        if detector is not None:
            try:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                faces = detector.detect_faces(rgb_frame)
                if faces and len(faces) > 0:
                    best_face = max(faces, key=lambda f: f['box'][2] * f['box'][3])
                    bx, by, bw, bh = best_face['box']
                    margin_x = int(bw * 0.25)
                    margin_y = int(bh * 0.25)
                    x1 = max(0, bx - margin_x)
                    y1 = max(0, by - margin_y)
                    x2 = min(w, bx + bw + margin_x)
                    y2 = min(h, by + bh + margin_y)
                    if (x2 - x1) > 40 and (y2 - y1) > 40:
                        crop = frame[y1:y2, x1:x2]
            except Exception:
                crop = None

        if crop is None:
            # Smart center square crop
            min_dim = min(h, w)
            cy, cx = h // 2, w // 2
            half = min_dim // 2
            crop = frame[max(0, cy - half):min(h, cy + half), max(0, cx - half):min(w, cx + half)]

        if crop is not None and crop.size > 0:
            crop_resized = cv2.resize(crop, (288, 288), interpolation=cv2.INTER_AREA)
            out_filename = f"{video_basename}_frame_{frame_idx:04d}.jpg"
            out_path = os.path.join(output_dir, out_filename)
            cv2.imwrite(out_path, crop_resized)
            extracted_paths.append(out_path)
            extracted_count += 1

    cap.release()
    return extracted_paths


def build_video_dataset(video_catalog, output_base=OUTPUT_DATASET_DIR, frames_per_video=25):
    """Builds split train/val dataset from scanned videos."""
    print("\n" + "=" * 65)
    print("STEP 1: EXTRACTING FACE FRAMES FROM VIDEOS")
    print("=" * 65)

    train_real_dir = os.path.join(output_base, "train", "real")
    train_fake_dir = os.path.join(output_base, "train", "fake")
    val_real_dir = os.path.join(output_base, "val", "real")
    val_fake_dir = os.path.join(output_base, "val", "fake")

    # Clean existing temp dataset to avoid stale cache
    if os.path.exists(output_base):
        try:
            shutil.rmtree(output_base)
        except Exception:
            pass

    for d in [train_real_dir, train_fake_dir, val_real_dir, val_fake_dir]:
        os.makedirs(d, exist_ok=True)

    detector = None
    try:
        from mtcnn import MTCNN
        detector = MTCNN()
        print("[+] MTCNN Face Detector loaded for high-precision face alignment.")
    except Exception as e:
        print(f"[*] Fallback to smart center square crop: {e}")

    real_faces = []
    fake_faces = []

    for idx, v in enumerate(video_catalog, 1):
        vpath = v["path"]
        label = v["label"]
        print(f"[{idx}/{len(video_catalog)}] Processing {os.path.basename(vpath)} (Label: {label.upper()})...", flush=True)

        target_dir = train_real_dir if label == "real" else train_fake_dir
        frames = extract_faces_from_video(vpath, max_frames=frames_per_video, output_dir=target_dir, detector=detector)
        if label == "real":
            real_faces.extend(frames)
        else:
            fake_faces.extend(frames)
        print(f"    -> Extracted {len(frames)} face frames.")

    print(f"\n[*] Total Extracted: {len(real_faces)} REAL face frames, {len(fake_faces)} FAKE face frames.")

    # Move 15% to validation
    np.random.seed(42)
    for face_list, val_dir in [(real_faces, val_real_dir), (fake_faces, val_fake_dir)]:
        num_val = max(1, int(len(face_list) * 0.15))
        val_samples = np.random.choice(face_list, num_val, replace=False)
        for s in val_samples:
            dst = os.path.join(val_dir, os.path.basename(s))
            if os.path.exists(s):
                shutil.move(s, dst)

    train_real_count = len(os.listdir(train_real_dir))
    train_fake_count = len(os.listdir(train_fake_dir))
    val_real_count = len(os.listdir(val_real_dir))
    val_fake_count = len(os.listdir(val_fake_dir))

    print(f"[+] Dataset Partitioning Complete:")
    print(f"    Train: {train_real_count} Real, {train_fake_count} Fake ({train_real_count + train_fake_count} total)")
    print(f"    Val  : {val_real_count} Real, {val_fake_count} Fake ({val_real_count + val_fake_count} total)")

    return output_base


# ============================================================
# DATA TRANSFORMS & TRAINING PIPELINE
# ============================================================

def get_transforms(img_size=224):
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return train_transform, val_transform


def train_video_model(dataset_dir, epochs=15, batch_size=16, lr=1.5e-4):
    """Fine-tunes the forensic ensemble on the extracted video frames."""
    print("\n" + "=" * 65)
    print("STEP 2: TRAINING MODEL ON EXTRACTED VIDEO FRAMES")
    print("=" * 65)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Training Device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")

    train_dir = os.path.join(dataset_dir, "train")
    val_dir = os.path.join(dataset_dir, "val")

    # 1. Train Stream B0 (224x224)
    print("\n[*] Training Stream 1: EfficientNet-B0 (224x224)...")
    train_tf_224, val_tf_224 = get_transforms(224)
    train_ds_224 = ImageFolder(train_dir, transform=train_tf_224)
    val_ds_224 = ImageFolder(val_dir, transform=val_tf_224)

    train_loader_224 = DataLoader(train_ds_224, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader_224 = DataLoader(val_ds_224, batch_size=batch_size, shuffle=False, num_workers=0)

    model_b0 = DeepfakeClassifier("efficientnet_b0", pretrained=True, drop_rate=0.35).to(device)

    # Class balance weights
    targets = [s[1] for s in train_ds_224.samples]
    class_counts = np.bincount(targets)
    if len(class_counts) > 1 and min(class_counts) > 0:
        class_weights = len(targets) / (len(class_counts) * class_counts.astype(np.float32))
        weight_tensor = torch.tensor(class_weights, dtype=torch.float, device=device)
        criterion = nn.CrossEntropyLoss(weight=weight_tensor, label_smoothing=0.03)
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer_b0 = torch.optim.AdamW(model_b0.parameters(), lr=lr, weight_decay=1e-2)
    scheduler_b0 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_b0, T_max=epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == "cuda"))

    best_val_acc_b0 = 0.0
    best_weights_b0 = None

    for epoch in range(1, epochs + 1):
        model_b0.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for bx, by in train_loader_224:
            bx, by = bx.to(device), by.to(device)
            optimizer_b0.zero_grad()
            with torch.amp.autocast('cuda', enabled=(device.type == "cuda")):
                out = model_b0(bx)
                loss = criterion(out, by)

            scaler.scale(loss).backward()
            scaler.step(optimizer_b0)
            scaler.update()

            running_loss += loss.item() * bx.size(0)
            preds = out.argmax(dim=1)
            correct += (preds == by).sum().item()
            total += bx.size(0)

        scheduler_b0.step()

        # Validation
        model_b0.eval()
        val_preds, val_targets, val_probs_real = [], [], []
        with torch.no_grad():
            for bx, by in val_loader_224:
                bx = bx.to(device)
                with torch.amp.autocast('cuda', enabled=(device.type == "cuda")):
                    out = model_b0(bx)
                probs = F.softmax(out, dim=1)
                preds = out.argmax(dim=1)
                val_probs_real.extend(probs[:, 1].cpu().numpy())
                val_preds.extend(preds.cpu().numpy())
                val_targets.extend(by.numpy())

        val_acc = accuracy_score(val_targets, val_preds) * 100.0
        train_acc = (correct / total) * 100.0 if total > 0 else 0

        if val_acc >= best_val_acc_b0 or best_weights_b0 is None:
            best_val_acc_b0 = val_acc
            best_weights_b0 = {k: v.cpu().clone() for k, v in model_b0.state_dict().items()}

        if epoch % 3 == 0 or epoch == epochs:
            print(f"  [B0] Epoch {epoch:02d}/{epochs:02d} | Train Acc: {train_acc:.1f}% | Val Acc: {val_acc:.2f}% (Best: {best_val_acc_b0:.2f}%)", flush=True)

    # 2. Train Stream B3 (288x288)
    print("\n[*] Training Stream 2: EfficientNet-B3 (288x288)...")
    train_tf_288, val_tf_288 = get_transforms(288)
    train_ds_288 = ImageFolder(train_dir, transform=train_tf_288)
    val_ds_288 = ImageFolder(val_dir, transform=val_tf_288)

    train_loader_288 = DataLoader(train_ds_288, batch_size=max(8, batch_size // 2), shuffle=True, num_workers=0)
    val_loader_288 = DataLoader(val_ds_288, batch_size=max(8, batch_size // 2), shuffle=False, num_workers=0)

    model_b3 = DeepfakeClassifier("efficientnet_b3", pretrained=True, drop_rate=0.35).to(device)
    optimizer_b3 = torch.optim.AdamW(model_b3.parameters(), lr=lr * 0.8, weight_decay=1e-2)
    scheduler_b3 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_b3, T_max=epochs, eta_min=1e-6)

    best_val_acc_b3 = 0.0
    best_weights_b3 = None

    for epoch in range(1, epochs + 1):
        model_b3.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for bx, by in train_loader_288:
            bx, by = bx.to(device), by.to(device)
            optimizer_b3.zero_grad()
            with torch.amp.autocast('cuda', enabled=(device.type == "cuda")):
                out = model_b3(bx)
                loss = criterion(out, by)

            scaler.scale(loss).backward()
            scaler.step(optimizer_b3)
            scaler.update()

            running_loss += loss.item() * bx.size(0)
            preds = out.argmax(dim=1)
            correct += (preds == by).sum().item()
            total += bx.size(0)

        scheduler_b3.step()

        # Validation
        model_b3.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for bx, by in val_loader_288:
                bx = bx.to(device)
                with torch.amp.autocast('cuda', enabled=(device.type == "cuda")):
                    out = model_b3(bx)
                preds = out.argmax(dim=1)
                val_preds.extend(preds.cpu().numpy())
                val_targets.extend(by.numpy())

        val_acc = accuracy_score(val_targets, val_preds) * 100.0
        train_acc = (correct / total) * 100.0 if total > 0 else 0

        if val_acc >= best_val_acc_b3 or best_weights_b3 is None:
            best_val_acc_b3 = val_acc
            best_weights_b3 = {k: v.cpu().clone() for k, v in model_b3.state_dict().items()}

        if epoch % 3 == 0 or epoch == epochs:
            print(f"  [B3] Epoch {epoch:02d}/{epochs:02d} | Train Acc: {train_acc:.1f}% | Val Acc: {val_acc:.2f}% (Best: {best_val_acc_b3:.2f}%)", flush=True)

    # 3. Ensemble Evaluation on Validation Set
    print("\n" + "=" * 65)
    print("STEP 3: ENSEMBLE EVALUATION & BENCHMARKING")
    print("=" * 65)

    model_b0.load_state_dict(best_weights_b0)
    model_b3.load_state_dict(best_weights_b3)
    model_b0.eval()
    model_b3.eval()

    val_probs_b0, val_targets = [], []
    with torch.no_grad():
        for bx, by in val_loader_224:
            bx = bx.to(device)
            out = model_b0(bx)
            val_probs_b0.append(F.softmax(out, dim=1).cpu())
            val_targets.extend(by.numpy())
    val_probs_b0 = torch.cat(val_probs_b0, dim=0)

    val_probs_b3 = []
    with torch.no_grad():
        for bx, _ in val_loader_288:
            bx = bx.to(device)
            out = model_b3(bx)
            val_probs_b3.append(F.softmax(out, dim=1).cpu())
    val_probs_b3 = torch.cat(val_probs_b3, dim=0)

    comb_probs = 0.35 * val_probs_b0 + 0.65 * val_probs_b3
    val_targets = np.array(val_targets)
    fake_idx = train_ds_224.class_to_idx["fake"]
    real_idx = train_ds_224.class_to_idx["real"]

    probs_real = comb_probs[:, real_idx].numpy()
    preds = (probs_real >= 0.50).astype(int)

    acc = accuracy_score(val_targets, preds) * 100.0
    try:
        auc = roc_auc_score(val_targets, probs_real) * 100.0
    except Exception:
        auc = acc

    prec = precision_score(val_targets, preds, pos_label=fake_idx, zero_division=0) * 100.0
    rec = recall_score(val_targets, preds, pos_label=fake_idx, zero_division=0) * 100.0
    f1 = f1_score(val_targets, preds, pos_label=fake_idx, zero_division=0) * 100.0

    real_mask = (val_targets == real_idx)
    fake_mask = (val_targets == fake_idx)
    real_acc = (preds[real_mask] == val_targets[real_mask]).mean() * 100.0 if np.any(real_mask) else 100.0
    fake_acc = (preds[fake_mask] == val_targets[fake_mask]).mean() * 100.0 if np.any(fake_mask) else 100.0

    print(f"Overall Video Validation Accuracy : {acc:.2f}%")
    print(f"Authentic (Real) Accuracy          : {real_acc:.2f}%")
    print(f"Synthetic (Fake) Accuracy          : {fake_acc:.2f}%")
    print(f"Validation ROC-AUC                 : {auc:.2f}%")
    print(f"Deepfake Precision                 : {prec:.2f}%")
    print(f"Deepfake Recall                    : {rec:.2f}%")
    print(f"Deepfake F1-Score                  : {f1:.2f}%")

    # 4. Save Checkpoint
    if os.path.exists(OUTPUT_MODEL_PATH):
        backup_path = OUTPUT_MODEL_PATH.replace(".pt", "_backup.pt")
        shutil.copyfile(OUTPUT_MODEL_PATH, backup_path)
        print(f"[+] Saved previous model backup to {backup_path}")

    checkpoint = {
        "is_ensemble": True,
        "model_name": "Video-Ensemble(EfficientNet-B0 + EfficientNet-B3)",
        "model_b0_state_dict": best_weights_b0,
        "model_b3_state_dict": best_weights_b3,
        "weight_b0": 0.35,
        "weight_b3": 0.65,
        "threshold": 0.50,
        "class_to_idx": train_ds_224.class_to_idx,
        "idx_to_class": {v: k for k, v in train_ds_224.class_to_idx.items()},
        "test_acc": round(acc, 2),
        "test_auc": round(auc, 2)
    }
    torch.save(checkpoint, OUTPUT_MODEL_PATH)
    print(f"[+] Successfully saved updated model weights: {OUTPUT_MODEL_PATH}")

    # 5. Update model_metadata.json
    metadata = {
        "model_name": "Video-Trained Multi-Scale Forensic Ensemble",
        "is_ensemble": True,
        "metrics": {
            "test_accuracy": round(acc, 2),
            "real_accuracy": round(real_acc, 2),
            "fake_accuracy": round(fake_acc, 2),
            "test_auc": round(auc, 2),
            "precision": round(prec, 2),
            "recall": round(rec, 2),
            "f1_score": round(f1, 2)
        },
        "trained_on": "Video Frames (Haar/Face extracted)",
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"[+] Updated metadata in {METADATA_PATH}")

    return metadata


# ============================================================
# CLI ENTRYPOINT
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="End-to-End Deepfake Video Training Pipeline")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs per stream")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--frames_per_video", type=int, default=20, help="Number of face frames to extract per video")
    parser.add_argument("--video_dir", type=str, default=None, help="Optional custom folder containing videos")
    args = parser.parse_args()

    print("=" * 70)
    print("DEEPFAKE INVESTIGATION SYSTEM — AUTOMATED VIDEO TRAINING ENGINE")
    print("=" * 70)

    catalog = scan_available_videos()
    if args.video_dir and os.path.exists(args.video_dir):
        for f in os.listdir(args.video_dir):
            if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
                p = os.path.join(args.video_dir, f)
                lbl = "real" if "real" in f.lower() else "fake"
                catalog.append({"path": p, "label": lbl, "source": "custom_dir"})

    if not catalog:
        print("[!] No videos found in sample_videos or DeepFake-Detect/train_sample_videos.")
        return

    print(f"[*] Discovered {len(catalog)} videos for training:")
    for v in catalog:
        print(f"    • {os.path.basename(v['path'])} -> {v['label'].upper()} ({v['source']})")

    # Step 1: Extract faces & construct video dataset
    dataset_dir = build_video_dataset(catalog, frames_per_video=args.frames_per_video)

    # Step 2: Train & fine-tune the multi-scale ensemble
    train_video_model(dataset_dir, epochs=args.epochs, batch_size=args.batch_size)

    print("\n" + "=" * 70)
    print("VIDEO TRAINING COMPLETE SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    main()
