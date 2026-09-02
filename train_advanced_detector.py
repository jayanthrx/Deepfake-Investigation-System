"""
train_advanced_detector.py
Ultra-Fast Ampere TensorCore Accelerated Deepfake Detector Training Pipeline
Uses mixed precision (FP16), in-memory GPU caching, rich augmentations, and optimal calibration.
"""

import os
import sys
import json
import time
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import timm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace", line_buffering=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="backslashreplace", line_buffering=True)


def load_tensor_dataset(root_dir, img_size=224):
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
                    img_rgb = img.convert("RGB").resize((img_size, img_size), Image.BILINEAR)
                    t = normalize(to_tensor(img_rgb))
                    tensor_list.append(t)
                    label_list.append(label)
            except Exception:
                pass

    all_tensors = torch.stack(tensor_list)
    all_labels = torch.tensor(label_list, dtype=torch.long)
    return all_tensors, all_labels, class_to_idx


def apply_gpu_augmentations(batch_x):
    # 1. Random Horizontal Flip
    flip_mask = torch.rand(batch_x.size(0), device=batch_x.device) > 0.5
    if flip_mask.any():
        batch_x[flip_mask] = torch.flip(batch_x[flip_mask], dims=[-1])

    # 2. Random Brightness / Contrast Scaling
    scale = 1.0 + (torch.rand((batch_x.size(0), 1, 1, 1), device=batch_x.device) - 0.5) * 0.25
    batch_x = batch_x * scale

    # 3. Subtle Gaussian Noise
    if torch.rand(1).item() > 0.5:
        noise = torch.randn_like(batch_x) * 0.015
        batch_x = batch_x + noise

    return batch_x


def main():
    print("=" * 70, flush=True)
    print("GPU TENSOR-CORE ACCELERATED DEEPFAKE DETECTOR TRAINING", flush=True)
    print("=" * 70, flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Compute Device: {device}", flush=True)
    if device.type == "cuda":
        print(f"[*] GPU: {torch.cuda.get_device_name(0)}", flush=True)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_base = os.path.join(base_dir, "DeepFake-Detect", "split_dataset")
    train_dir = os.path.join(dataset_base, "train")
    val_dir = os.path.join(dataset_base, "val")
    test_dir = os.path.join(dataset_base, "test")

    IMG_SIZE = 224
    BATCH_SIZE = 32
    NUM_EPOCHS = 25
    LEARNING_RATE = 3e-4
    WEIGHT_DECAY = 1e-2
    MODEL_NAME = "efficientnet_b0"
    OUTPUT_MODEL_PATH = os.path.join(base_dir, "deepfake_detector_best.pt")
    METADATA_PATH = os.path.join(base_dir, "model_metadata.json")

    print("[*] Preloading and normalizing datasets into memory...", flush=True)
    t0 = time.time()
    train_x, train_y, class_to_idx = load_tensor_dataset(train_dir, img_size=IMG_SIZE)
    val_x, val_y, _ = load_tensor_dataset(val_dir, img_size=IMG_SIZE)
    test_x, test_y, _ = load_tensor_dataset(test_dir, img_size=IMG_SIZE)

    idx_to_class = {v: k for k, v in class_to_idx.items()}
    print(f"[*] Datasets Loaded in {time.time() - t0:.2f}s | Train: {len(train_x)}, Val: {len(val_x)}, Test: {len(test_x)}", flush=True)

    # Move tensors to GPU VRAM
    train_x = train_x.to(device)
    train_y = train_y.to(device)
    val_x = val_x.to(device)
    val_y = val_y.to(device)
    test_x = test_x.to(device)
    test_y = test_y.to(device)

    # Class Weights for balanced loss
    class_counts = np.bincount(train_y.cpu().numpy())
    class_weights = len(train_y) / (len(class_counts) * class_counts.astype(np.float32))
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float, device=device)

    # Model & Optimizer
    print(f"[*] Initializing {MODEL_NAME} with pre-trained weights...", flush=True)
    model = timm.create_model(MODEL_NAME, pretrained=True, num_classes=2, drop_rate=0.3)
    model.to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor, label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == "cuda"))

    best_val_auc = 0.0
    best_val_acc = 0.0
    best_epoch = 0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "val_auc": []}

    N_train = train_x.size(0)
    N_val = val_x.size(0)

    print("\n" + "=" * 70, flush=True)
    print(f"TRAINING STARTED ({NUM_EPOCHS} Epochs on {device})", flush=True)
    print("=" * 70, flush=True)

    train_start = time.time()

    for epoch in range(1, NUM_EPOCHS + 1):
        t_epoch = time.time()
        model.train()

        perm = torch.randperm(N_train, device=device)
        running_loss = 0.0
        train_correct = 0

        for i in range(0, N_train, BATCH_SIZE):
            batch_indices = perm[i:i + BATCH_SIZE]
            bx = train_x[batch_indices].clone()
            by = train_y[batch_indices]

            bx = apply_gpu_augmentations(bx)

            optimizer.zero_grad()
            with torch.amp.autocast('cuda', enabled=(device.type == "cuda")):
                outputs = model(bx)
                loss = criterion(outputs, by)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * bx.size(0)
            preds = outputs.argmax(dim=1)
            train_correct += (preds == by).sum().item()

        scheduler.step()

        epoch_train_loss = running_loss / N_train
        epoch_train_acc = (train_correct / N_train) * 100.0

        # Validation Step
        model.eval()
        val_loss = 0.0
        val_probs_real = []
        val_preds = []

        with torch.no_grad():
            for i in range(0, N_val, BATCH_SIZE):
                bx = val_x[i:i + BATCH_SIZE]
                by = val_y[i:i + BATCH_SIZE]

                with torch.amp.autocast('cuda', enabled=(device.type == "cuda")):
                    outputs = model(bx)
                    loss = criterion(outputs, by)

                val_loss += loss.item() * bx.size(0)
                probs = torch.softmax(outputs, dim=1)
                preds = outputs.argmax(dim=1)

                val_probs_real.extend(probs[:, class_to_idx["real"]].cpu().numpy())
                val_preds.extend(preds.cpu().numpy())

        epoch_val_loss = val_loss / N_val
        val_y_cpu = val_y.cpu().numpy()
        epoch_val_acc = accuracy_score(val_y_cpu, val_preds) * 100.0
        epoch_val_auc = roc_auc_score(val_y_cpu, val_probs_real) * 100.0

        history["train_loss"].append(epoch_train_loss)
        history["train_acc"].append(epoch_train_acc)
        history["val_loss"].append(epoch_val_loss)
        history["val_acc"].append(epoch_val_acc)
        history["val_auc"].append(epoch_val_auc)

        ep_duration = time.time() - t_epoch
        print(f"Epoch [{epoch:02d}/{NUM_EPOCHS:02d}] ({ep_duration:.2f}s) | "
              f"Train Loss: {epoch_train_loss:.4f}, Acc: {epoch_train_acc:.2f}% | "
              f"Val Loss: {epoch_val_loss:.4f}, Acc: {epoch_val_acc:.2f}%, AUC: {epoch_val_auc:.2f}%", flush=True)

        if epoch_val_auc > best_val_auc:
            best_val_auc = epoch_val_auc
            best_val_acc = epoch_val_acc
            best_epoch = epoch
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "model_name": MODEL_NAME,
                "img_size": IMG_SIZE,
                "class_to_idx": class_to_idx,
                "val_auc": epoch_val_auc,
                "val_acc": epoch_val_acc
            }, OUTPUT_MODEL_PATH)
            print(f"   --> [CHECKPOINT SAVED] Val AUC: {epoch_val_auc:.2f}%, Val Acc: {epoch_val_acc:.2f}%", flush=True)

    print("\n" + "=" * 70, flush=True)
    print(f"TRAINING COMPLETE in {time.time() - train_start:.2f} seconds", flush=True)
    print(f"Best Validation Checkpoint: Epoch {best_epoch} (AUC: {best_val_auc:.2f}%, Acc: {best_val_acc:.2f}%)", flush=True)
    print("=" * 70, flush=True)

    # Test Set Benchmark
    print("\n[*] Evaluating on Test Set with Best Checkpoint...", flush=True)
    checkpoint = torch.load(OUTPUT_MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    test_preds_list = []
    test_probs_real_list = []
    N_test = test_x.size(0)

    with torch.no_grad():
        for i in range(0, N_test, BATCH_SIZE):
            bx = test_x[i:i + BATCH_SIZE]
            with torch.amp.autocast('cuda', enabled=(device.type == "cuda")):
                outputs = model(bx)
            probs = torch.softmax(outputs, dim=1)
            preds = outputs.argmax(dim=1)

            test_probs_real_list.extend(probs[:, class_to_idx["real"]].cpu().numpy())
            test_preds_list.extend(preds.cpu().numpy())

    test_y_cpu = test_y.cpu().numpy()
    test_acc = accuracy_score(test_y_cpu, test_preds_list) * 100.0
    test_auc = roc_auc_score(test_y_cpu, test_probs_real_list) * 100.0
    fake_idx = class_to_idx["fake"]
    real_idx = class_to_idx["real"]

    test_prec = precision_score(test_y_cpu, test_preds_list, pos_label=fake_idx) * 100.0
    test_rec = recall_score(test_y_cpu, test_preds_list, pos_label=fake_idx) * 100.0
    test_f1 = f1_score(test_y_cpu, test_preds_list, pos_label=fake_idx) * 100.0
    cm = confusion_matrix(test_y_cpu, test_preds_list)

    real_mask = (test_y_cpu == real_idx)
    fake_mask = (test_y_cpu == fake_idx)
    real_acc = (np.array(test_preds_list)[real_mask] == test_y_cpu[real_mask]).mean() * 100.0
    fake_acc = (np.array(test_preds_list)[fake_mask] == test_y_cpu[fake_mask]).mean() * 100.0

    print("\n" + "=" * 70, flush=True)
    print("FINAL TEST SET BENCHMARK RESULTS", flush=True)
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

    metadata = {
        "model_name": MODEL_NAME,
        "img_size": IMG_SIZE,
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
        "history": history,
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"[*] Saved training metadata to {METADATA_PATH}", flush=True)


if __name__ == "__main__":
    main()
