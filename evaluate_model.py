"""
evaluate_model.py
Evaluates the deepfake detection model on the test dataset.
Computes accuracy, precision, recall, F1 score, ROC-AUC, and confusion matrix.
"""

import os
import sys
import torch
import numpy as np
from PIL import Image
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="backslashreplace")

from predictor import predict_image, model, model_type, PYTORCH_MODEL_PATH


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    test_folder = os.path.join(base_dir, "DeepFake-Detect", "split_dataset", "test")

    print("=" * 70)
    print("DEEPFAKE MODEL PERFORMANCE EVALUATION")
    print(f"Active Model: {PYTORCH_MODEL_PATH if os.path.exists(PYTORCH_MODEL_PATH) else 'Keras Backup'}")
    print(f"Inference Engine: {model_type}")
    print(f"Test Directory: {test_folder}")
    print("=" * 70)

    if not os.path.exists(test_folder):
        print(f"[!] ERROR: Test folder {test_folder} does not exist.")
        return

    labels_true = []
    labels_pred = []
    probs_fake = []

    real_correct = 0
    real_total = 0
    fake_correct = 0
    fake_total = 0
    total = 0
    correct = 0

    for folder in ["real", "fake"]:
        folder_path = os.path.join(test_folder, folder)
        if not os.path.exists(folder_path):
            print("Missing:", folder_path)
            continue

        for file in os.listdir(folder_path):
            image_path = os.path.join(folder_path, file)
            if not file.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            try:
                label, confidence, risk, fake_prob, real_prob = predict_image(image_path, use_tta=True)

                predicted_label = label.lower()
                true_label = folder.lower()

                labels_true.append(1 if true_label == "fake" else 0)
                labels_pred.append(1 if predicted_label == "fake" else 0)
                probs_fake.append(fake_prob / 100.0)

                total += 1
                is_correct = (predicted_label == true_label)
                if is_correct:
                    correct += 1

                if folder == "real":
                    real_total += 1
                    if is_correct:
                        real_correct += 1
                else:
                    fake_total += 1
                    if is_correct:
                        fake_correct += 1

                status_mark = "[PASS]" if is_correct else "[FAIL]"
                print(f"{status_mark} {file:<30} => Predicted: {label:<8} (Confidence: {confidence:.1f}%, Fake: {fake_prob:.1f}%, Real: {real_prob:.1f}%)")

            except Exception as e:
                print(f"[!] Error on {file}: {e}")

    if total == 0:
        print("[!] No test images evaluated.")
        return

    acc = (correct / total) * 100.0
    real_acc = (real_correct / real_total * 100.0) if real_total > 0 else 0.0
    fake_acc = (fake_correct / fake_total * 100.0) if fake_total > 0 else 0.0
    prec = precision_score(labels_true, labels_pred, zero_division=0) * 100.0
    rec = recall_score(labels_true, labels_pred, zero_division=0) * 100.0
    f1 = f1_score(labels_true, labels_pred, zero_division=0) * 100.0
    auc = roc_auc_score(labels_true, probs_fake) * 100.0
    cm = confusion_matrix(labels_true, labels_pred)

    print("\n" + "=" * 70)
    print("ACCURACY BENCHMARK & EVALUATION RESULTS")
    print("=" * 70)
    print(f"Total Test Accuracy : {acc:.2f}% ({correct}/{total})")
    print(f"Real Class Accuracy : {real_acc:.2f}% ({real_correct}/{real_total})")
    print(f"Fake Class Accuracy : {fake_acc:.2f}% ({fake_correct}/{fake_total})")
    print(f"Test ROC-AUC        : {auc:.2f}%")
    print(f"Fake Precision      : {prec:.2f}%")
    print(f"Fake Recall         : {rec:.2f}%")
    print(f"F1-Score            : {f1:.2f}%")
    print(f"Confusion Matrix    :\n{cm}")
    print("=" * 70)


if __name__ == "__main__":
    main()