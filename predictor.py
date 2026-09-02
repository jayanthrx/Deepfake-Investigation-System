# predictor.py
# High-Accuracy Deepfake Predictor with Dual-Stream Frequency Engine & PyTorch GPU Acceleration

import os
import sys
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import timm

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="backslashreplace")

# ==============================
# CONFIGURATION & MODEL DEFINITION
# ==============================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PYTORCH_MODEL_PATH = os.path.join(BASE_DIR, "deepfake_detector_best.pt")
KERAS_MODEL_PATH = os.path.join(BASE_DIR, "deepfake_efficientnet_v3.keras")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = None
model_type = None
class_to_idx = {"fake": 0, "real": 1}
idx_to_class = {0: "fake", 1: "real"}
IMG_SIZE = 224


def compute_highpass_residual(tensors):
    """
    Computes high-pass frequency residual map to extract compression artifacts.
    """
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


class DualStreamDeepfakeDetector(nn.Module):
    """
    Dual-Stream network combining RGB Spatial Stream + High-Frequency Artifact Stream.
    """
    def __init__(self, backbone_name="efficientnet_b0", num_classes=2, drop_rate=0.40):
        super().__init__()
        self.rgb_backbone = timm.create_model(backbone_name, pretrained=False, num_classes=0, drop_rate=drop_rate)
        self.res_backbone = timm.create_model(backbone_name, pretrained=False, num_classes=0, drop_rate=drop_rate)

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


class ForensicEnsembleDetector(nn.Module):
    """
    Dual-Model Ensemble with calibrated probabilistic fusion.
    """
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


# Transform for PyTorch inference
eval_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE), Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def load_detector_model():
    global model, model_type, class_to_idx, idx_to_class, IMG_SIZE, eval_transform

    # 1. Try loading best PyTorch model
    if os.path.exists(PYTORCH_MODEL_PATH):
        try:
            print(f"[*] Loading PyTorch Deepfake Detector: {PYTORCH_MODEL_PATH}")
            checkpoint = torch.load(PYTORCH_MODEL_PATH, map_location=device)
            class_to_idx = checkpoint.get("class_to_idx", {"fake": 0, "real": 1})
            idx_to_class = {v: k for k, v in class_to_idx.items()}
            IMG_SIZE = checkpoint.get("img_size", 224)

            if checkpoint.get("is_ensemble"):
                m_a_name = checkpoint.get("model_a_name", "efficientnet_b0")
                m_b_name = checkpoint.get("model_b_name", "resnet34")
                m_a = DualStreamDeepfakeDetector(backbone_name=m_a_name, num_classes=2)
                m_b = DualStreamDeepfakeDetector(backbone_name=m_b_name, num_classes=2)
                m_a.load_state_dict(checkpoint["model_a_state_dict"])
                m_b.load_state_dict(checkpoint["model_b_state_dict"])
                w_a = checkpoint.get("weight_a", 0.55)
                w_b = checkpoint.get("weight_b", 0.45)
                py_model = ForensicEnsembleDetector(m_a, m_b, weight_a=w_a, weight_b=w_b)
                arch = f"Ensemble({m_a_name} + {m_b_name})"
            elif checkpoint.get("is_dual_stream"):
                backbone = checkpoint.get("backbone", "efficientnet_b0")
                py_model = DualStreamDeepfakeDetector(backbone_name=backbone, num_classes=2)
                py_model.load_state_dict(checkpoint["model_state_dict"])
                arch = f"DualStream({backbone})"
            else:
                arch = checkpoint.get("model_name", "efficientnet_b0")
                py_model = timm.create_model(arch, pretrained=False, num_classes=2)
                py_model.load_state_dict(checkpoint["model_state_dict"])

            py_model.to(device)
            py_model.eval()

            model = py_model
            model_type = "pytorch"
            print(f"[+] PyTorch model loaded successfully on {device} (Arch: {arch})")
            return
        except Exception as e:
            print(f"[!] Warning: Failed loading PyTorch model: {e}")

    # 2. Fallback to Keras model if available
    keras_candidates = [
        os.path.join(BASE_DIR, "deepfake_efficientnet_v3.keras"),
        os.path.join(BASE_DIR, "deepfake_efficientnet_v2.keras"),
        os.path.join(BASE_DIR, "deepfake_efficientnet.keras")
    ]
    keras_path = next((p for p in keras_candidates if os.path.exists(p)), None)

    if keras_path:
        try:
            import tensorflow as tf
            print(f"[*] Loading Keras fallback model: {keras_path}")
            model = tf.keras.models.load_model(keras_path)
            model_type = "keras"
            print("[+] Keras model loaded successfully")
            return
        except Exception as e:
            print(f"[!] Warning: Failed loading Keras model: {e}")

    print("[!] ERROR: No deepfake detector weights found!")

load_detector_model()


# ==============================
# FACE EXTRACTION
# ==============================

face_cascade = None
try:
    if hasattr(cv2, "CascadeClassifier") and hasattr(cv2, "data"):
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
except Exception as fe:
    print("Face cascade init note:", fe)


def extract_face(img_cv, padding_ratio=0.22):
    """
    Extracts the most prominent face with contextual padding to capture
    blending boundaries, jawlines, and facial perimeters.
    """
    if face_cascade is None or img_cv is None:
        return img_cv

    try:
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=4,
            minSize=(60, 60)
        )
        if len(faces) == 0:
            return img_cv

        largest_face = max(faces, key=lambda r: r[2] * r[3])
        x, y, w, h = largest_face

        pad_w = int(w * padding_ratio)
        pad_h = int(h * padding_ratio)

        img_h, img_w = img_cv.shape[:2]
        x1 = max(0, x - pad_w)
        y1 = max(0, y - pad_h)
        x2 = min(img_w, x + w + pad_w)
        y2 = min(img_h, y + h + pad_h)

        cropped = img_cv[y1:y2, x1:x2]
        if cropped.size > 0 and cropped.shape[0] > 30 and cropped.shape[1] > 30:
            return cropped
    except Exception as e:
        print("Face extraction notice:", e)

    return img_cv


# ==============================
# RISK CALCULATION
# ==============================

def calculate_risk(label, confidence):
    if label.lower() == "real":
        return "Low"

    if label.lower() == "uncertain":
        return "Medium"

    if confidence >= 85:
        return "Very High"
    elif confidence >= 70:
        return "High"
    elif confidence >= 55:
        return "Medium"
    else:
        return "Low"


# ==============================
# IMAGE PREDICTION
# ==============================

def predict_image(image_path, use_tta=True):
    """
    Analyzes an image and returns:
    (label, confidence, risk, fake_probability, real_probability)
    """
    global model, model_type

    if not os.path.exists(image_path):
        return ("File Not Found", 0.0, "Low", 0.0, 0.0)

    if model is None:
        load_detector_model()
        if model is None:
            return ("Model Not Loaded", 0.0, "Low", 0.0, 0.0)

    try:
        cv_img = cv2.imread(image_path)
        if cv_img is None:
            return ("Invalid Image", 0.0, "Low", 0.0, 0.0)

        cropped_face = extract_face(cv_img)
        rgb_face = cv2.cvtColor(cropped_face, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_face)

    except Exception as e:
        print("Image Loading Error:", e)
        return ("Invalid Image", 0.0, "Low", 0.0, 0.0)

    # ----------------------------------------------------
    # PYTORCH INFERENCE WITH TEST-TIME AUGMENTATION (TTA)
    # ----------------------------------------------------
    if model_type == "pytorch":
        try:
            tensor_orig = eval_transform(pil_img)
            
            if use_tta:
                tensor_flip = torch.flip(tensor_orig, dims=[-1])
                batch = torch.stack([tensor_orig, tensor_flip]).to(device)
            else:
                batch = tensor_orig.unsqueeze(0).to(device)

            with torch.no_grad():
                outputs = model(batch)
                if isinstance(model, ForensicEnsembleDetector):
                    probs = outputs.mean(dim=0)
                else:
                    probs = torch.softmax(outputs, dim=1).mean(dim=0)

            real_idx = class_to_idx.get("real", 1)
            fake_idx = class_to_idx.get("fake", 0)

            real_probability = float(probs[real_idx].item() * 100.0)
            fake_probability = float(probs[fake_idx].item() * 100.0)

        except Exception as e:
            print("PyTorch Inference Error:", e)
            return ("Inference Error", 0.0, "Low", 0.0, 0.0)

    # ----------------------------------------------------
    # KERAS INFERENCE FALLBACK
    # ----------------------------------------------------
    else:
        try:
            from tensorflow.keras.applications.efficientnet import preprocess_input
            resized = cv2.resize(rgb_face, (IMG_SIZE, IMG_SIZE))
            img_arr = np.expand_dims(resized.astype(np.float32), axis=0)
            img_arr = preprocess_input(img_arr)
            score = float(model.predict(img_arr, verbose=0)[0][0])

            real_probability = score * 100.0
            fake_probability = (1.0 - score) * 100.0
        except Exception as e:
            print("Keras Inference Error:", e)
            return ("Inference Error", 0.0, "Low", 0.0, 0.0)

    # ==============================
    # CALIBRATED DECISION LOGIC
    # ==============================
    if fake_probability >= 52.0:
        label = "Fake"
        confidence = fake_probability
    elif real_probability >= 52.0:
        label = "Real"
        confidence = real_probability
    else:
        label = "Uncertain"
        confidence = max(fake_probability, real_probability)

    confidence = round(float(confidence), 2)
    fake_probability = round(float(fake_probability), 2)
    real_probability = round(float(real_probability), 2)

    risk = calculate_risk(label, confidence)

    return (
        label,
        confidence,
        risk,
        fake_probability,
        real_probability
    )