# predictor.py
# High-Accuracy Deepfake Predictor with Dual-Stream Frequency Engine & Memory-Optimized Inference

import os
import sys
import numpy as np
import cv2
from PIL import Image

# Enforce low memory overhead for free-tier cloud containers (512MB limit)
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"
os.environ["MALLOC_TRIM_THRESHOLD_"] = "100000"

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

device = None
model = None
model_type = None
torch = None
tf = None
class_to_idx = {"fake": 0, "real": 1}
idx_to_class = {0: "fake", 1: "real"}
IMG_SIZE = 224
eval_transform = None


def compute_highpass_residual(tensors, torch_module):
    """
    Computes high-pass frequency residual map to extract compression artifacts.
    """
    F = torch_module.nn.functional
    kernel = torch_module.tensor([
        [1, 4, 6, 4, 1],
        [4, 16, 24, 16, 4],
        [6, 24, 36, 24, 6],
        [4, 16, 24, 16, 4],
        [1, 4, 6, 4, 1]
    ], dtype=torch_module.float32, device=tensors.device) / 256.0

    kernel = kernel.repeat(3, 1, 1, 1)
    blurred = F.conv2d(tensors, kernel, padding=2, groups=3)
    return tensors - blurred


def get_dual_stream_classes(torch_mod, timm_mod):
    nn = torch_mod.nn
    F = torch_mod.nn.functional

    class DualStreamDeepfakeDetector(nn.Module):
        def __init__(self, backbone_name="efficientnet_b0", num_classes=2, drop_rate=0.40):
            super().__init__()
            self.rgb_backbone = timm_mod.create_model(backbone_name, pretrained=False, num_classes=0, drop_rate=drop_rate)
            self.res_backbone = timm_mod.create_model(backbone_name, pretrained=False, num_classes=0, drop_rate=drop_rate)

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
            res = compute_highpass_residual(x, torch_mod)
            feat_rgb = self.rgb_backbone(x)
            feat_res = self.res_backbone(res)
            combined = torch_mod.cat([feat_rgb, feat_res], dim=1)
            return self.classifier(combined)

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

    return DualStreamDeepfakeDetector, ForensicEnsembleDetector


DualStreamDetector = None
EnsembleDetector = None
ForensicEnsembleDetector = None
DualStreamDeepfakeDetector = None


def load_detector_model():
    global model, model_type, class_to_idx, idx_to_class, IMG_SIZE, eval_transform, device, torch, tf
    global DualStreamDetector, EnsembleDetector, ForensicEnsembleDetector, DualStreamDeepfakeDetector

    # 1. Try loading PyTorch model if .pt exists
    if os.path.exists(PYTORCH_MODEL_PATH):
        try:
            print(f"[*] Loading PyTorch Deepfake Detector: {PYTORCH_MODEL_PATH}")
            import torch as _torch
            import torchvision.transforms as transforms
            import timm

            torch = _torch
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            checkpoint = torch.load(PYTORCH_MODEL_PATH, map_location=device)
            class_to_idx = checkpoint.get("class_to_idx", {"fake": 0, "real": 1})
            idx_to_class = {v: k for k, v in class_to_idx.items()}
            IMG_SIZE = checkpoint.get("img_size", 224)

            eval_transform = transforms.Compose([
                transforms.Resize((IMG_SIZE, IMG_SIZE), Image.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

            DualStreamDetector, EnsembleDetector = get_dual_stream_classes(torch, timm)
            DualStreamDeepfakeDetector = DualStreamDetector
            ForensicEnsembleDetector = EnsembleDetector

            if checkpoint.get("is_ensemble"):
                m_a_name = checkpoint.get("model_a_name", "efficientnet_b0")
                m_b_name = checkpoint.get("model_b_name", "resnet34")
                m_a = DualStreamDetector(backbone_name=m_a_name, num_classes=2)
                m_b = DualStreamDetector(backbone_name=m_b_name, num_classes=2)
                m_a.load_state_dict(checkpoint["model_a_state_dict"])
                m_b.load_state_dict(checkpoint["model_b_state_dict"])
                w_a = checkpoint.get("weight_a", 0.55)
                w_b = checkpoint.get("weight_b", 0.45)
                py_model = EnsembleDetector(m_a, m_b, weight_a=w_a, weight_b=w_b)
                arch = f"Ensemble({m_a_name} + {m_b_name})"
            elif checkpoint.get("is_dual_stream"):
                backbone = checkpoint.get("backbone", "efficientnet_b0")
                py_model = DualStreamDetector(backbone_name=backbone, num_classes=2)
                py_model.load_state_dict(checkpoint["model_state_dict"])
                arch = f"DualStream({backbone})"
            else:
                arch = checkpoint.get("model_name", "efficientnet_b0")
                py_model = timm.create_model(arch, pretrained=False, num_classes=2)
                py_model.load_state_dict(checkpoint["model_state_dict"])
                arch = f"Single({arch})"

            py_model.to(device)
            py_model.eval()

            model = py_model
            model_type = "pytorch"
            print(f"[+] PyTorch model loaded successfully on {device} (Arch: {arch})")
            return
        except Exception as e:
            print(f"[!] Warning: Failed loading PyTorch model: {e}")

    # 2. Fallback to Keras model (load with compile=False and single-thread for memory safety)
    keras_candidates = [
        os.path.join(BASE_DIR, "deepfake_efficientnet_v3.keras"),
        os.path.join(BASE_DIR, "deepfake_efficientnet_v2.keras"),
        os.path.join(BASE_DIR, "deepfake_efficientnet.keras")
    ]
    keras_path = next((p for p in keras_candidates if os.path.exists(p)), None)

    if keras_path:
        try:
            print(f"[*] Loading Keras fallback model: {keras_path}")
            import tensorflow as tf
            tf.config.threading.set_inter_op_parallelism_threads(1)
            tf.config.threading.set_intra_op_parallelism_threads(1)
            model = tf.keras.models.load_model(keras_path, compile=False)
            model_type = "keras"
            in_shape = getattr(model, "input_shape", None)
            if in_shape and len(in_shape) >= 3 and in_shape[1] is not None:
                IMG_SIZE = int(in_shape[1])
            else:
                IMG_SIZE = 224
            print(f"[+] Keras model loaded successfully (lightweight mode, size={IMG_SIZE}x{IMG_SIZE})")
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
    Optimized for high-speed CPU execution.
    """
    if face_cascade is None or img_cv is None:
        return img_cv

    try:
        img_h, img_w = img_cv.shape[:2]
        # Downscale for ultra-fast face detection if image is large
        scale = 1.0
        max_dim = max(img_h, img_w)
        if max_dim > 640:
            scale = 640.0 / max_dim
            small_w = int(img_w * scale)
            small_h = int(img_h * scale)
            small_img = cv2.resize(img_cv, (small_w, small_h), interpolation=cv2.INTER_AREA)
        else:
            small_img = img_cv

        gray = cv2.cvtColor(small_img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.15,
            minNeighbors=3,
            minSize=(30, 30)
        )
        if len(faces) == 0:
            return img_cv

        largest_face = max(faces, key=lambda r: r[2] * r[3])
        x, y, w, h = largest_face

        # Rescale bounding box to original dimensions
        x = int(x / scale)
        y = int(y / scale)
        w = int(w / scale)
        h = int(h / scale)

        pad_w = int(w * padding_ratio)
        pad_h = int(h * padding_ratio)

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