# predictor.py
# High-Accuracy Deepfake Predictor with Multi-Scale Forensic Concat-Pooling Ensemble

import os
import sys
import numpy as np
import cv2
from PIL import Image

# Offline mode to prevent hanging on huggingface hub
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="backslashreplace")

# ==============================
# CONFIGURATION & PATHS
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
eval_transform_b0 = None
eval_transform_b3 = None
is_ensemble_model = False


# ==============================
# MODEL ARCHITECTURES
# ==============================

def get_forensic_classes(torch_mod, timm_mod):
    nn = torch_mod.nn

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
            combined = torch_mod.cat([avg_f, max_f], dim=1)
            return self.fc(combined)

    class DeepfakeClassifier(nn.Module):
        def __init__(self, backbone_name="efficientnet_b0", pretrained=False, drop_rate=0.35):
            super().__init__()
            self.backbone_name = backbone_name
            self.backbone = timm_mod.create_model(backbone_name, pretrained=pretrained, num_classes=0, drop_rate=drop_rate)
            in_features = self.backbone.num_features
            self.head = ForensicHead(in_features, num_classes=2, drop_rate=drop_rate)

        def forward(self, x):
            feat = self.backbone.forward_features(x)
            return self.head(feat)

        @property
        def conv_head(self):
            if hasattr(self.backbone, "conv_head"):
                return self.backbone.conv_head
            elif hasattr(self.backbone, "head") and hasattr(self.backbone.head, "conv"):
                return self.backbone.head.conv
            elif hasattr(self.backbone, "blocks"):
                return self.backbone.blocks[-1]
            return None

    class EnsembleForensicDetector(nn.Module):
        def __init__(self, model_b0, model_b3, weight_b0=0.40, weight_b3=0.60):
            super().__init__()
            self.model_b0 = model_b0
            self.model_b3 = model_b3
            self.weight_b0 = weight_b0
            self.weight_b3 = weight_b3

        def forward(self, x_b0, x_b3=None):
            if x_b3 is None:
                x_b3 = x_b0
            prob_b0 = torch_mod.softmax(self.model_b0(x_b0), dim=1)
            prob_b3 = torch_mod.softmax(self.model_b3(x_b3), dim=1)
            return self.weight_b0 * prob_b0 + self.weight_b3 * prob_b3

        @property
        def conv_head(self):
            return self.model_b0.conv_head

    return ForensicHead, DeepfakeClassifier, EnsembleForensicDetector


# ==============================
# MODEL LOADER
# ==============================

def load_detector_model():
    global model, model_type, class_to_idx, idx_to_class, IMG_SIZE
    global eval_transform, eval_transform_b0, eval_transform_b3, is_ensemble_model
    global device, torch, tf

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

            norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            eval_transform_b0 = transforms.Compose([
                transforms.Resize((224, 224), Image.BICUBIC),
                transforms.ToTensor(),
                norm
            ])
            eval_transform_b3 = transforms.Compose([
                transforms.Resize((288, 288), Image.BICUBIC),
                transforms.ToTensor(),
                norm
            ])
            eval_transform = eval_transform_b0
            IMG_SIZE = 224

            ForensicHead, DeepfakeClassifier, EnsembleForensicDetector = get_forensic_classes(torch, timm)

            if checkpoint.get("is_ensemble"):
                is_ensemble_model = True
                m_b0 = DeepfakeClassifier("efficientnet_b0", pretrained=False)
                m_b3 = DeepfakeClassifier("efficientnet_b3", pretrained=False)
                m_b0.load_state_dict(checkpoint["model_b0_state_dict"])
                m_b3.load_state_dict(checkpoint["model_b3_state_dict"])
                w0 = checkpoint.get("weight_b0", 0.40)
                w3 = checkpoint.get("weight_b3", 0.60)
                py_model = EnsembleForensicDetector(m_b0, m_b3, weight_b0=w0, weight_b3=w3)
                arch = f"Ensemble(EfficientNet-B0 [{w0}] + EfficientNet-B3 [{w3}])"
            else:
                is_ensemble_model = False
                backbone = checkpoint.get("backbone_name", "efficientnet_b0")
                py_model = DeepfakeClassifier(backbone, pretrained=False)
                py_model.load_state_dict(checkpoint["model_state_dict"])
                arch = f"Single({backbone})"

            py_model.to(device)
            py_model.eval()

            model = py_model
            model_type = "pytorch"
            print(f"[+] PyTorch model loaded successfully on {device} (Arch: {arch})")
            return
        except Exception as e:
            print(f"[!] Warning: Failed loading PyTorch model: {e}")

    # Fallback to Keras model
    keras_candidates = [
        os.path.join(BASE_DIR, "deepfake_efficientnet_v3.keras"),
        os.path.join(BASE_DIR, "deepfake_efficientnet_v2.keras"),
        os.path.join(BASE_DIR, "deepfake_efficientnet.keras")
    ]
    keras_path = next((p for p in keras_candidates if os.path.exists(p)), None)

    if keras_path:
        try:
            print(f"[*] Loading Keras fallback model: {keras_path}")
            import tensorflow as _tf
            tf = _tf
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
    """
    if face_cascade is None or img_cv is None:
        return img_cv

    try:
        img_h, img_w = img_cv.shape[:2]
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
            with torch.no_grad():
                if is_ensemble_model:
                    # Multi-scale evaluation for ensemble
                    t_b0_orig = eval_transform_b0(pil_img)
                    t_b3_orig = eval_transform_b3(pil_img)

                    if use_tta:
                        t_b0_flip = torch.flip(t_b0_orig, dims=[-1])
                        t_b3_flip = torch.flip(t_b3_orig, dims=[-1])
                        b0_batch = torch.stack([t_b0_orig, t_b0_flip]).to(device)
                        b3_batch = torch.stack([t_b3_orig, t_b3_flip]).to(device)
                    else:
                        b0_batch = t_b0_orig.unsqueeze(0).to(device)
                        b3_batch = t_b3_orig.unsqueeze(0).to(device)

                    with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                        probs = model(b0_batch, b3_batch).mean(dim=0)

                else:
                    tensor_orig = eval_transform(pil_img)
                    if use_tta:
                        tensor_flip = torch.flip(tensor_orig, dims=[-1])
                        batch = torch.stack([tensor_orig, tensor_flip]).to(device)
                    else:
                        batch = tensor_orig.unsqueeze(0).to(device)

                    with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                        out = model(batch)
                    probs = torch.softmax(out, dim=1).mean(dim=0)

            fake_idx = class_to_idx.get("fake", 0)
            real_idx = class_to_idx.get("real", 1)

            fake_probability = float(probs[fake_idx].item() * 100.0)
            real_probability = float(probs[real_idx].item() * 100.0)

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
    if fake_probability >= 50.0:
        label = "Fake"
        confidence = fake_probability
    else:
        label = "Real"
        confidence = real_probability

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