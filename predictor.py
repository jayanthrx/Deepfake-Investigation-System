# predictor.py
# EfficientNet Deepfake Predictor

import os
import sys
import tensorflow as tf
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="backslashreplace")

from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input


# ==============================
# MODEL PATH
# ==============================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_CANDIDATES = [
    os.path.join(BASE_DIR, "deepfake_efficientnet_v3.keras"),
    os.path.join(BASE_DIR, "deepfake_efficientnet_v2.keras"),
    os.path.join(BASE_DIR, "deepfake_efficientnet.keras"),
    "deepfake_efficientnet_v3.keras"
]

MODEL_PATH = next((p for p in MODEL_CANDIDATES if os.path.exists(p)), "deepfake_efficientnet_v3.keras")


# ==============================
# LOAD MODEL
# ==============================

print("Loading EfficientNet model from:", MODEL_PATH)

model = tf.keras.models.load_model(MODEL_PATH)

print("EfficientNet Model Loaded Successfully")


# ==============================
# IMAGE SIZE
# ==============================

IMG_SIZE = (300, 300)


import cv2

face_cascade = None
try:
    if hasattr(cv2, "CascadeClassifier") and hasattr(cv2, "data"):
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
except Exception as fe:
    print("Face cascade init note:", fe)

def extract_face(img_cv, padding_ratio=0.2):
    """
    Extract the most prominent face from an image with padding.
    If no face cascade is available or no face is detected, returns original image.
    """
    if face_cascade is None:
        return img_cv
    try:
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(60, 60)
        )
        if len(faces) == 0:
            return img_cv

        # Select the largest face by area
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
        if cropped.size > 0:
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

def predict_image(image_path):

    if not os.path.exists(image_path):

        return (
            "File Not Found",
            0.0,
            "Low",
            0.0,
            0.0
        )

    try:

        cv_img = cv2.imread(image_path)
        if cv_img is not None:
            cropped_face = extract_face(cv_img)
            rgb_face = cv2.cvtColor(cropped_face, cv2.COLOR_BGR2RGB)
            resized_face = cv2.resize(rgb_face, IMG_SIZE)
            img_array = np.expand_dims(resized_face.astype(np.float32), axis=0)
            img_array = preprocess_input(img_array)
        else:
            img = image.load_img(
                image_path,
                target_size=IMG_SIZE
            )
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(
                img_array,
                axis=0
            )
            img_array = preprocess_input(
                img_array
            )

    except Exception as e:

        print("Image Loading Error:", e)

        return (
            "Invalid Image",
            0.0,
            "Low",
            0.0,
            0.0
        )

    # ==============================
    # MODEL PREDICTION
    # ==============================

    prediction = model.predict(
        img_array,
        verbose=0
    )

    score = float(prediction[0][0])

    # Model output:
    # 0 = Fake
    # 1 = Real

    real_probability = score * 100
    fake_probability = (1 - score) * 100

    # ==============================
    # DECISION LOGIC
    # ==============================

    if fake_probability >= 55:

        label = "Fake"
        confidence = fake_probability

    elif real_probability >= 55:

        label = "Real"
        confidence = real_probability

    else:

        label = "Uncertain"
        confidence = max(
            fake_probability,
            real_probability
        )

    confidence = round(confidence, 2)
    fake_probability = round(fake_probability, 2)
    real_probability = round(real_probability, 2)

    risk = calculate_risk(
        label,
        confidence
    )

    # ==============================
    # TERMINAL OUTPUT
    # ==============================

    print("======================")
    print("File:", image_path)
    print("Prediction:", label)
    print("Confidence:", confidence, "%")
    print("Fake Probability:", fake_probability, "%")
    print("Real Probability:", real_probability, "%")
    print("Risk:", risk)
    print("======================")

    # ==============================
    # RETURN VALUES
    # ==============================

    return (
        label,
        confidence,
        risk,
        fake_probability,
        real_probability
    )