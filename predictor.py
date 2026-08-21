# predictor.py
# EfficientNet Deepfake Predictor

import os
import tensorflow as tf
import numpy as np

from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input


# ==============================
# MODEL PATH
# ==============================

MODEL_PATH = "deepfake_efficientnet_v3.keras"


# ==============================
# LOAD MODEL
# ==============================

print("Loading EfficientNet model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("EfficientNet Model Loaded Successfully")


# ==============================
# IMAGE SIZE
# ==============================

IMG_SIZE = (300, 300)


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