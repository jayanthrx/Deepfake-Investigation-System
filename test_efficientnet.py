import os
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from tensorflow.keras.applications.efficientnet import preprocess_input

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_CANDIDATES = [
    os.path.join(BASE_DIR, "deepfake_efficientnet_v3.keras"),
    os.path.join(BASE_DIR, "deepfake_efficientnet_v2.keras"),
    os.path.join(BASE_DIR, "deepfake_efficientnet.keras"),
    os.path.join(BASE_DIR, "DeepFake-Detect/tmp_checkpoint/best_model.keras")
]

MODEL_PATH = next((p for p in MODEL_CANDIDATES if os.path.exists(p)), "deepfake_efficientnet_v3.keras")

print("Loading model from:", MODEL_PATH)
model = tf.keras.models.load_model(MODEL_PATH)

img_path = os.path.join(BASE_DIR, "test.jpg")
if not os.path.exists(img_path):
    img_path = "test.jpg"

# Determine input shape from model
try:
    input_shape = model.input_shape[1:3]
    if input_shape[0] is None:
        target_size = (300, 300)
    else:
        target_size = input_shape
except Exception:
    target_size = (300, 300)

print("Target size:", target_size)

img = image.load_img(
    img_path,
    target_size=target_size
)

img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = preprocess_input(img_array)

prediction = float(model.predict(img_array)[0][0])

real_prob = prediction * 100
fake_prob = (1 - prediction) * 100

print(f"Prediction raw score: {prediction:.4f}")
print(f"Fake Probability: {fake_prob:.2f}%")
print(f"Real Probability: {real_prob:.2f}%")

# Decision logic: 0 = Fake, 1 = Real
if prediction >= 0.5:
    print(f"Class: Real (Confidence: {real_prob:.2f}%)")
else:
    print(f"Class: Fake (Confidence: {fake_prob:.2f}%)")