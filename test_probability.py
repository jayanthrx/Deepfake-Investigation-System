import os
from predictor import predict_image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

image_path = os.path.join(BASE_DIR, "uploads", "real_00010.jpg")
if not os.path.exists(image_path):
    image_path = os.path.join(BASE_DIR, "test.jpg")

print("Testing Image:", image_path)

label, confidence, risk, fake_prob, real_prob = predict_image(image_path)

print("\n--- Summary ---")
print("Prediction        :", label)
print("Confidence        :", confidence, "%")
print("Real Probability  :", real_prob, "%")
print("Fake Probability  :", fake_prob, "%")
print("Risk Level        :", risk)