import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image

MODEL_PATH = "./model"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Running on:", device)

processor = AutoImageProcessor.from_pretrained(MODEL_PATH)

model = AutoModelForImageClassification.from_pretrained(
    MODEL_PATH
)

print("ID2LABEL:", model.config.id2label)
print("LABEL2ID:", model.config.label2id)

model.to(device)
model.eval()

# Put your test image here
image = Image.open("test.jpg").convert("RGB")

inputs = processor(
    images=image,
    return_tensors="pt"
)

inputs = {k: v.to(device) for k, v in inputs.items()}

with torch.no_grad():
    outputs = model(**inputs)

probabilities = torch.nn.functional.softmax(
    outputs.logits,
    dim=1
)

prediction = probabilities.argmax(dim=1).item()

print("\nResults:")
print("Class:", model.config.id2label[prediction])
print("Confidence:", float(probabilities[0][prediction]))