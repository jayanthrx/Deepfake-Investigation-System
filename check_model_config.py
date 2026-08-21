from transformers import AutoModelForImageClassification


model = AutoModelForImageClassification.from_pretrained(
    "model"
)


print(model.config.id2label)

print(model.config.label2id)