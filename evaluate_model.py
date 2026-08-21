import os
import torch

from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification



MODEL_PATH = "model"

TEST_FOLDER = "DeepFake-Detect/split_dataset/test"



device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)



processor = AutoImageProcessor.from_pretrained(
    MODEL_PATH
)


model = AutoModelForImageClassification.from_pretrained(
    MODEL_PATH
)


model.to(device)

model.eval()



labels = model.config.id2label


correct = 0
total = 0



real_correct = 0
real_total = 0


fake_correct = 0
fake_total = 0





for folder in ["real", "fake"]:

    folder_path = os.path.join(
        TEST_FOLDER,
        folder
    )


    if not os.path.exists(folder_path):

        print(
            "Missing:",
            folder_path
        )

        continue



    for file in os.listdir(folder_path):


        image_path = os.path.join(
            folder_path,
            file
        )


        try:


            image = Image.open(
                image_path
            ).convert("RGB")



            inputs = processor(
                images=image,
                return_tensors="pt"
            )



            inputs = {
                k:v.to(device)
                for k,v in inputs.items()
            }




            with torch.no_grad():

                output = model(
                    **inputs
                )



            prediction = torch.argmax(
                output.logits,
                dim=1
            ).item()



            predicted_label = labels[prediction]



            total += 1



            if predicted_label.lower() == folder.lower():

                correct += 1



                if folder == "real":
                    real_correct += 1

                else:
                    fake_correct += 1



            if folder == "real":
                real_total += 1

            else:
                fake_total += 1



            print(
                file,
                "=>",
                predicted_label
            )



        except Exception as e:

            print(
                "Error:",
                file,
                e
            )




print("\n===================")

print(
    "Total Accuracy:",
    round(
        correct/total*100,
        2
    ),
    "%"
)



print(
    "Real Accuracy:",
    round(
        real_correct/real_total*100,
        2
    ) if real_total else 0,
    "%"
)



print(
    "Fake Accuracy:",
    round(
        fake_correct/fake_total*100,
        2
    ) if fake_total else 0,
    "%"
)


print("===================")