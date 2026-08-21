import os
import tensorflow as tf
import numpy as np

from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input


MODEL_PATH = "deepfake_efficientnet_v2.keras"

TEST_DIR = "DeepFake-Detect/split_dataset/test"


IMG_SIZE = (224,224)


print("Loading EfficientNet...")

model = tf.keras.models.load_model(
    MODEL_PATH
)

print("Model Loaded")


correct = 0
total = 0

real_correct = 0
real_total = 0

fake_correct = 0
fake_total = 0



for folder in ["real","fake"]:

    folder_path = os.path.join(
        TEST_DIR,
        folder
    )


    for file in os.listdir(folder_path):

        path = os.path.join(
            folder_path,
            file
        )


        try:

            img = image.load_img(
                path,
                target_size=IMG_SIZE
            )


            img = image.img_to_array(img)

            img = np.expand_dims(
                img,
                axis=0
            )

            img = preprocess_input(img)


            prediction = model.predict(
                img,
                verbose=0
            )[0][0]


            # IMPORTANT
            # 0 = Fake
            # 1 = Real


            if prediction >= 0.5:

                predicted="real"

            else:

                predicted="fake"



            print(
                file,
                "=>",
                predicted
            )


            total +=1


            if predicted == folder:

                correct +=1


                if folder=="real":
                    real_correct+=1

                else:
                    fake_correct+=1



            if folder=="real":
                real_total+=1

            else:
                fake_total+=1



        except Exception as e:

            print(
                "Error",
                file,
                e
            )




print("===================")

print(
"Total Accuracy:",
round(correct/total*100,2),
"%"
)


print(
"Real Accuracy:",
round(real_correct/real_total*100,2),
"%"
)


print(
"Fake Accuracy:",
round(fake_correct/fake_total*100,2),
"%"
)


print("===================")