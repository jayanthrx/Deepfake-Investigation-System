import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input
import numpy as np


model = tf.keras.models.load_model("deepfake_efficientnet.keras")


def test(path):

    img = image.load_img(
        path,
        target_size=(224,224)
    )

    img = image.img_to_array(img)

    img = np.expand_dims(img,0)

    img = preprocess_input(img)

    prediction = model.predict(img)

    print(path)
    print("RAW OUTPUT:", prediction[0][0])


test("uploads/real_00010.jpg")

test("uploads/easy_1_1110.jpg")