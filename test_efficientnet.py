import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from tensorflow.keras.applications.efficientnet import preprocess_input


MODEL_PATH = "./DeepFake-Detect/tmp_checkpoint/best_model.keras"

model = tf.keras.models.load_model(MODEL_PATH)

img_path = "test.jpg"

img = image.load_img(
    img_path,
    target_size=(224,224)
)

img_array = image.img_to_array(img)

img_array = np.expand_dims(img_array, axis=0)

img_array = preprocess_input(img_array)


prediction = model.predict(img_array)[0][0]


print("Prediction score:", prediction)


if prediction >= 0.5:
    print("Class: Fake")
else:
    print("Class: Real")