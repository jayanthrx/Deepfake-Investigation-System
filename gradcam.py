# gradcam.py
# EfficientNet v3 Grad-CAM Heatmap Generator

import os
import tensorflow as tf
import numpy as np
import cv2

from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input


IMG_SIZE = 300


# ==============================
# GENERATE HEATMAP
# ==============================

def generate_heatmap(model, img_path, processor=None, device=None):

    try:

        # ==============================
        # LOAD IMAGE
        # ==============================

        img = image.load_img(
            img_path,
            target_size=(IMG_SIZE, IMG_SIZE)
        )


        img_array = image.img_to_array(img)


        img_array = np.expand_dims(
            img_array,
            axis=0
        )


        img_array = preprocess_input(
            img_array
        )



        # ==============================
        # FIND LAST CONVOLUTION LAYER
        # ==============================

        last_conv_layer = None


        for layer in reversed(model.layers):

            if isinstance(layer, tf.keras.layers.Conv2D):

                last_conv_layer = layer
                break


        if last_conv_layer is None:

            print("No convolution layer found")

            return None



        print(
            "Using layer:",
            last_conv_layer.name
        )



        # ==============================
        # GRAD MODEL
        # ==============================

        grad_model = tf.keras.models.Model(

            inputs=model.inputs,

            outputs=[
                last_conv_layer.output,
                model.output
            ]

        )



        # ==============================
        # GRADIENT
        # ==============================

        with tf.GradientTape() as tape:


            conv_output, prediction = grad_model(
               {"input_layer": img_array}
            )


            loss = prediction[:,0]



        grads = tape.gradient(

            loss,

            conv_output

        )


        pooled_grads = tf.reduce_mean(

            grads,

            axis=(0,1,2)

        )


        conv_output = conv_output[0]



        heatmap = conv_output @ pooled_grads[..., tf.newaxis]


        heatmap = tf.squeeze(
            heatmap
        )


        heatmap = heatmap.numpy()



        # ==============================
        # NORMALIZE
        # ==============================

        heatmap = np.maximum(
            heatmap,
            0
        )


        heatmap = heatmap / (
            np.max(heatmap)+1e-8
        )



        heatmap = cv2.resize(

            heatmap,

            (IMG_SIZE,IMG_SIZE)

        )



        # ==============================
        # ORIGINAL IMAGE
        # ==============================


        original = cv2.imread(
            img_path
        )


        original = cv2.resize(

            original,

            (IMG_SIZE,IMG_SIZE)

        )



        heatmap = np.uint8(

            255 * heatmap

        )


        heatmap = cv2.applyColorMap(

            heatmap,

            cv2.COLORMAP_JET

        )



        result = cv2.addWeighted(

            original,

            0.6,

            heatmap,

            0.4,

            0

        )



        # ==============================
        # SAVE
        # ==============================


        os.makedirs(

            "uploads",

            exist_ok=True

        )


        output_path = os.path.join(

            "uploads",

            "heatmap.jpg"

        )


        cv2.imwrite(

            output_path,

            result

        )



        print(

            "Heatmap saved:",

            output_path

        )


        return output_path



    except Exception as e:


        print(

            "GradCAM Error:",

            e

        )


        return None