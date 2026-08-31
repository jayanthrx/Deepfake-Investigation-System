# gradcam.py
# EfficientNet v3 Grad-CAM Heatmap Generator

import os
import sys
import tensorflow as tf
import numpy as np
import cv2

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="backslashreplace")

from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input


IMG_SIZE = 300


from predictor import extract_face

# ==============================
# GENERATE HEATMAP
# ==============================

def generate_heatmap(model, img_path, processor=None, device=None, output_path=None):

    try:

        # ==============================
        # LOAD IMAGE & EXTRACT FACE
        # ==============================

        cv_img = cv2.imread(img_path)
        if cv_img is not None:
            cropped_face = extract_face(cv_img)
            rgb_face = cv2.cvtColor(cropped_face, cv2.COLOR_BGR2RGB)
            resized_face = cv2.resize(rgb_face, (IMG_SIZE, IMG_SIZE))
            img_array = np.expand_dims(resized_face.astype(np.float32), axis=0)
            img_array = preprocess_input(img_array)
            base_overlay_img = cropped_face
        else:
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
            base_overlay_img = None



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

            try:
                conv_output, prediction = grad_model(img_array)
            except Exception:
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

        if base_overlay_img is not None:
            original = base_overlay_img
        else:
            original = cv2.imread(img_path)

        if original is None:
            return None

        original = cv2.resize(
            original,
            (IMG_SIZE, IMG_SIZE)
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

        if output_path is None:
            os.makedirs(
                "uploads",
                exist_ok=True
            )

            output_path = os.path.join(
                "uploads",
                "heatmap.jpg"
            )
        else:
            out_dir = os.path.dirname(output_path)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)


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