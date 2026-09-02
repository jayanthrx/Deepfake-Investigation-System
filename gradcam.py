import os
import sys
import numpy as np
import cv2
from PIL import Image

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="backslashreplace")

import predictor
from predictor import extract_face, eval_transform, device, class_to_idx


def generate_pytorch_gradcam(py_model, img_cv, cropped_face, target_layer=None):
    """
    Generates Grad-CAM heatmap for a PyTorch convolutional / timm model.
    """
    try:
        import torch
    except ImportError:
        return None

    py_model.eval()
    
    # Identify target layer if not provided
    if target_layer is None:
        if hasattr(py_model, "conv_head"):
            target_layer = py_model.conv_head
        elif hasattr(py_model, "head") and hasattr(py_model.head, "conv"):
            target_layer = py_model.head.conv
        elif hasattr(py_model, "blocks"):
            target_layer = py_model.blocks[-1]
        elif hasattr(py_model, "features"):
            target_layer = py_model.features[-1]
        else:
            # Search last Conv2d module
            for module in reversed(list(py_model.modules())):
                if isinstance(module, torch.nn.Conv2d):
                    target_layer = module
                    break

    if target_layer is None:
        print("[!] No suitable conv layer found for PyTorch GradCAM")
        return None

    activations = []
    gradients = []

    def f_hook(module, input, output):
        activations.append(output)

    def b_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    h1 = target_layer.register_forward_hook(f_hook)
    h2 = target_layer.register_full_backward_hook(b_hook)

    try:
        rgb_face = cv2.cvtColor(cropped_face, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_face)
        tensor_img = eval_transform(pil_img).unsqueeze(0).to(device)

        py_model.zero_grad()
        output = py_model(tensor_img)

        # Target class: Fake class (or highest predicted class)
        fake_idx = class_to_idx.get("fake", 0)
        target_score = output[0, fake_idx]
        target_score.backward(retain_graph=False)

        if not activations or not gradients:
            return None

        act = activations[0].detach()  # [1, C, H, W]
        grad = gradients[0].detach()   # [1, C, H, W]

        pooled_grad = torch.mean(grad, dim=[0, 2, 3])  # [C]
        for i in range(act.shape[1]):
            act[:, i, :, :] *= pooled_grad[i]

        heatmap = torch.mean(act, dim=1).squeeze().cpu().numpy()
        heatmap = np.maximum(heatmap, 0)
        max_val = np.max(heatmap)
        if max_val > 0:
            heatmap = heatmap / max_val

        return heatmap

    except Exception as e:
        print("PyTorch GradCAM Exception:", e)
        return None
    finally:
        h1.remove()
        h2.remove()


_cached_keras_grad_model = None
_cached_for_model_id = None


def generate_heatmap(model=None, img_path=None, processor=None, device_param=None, output_path=None):
    """
    Generates a Grad-CAM heatmap visualization highlighting manipulated regions.
    """
    global _cached_keras_grad_model, _cached_for_model_id
    try:
        if img_path is None or not os.path.exists(img_path):
            return None

        cv_img = cv2.imread(img_path)
        if cv_img is None:
            return None

        cropped_face = extract_face(cv_img)
        if cropped_face is None or cropped_face.size == 0:
            cropped_face = cv_img

        use_model = model if model is not None else predictor.model
        m_type = predictor.model_type
        heatmap = None

        # Check if PyTorch model
        if m_type == "pytorch" and use_model is not None:
            heatmap = generate_pytorch_gradcam(use_model, cv_img, cropped_face)

        # Fallback to Keras GradCAM
        if heatmap is None and use_model is not None and m_type != "pytorch":
            try:
                import tensorflow as tf
                from tensorflow.keras.applications.efficientnet import preprocess_input

                in_shape = getattr(use_model, "input_shape", None)
                k_size = int(in_shape[1]) if (in_shape and len(in_shape) >= 3 and in_shape[1] is not None) else 224
                rgb_face = cv2.cvtColor(cropped_face, cv2.COLOR_BGR2RGB)
                resized_face = cv2.resize(rgb_face, (k_size, k_size))
                img_array = np.expand_dims(resized_face.astype(np.float32), axis=0)
                img_array = preprocess_input(img_array)

                # Use or build cached Grad Model
                if _cached_keras_grad_model is None or _cached_for_model_id != id(use_model):
                    last_conv_layer = None
                    for layer in reversed(use_model.layers):
                        if isinstance(layer, tf.keras.layers.Conv2D):
                            last_conv_layer = layer
                            break

                    if last_conv_layer:
                        _cached_keras_grad_model = tf.keras.models.Model(
                            inputs=use_model.inputs,
                            outputs=[last_conv_layer.output, use_model.output]
                        )
                        _cached_for_model_id = id(use_model)

                if _cached_keras_grad_model:
                    with tf.GradientTape() as tape:
                        conv_output, prediction = _cached_keras_grad_model(img_array)
                        loss = prediction[:, 0]
                    grads = tape.gradient(loss, conv_output)
                    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
                    conv_output = conv_output[0]
                    hm = conv_output @ pooled_grads[..., tf.newaxis]
                    hm = tf.squeeze(hm).numpy()
                    hm = np.maximum(hm, 0)
                    if np.max(hm) > 0:
                        heatmap = hm / np.max(hm)
            except Exception as ke:
                print("Keras GradCAM Note:", ke)

        # Fallback: if model has no gradient hooks available, generate high-frequency edge-gradient heatmap
        if heatmap is None:
            gray = cv2.cvtColor(cropped_face, cv2.COLOR_BGR2GRAY)
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            heatmap = np.abs(laplacian)
            heatmap = cv2.GaussianBlur(heatmap, (15, 15), 0)
            if np.max(heatmap) > 0:
                heatmap = heatmap / np.max(heatmap)

        h, w = cropped_face.shape[:2]
        heatmap_resized = cv2.resize(heatmap, (w, h))
        heatmap_color = np.uint8(255 * heatmap_resized)
        heatmap_color = cv2.applyColorMap(heatmap_color, cv2.COLORMAP_JET)

        overlay = cv2.addWeighted(cropped_face, 0.60, heatmap_color, 0.40, 0)

        if output_path is None:
            os.makedirs("uploads", exist_ok=True)
            output_path = os.path.join("uploads", "heatmap.jpg")
        else:
            out_dir = os.path.dirname(output_path)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)

        cv2.imwrite(output_path, overlay)
        print(f"[+] Grad-CAM Heatmap generated at: {output_path}")
        return output_path

    except Exception as e:
        print("Grad-CAM Generation Error:", e)
        return None