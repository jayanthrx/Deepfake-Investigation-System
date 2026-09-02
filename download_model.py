"""
Model downloader script for cloud deployment.
Downloads deepfake_efficientnet_v3.keras from Hugging Face Hub at startup.
Run this before starting the app if the model is not present locally.
"""

import os
import sys

MODEL_FILES = [
    os.environ.get("HF_FILENAME", "deepfake_detector_best.pt"),
    "deepfake_efficientnet_v3.keras"
]

HF_REPO_ID = os.environ.get("HF_REPO_ID", "jayanthrx/deepfake-efficientnet")


def download_model():
    # Check if at least one working model exists locally
    local_found = [f for f in ["deepfake_detector_best.pt", "deepfake_efficientnet_v3.keras", "deepfake_efficientnet_v2.keras", "deepfake_efficientnet.keras"] if os.path.exists(f)]
    if local_found:
        print(f"[Model] Found local model(s): {', '.join(local_found)}")

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("[Model] Installing huggingface_hub...")
        os.system(f"{sys.executable} -m pip install huggingface_hub -q")
        from huggingface_hub import hf_hub_download

    downloaded_any = False
    for filename in MODEL_FILES:
        if os.path.exists(filename):
            print(f"[Model] {filename} already exists locally.")
            downloaded_any = True
            continue
        try:
            print(f"[Model] Attempting to download {filename} from Hugging Face repo: {HF_REPO_ID}...")
            path = hf_hub_download(
                repo_id=HF_REPO_ID,
                filename=filename,
                local_dir=".",
            )
            print(f"[Model] Successfully downloaded {filename} to {path}")
            downloaded_any = True
        except Exception as e:
            print(f"[Model] Notice: Could not download {filename} ({e})")

    if not downloaded_any and not local_found:
        print("[Model] Warning: No models found or downloaded. Please check HF_REPO_ID.")
        return False
    return True


if __name__ == "__main__":
    success = download_model()
    sys.exit(0 if success else 1)
