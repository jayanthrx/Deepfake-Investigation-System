"""
Model downloader script for cloud deployment.
Downloads deepfake_efficientnet_v3.keras from Hugging Face Hub at startup.
Run this before starting the app if the model is not present locally.
"""

import os
import sys

MODEL_PATH = "deepfake_efficientnet_v3.keras"

# ── Set this to your Hugging Face repo after uploading ──────────────────────
# Format: "your-hf-username/your-repo-name"
HF_REPO_ID = os.environ.get("HF_REPO_ID", "jayanthrx/deepfake-efficientnet")
HF_FILENAME = "deepfake_efficientnet_v3.keras"
# ────────────────────────────────────────────────────────────────────────────


def download_model():
    if os.path.exists(MODEL_PATH):
        print(f"[Model] Already present at: {MODEL_PATH}")
        return True

    print(f"[Model] Not found locally. Downloading from Hugging Face: {HF_REPO_ID}...")
    try:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=HF_FILENAME,
            local_dir=".",
        )
        print(f"[Model] Downloaded successfully to: {path}")
        return True
    except ImportError:
        print("[Model] huggingface_hub not installed. Installing...")
        os.system(f"{sys.executable} -m pip install huggingface_hub -q")
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=HF_FILENAME,
            local_dir=".",
        )
        print(f"[Model] Downloaded successfully to: {path}")
        return True
    except Exception as e:
        print(f"[Model] ERROR: Could not download model — {e}")
        print(f"[Model] Set HF_REPO_ID environment variable to your Hugging Face repo.")
        return False


if __name__ == "__main__":
    success = download_model()
    sys.exit(0 if success else 1)
