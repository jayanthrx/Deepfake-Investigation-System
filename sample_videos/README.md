# 🎬 Benchmark Sample Videos Library

This directory contains curated, verified benchmark MP4 videos for testing and evaluating the Deepfake Investigation System.

## 📋 Manifest

| Filename | Category | Title | Size | Description |
|---|---|---|---|---|
| `real_sample_1_authentic_speaker.mp4` | **Authentic Real** | Authentic Interview (Real Speaker) | 4.8 MB | Verified authentic human speaker with natural eye micro-movements, facial kinematics, and organic illumination. |
| `real_sample_2_conversational_face.mp4` | **Authentic Real** | Natural Conversational Portrait | 4.8 MB | Ground-truth authentic subject with natural spontaneous speech and zero digital neural manipulation. |
| `fake_sample_1_neural_faceswap.mp4` | **Synthetic Deepfake** | Neural Face-Swap Deepfake | 10.4 MB | High-resolution identity face-swap with boundary warp blending artifacts and subtle facial jitter. |
| `fake_sample_2_expression_manipulation.mp4` | **Synthetic Deepfake** | Expression Reenactment Deepfake | 4.3 MB | Target facial expression driven by source actor, causing synthetic mouth warping and skin texture smoothing. |
| `fake_sample_3_deepfake_reenactment.mp4` | **Synthetic Deepfake** | DFDC Forensic Deepfake Benchmark | 2.7 MB | Standardized Deepfake Detection Challenge manipulated clip with synthetic neural facial replacement. |
| `fake_sample_4_ai_synthesis.mp4` | **Synthetic Deepfake** | Generative AI Facial Synthesis | 2.1 MB | Fully AI-synthesized viral deepfake video with generative facial textures and altered temporal frequencies. |
| `fake_sample_5_lipsync_tampering.mp4` | **Synthetic Deepfake** | Neural Lip-Sync Tampering | 3.4 MB | Audio-driven synthetic lip sync deepfake demonstrating biometric mismatch between audio and lip articulators. |

## 🚀 Usage

### Web Interface
In the web application, visit the home page to view cards for each sample with:
- **⬇️ Download**: Download individual MP4 videos directly.
- **📦 Download All (.zip)**: Download the complete benchmark dataset as a compressed archive.
- **⚡ Run Detect**: Run instant deepfake detection on the sample video without manual re-upload.

### Command-Line Utility
```bash
# List all sample videos
python download_samples.py --list

# Download / export samples to a folder
python download_samples.py --category all --outdir ./exported_videos

# Package as a zip archive
python download_samples.py --category fake --zip fake_dataset.zip

# Run detection directly
python download_samples.py --detect real_sample_1_authentic_speaker.mp4
python download_samples.py --detect fake_sample_1_neural_faceswap.mp4
```
