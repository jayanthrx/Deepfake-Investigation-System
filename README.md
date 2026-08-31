# 🛡️ Deepfake Investigation System

> **Advanced AI-Powered Digital Forensics & Media Authenticity Verification Platform**  
> *Built with EfficientNet Deep CNNs, Explainable AI (Grad-CAM), OpenCV, Flask, and ReportLab.*

---

## 📌 Overview

The **Deepfake Investigation System** is an enterprise-grade digital forensics platform engineered to detect synthetic facial manipulation, generative AI face swaps, and deepfake artifacts in both **images** and **videos**. 

By combining multi-scale facial landmark auto-cropping, deep convolutional feature extraction (**EfficientNet-B3**), and **Explainable AI (Grad-CAM)**, the system provides transparent authenticity confidence percentages, risk assessments, and cryptographically structured forensic PDF investigation reports.

---

## ✨ Key Features

- 👁️ **Face Auto-Extraction Pipeline:** Automatically isolates and centers facial regions with proportional padding before inference to eliminate background noise.
- 🧠 **EfficientNet-B3 Deep Learning:** Binary classification neural network predicting pristine authentic media vs. deepfake manipulations.
- 🔥 **Explainable AI (Grad-CAM):** Generates transparent activation heatmaps showing exact pixel regions that triggered the deepfake classification.
- 🎞️ **Frame-by-Frame Video Forensics:** Samples sequential video frames, computes temporal anomaly metrics, and aggregates confidence ratios.
- 📄 **Forensic PDF Report Generator:** Compiles case metadata, media snapshots, Grad-CAM attention maps, and breakdown metrics into a printable evidence document.
- 📊 **Real-time Analytics Dashboard:** Interactive Chart.js donut and bar visualizations tracking historical detection distributions.
- 📥 **1-Click CSV Data Export:** Instantly download all historical case registries in standard RFC 4180 CSV format.
- ⚡ **Headless REST API (`/api/predict`):** Seamless programmatic JSON integration for external microservices and mobile applications.

---

## 🏗️ System Architecture

```
[ Uploaded Media (Image / Video) ]
               │
               ▼
[ 1. Input Validation & MIME Guard ]
               │
               ▼
[ 2. OpenCV Facial Landmark Extraction & Auto-Crop ]
               │
               ▼
[ 3. EfficientNet-B3 Deep Feature Inference ]
       │                               │
       ▼                               ▼
[ 4. Grad-CAM XAI Heatmap ]   [ 5. Risk & Probability Calculation ]
       │                               │
       └───────────────┬───────────────┘
                       ▼
[ 6. SQLite Case Logging & ReportLab PDF Assembly ]
                       │
                       ▼
[ 7. Interactive Forensic Web Dashboard / JSON REST API ]
```

---

## 🚀 Quickstart & Installation

### Option 1: 1-Click Launch (Windows)
Double-click the included `run.bat` file in the root directory. It automatically verifies Python dependencies, launches the server, and opens your browser.

```cmd
run.bat
```

---

### Option 2: Manual Terminal Setup

1. **Clone or navigate to the repository:**
   ```bash
   cd Deepfake-investigation-system
   ```

2. **Create and activate a virtual environment (optional but recommended):**
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On Linux/macOS:
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Launch the application:**
   ```bash
   python app.py
   ```

5. **Open the web dashboard in your browser:**
   ```
   http://127.0.0.1:5000
   ```

---

## 🌐 Web Interface Endpoints

| Route | Method | Description |
|---|---|---|
| `/` | `GET` | Main Forensic Ingestion Studio with Drag & Drop |
| `/predict` | `POST` | Processes uploaded image/video and displays verdict |
| `/dashboard` | `GET` | Interactive KPI counters and Chart.js visualizations |
| `/history` | `GET` | Searchable historical case registry |
| `/export_csv` | `GET` | 1-Click download of case history as CSV |
| `/architecture` | `GET` | Interactive pipeline flow and architecture diagrams |
| `/results` | `GET` | Empirical model benchmarks and confusion matrix |
| `/about` | `GET` | Mission overview and technology specifications |
| `/health` | `GET` | System heartbeat health check |

---

## ⚡ REST API Reference

### POST `/api/predict`
Accepts multipart form-data file uploads and returns structured JSON analysis.

#### Example Request (`cURL`):
```bash
curl -X POST http://127.0.0.1:5000/api/predict \
     -F "file=@suspect_image.jpg"
```

#### Example Response (Image):
```json
{
  "success": true,
  "media_type": "Image",
  "filename": "suspect_image.jpg",
  "prediction": "Fake",
  "confidence": 89.64,
  "fake_probability": 89.64,
  "real_probability": 10.36,
  "risk": "Very High",
  "image_url": "/uploads/images/a1b2c3d4.jpg",
  "heatmap_url": "/uploads/images/heatmap_a1b2c3d4.jpg",
  "report_download_url": "/download/deepfake_report_9f8e7d6c.pdf"
}
```

#### Example Response (Video):
```json
{
  "success": true,
  "media_type": "Video",
  "filename": "suspect_video.mp4",
  "prediction": "Fake",
  "confidence": 74.49,
  "risk": "High",
  "fake_frames": 26,
  "real_frames": 9,
  "uncertain_frames": 0,
  "fake_percentage": 74.29,
  "real_percentage": 25.71,
  "uncertain_percentage": 0.0,
  "video_url": "/uploads/videos/video_uuid.mp4",
  "report_download_url": "/download/deepfake_report_video_uuid.pdf"
}
```

---

## 📊 Empirical Model Evaluation

Evaluated across standardized deepfake forensic datasets (e.g., FaceForensics++, Celeb-DF):

| Target Metric | Benchmark Score |
|---|---|
| **Overall Accuracy** | **69.86%** |
| **Synthetic Deepfake Recall** | **74.49%** |
| **Authentic Precision** | **65.77%** |
| **Macro F1-Score** | **0.684** |
| **Validation Loss** | **0.584** |

---

## 📁 Repository Structure

```
Deepfake-investigation-system/
├── app.py                      # Core Flask web server & REST API controller
├── predictor.py                # EfficientNet-B3 inference & face auto-crop pipeline
├── gradcam.py                  # Explainable AI (Grad-CAM) activation generator
├── video_detector.py           # Frame sampling & temporal sequence analyzer
├── pdf_report.py               # ReportLab forensic evidence PDF builder
├── database.py                 # SQLAlchemy SQLite data model
├── run.bat                     # 1-Click Windows execution launcher
├── requirements.txt            # Dependency specification
├── deepfake_efficientnet_v3.keras # Trained deep learning neural network
├── static/                     # Assets (architecture diagram, charts, favicon)
│   ├── architecture.png
│   ├── chart.png
│   └── favicon.ico
├── templates/                  # Modern Glassmorphic Dark UI
│   ├── index.html              # Drag-and-drop ingestion studio
│   ├── result.html             # Verdict & Grad-CAM heatmap view
│   ├── dashboard.html          # Interactive charts & CSV export
│   ├── history.html            # Case registry logs
│   ├── architecture.html       # Pipeline architecture view
│   ├── results.html            # Model evaluation metrics
│   └── about.html              # System specs & documentation
└── instance/
    └── history.db              # SQLite case registry database
```

---

## ⚖️ License & Ethical Notice
This project is developed strictly for digital forensic investigation, academic research, and media verification. Synthetic manipulation detection models must be used responsibly within lawful evidentiary standards.
