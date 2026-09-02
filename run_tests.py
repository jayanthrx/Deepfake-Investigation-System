"""
run_tests.py
Automated End-to-End Test Suite for Deepfake Investigation System
"""

import os
import sys
import unittest
import json
import urllib.request

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="backslashreplace")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


class TestDeepfakeInvestigationSystem(unittest.TestCase):

    def setUp(self):
        self.server_url = "http://127.0.0.1:5000"
        self.sample_img = os.path.join(BASE_DIR, "test.jpg")
        if not os.path.exists(self.sample_img):
            import numpy as np
            import cv2
            blank = np.zeros((300, 300, 3), dtype=np.uint8)
            cv2.imwrite(self.sample_img, blank)

    def test_01_model_loading(self):
        """Test EfficientNet deep learning model loading"""
        from predictor import model, IMG_SIZE
        self.assertIsNotNone(model, "EfficientNet model should be loaded")
        self.assertTrue(IMG_SIZE in [224, (224, 224), 300, (300, 300)], f"Target input size {IMG_SIZE} should be valid")
        print("  [PASS] 01 - Deep Learning model loading & input dimension verification")

    def test_02_image_prediction(self):
        """Test image prediction pipeline"""
        from predictor import predict_image
        prediction, confidence, risk, fake_prob, real_prob = predict_image(self.sample_img)
        self.assertIn(prediction, ["Real", "Fake", "Uncertain"])
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 100.0)
        self.assertIn(risk, ["Low", "Medium", "High", "Very High"])
        self.assertAlmostEqual(fake_prob + real_prob, 100.0, places=1)
        print(f"  [PASS] 02 - Image Prediction: {prediction} ({confidence:.2f}% confidence, Risk: {risk})")

    def test_03_gradcam_generation(self):
        """Test Grad-CAM explainable AI heatmap generation"""
        from gradcam import generate_heatmap
        from predictor import model
        out_heatmap = os.path.join(BASE_DIR, "reports", "test_heatmap.jpg")
        res = generate_heatmap(model=model, img_path=self.sample_img, output_path=out_heatmap)
        self.assertTrue(os.path.exists(out_heatmap), "Heatmap file should be created")
        if os.path.exists(out_heatmap):
            os.remove(out_heatmap)
        print("  [PASS] 03 - Grad-CAM XAI Heatmap Synthesis")

    def test_04_pdf_report_generation(self):
        """Test ReportLab forensic PDF generation"""
        from pdf_report import generate_report
        out_pdf = os.path.join(BASE_DIR, "reports", "test_report.pdf")
        generate_report(
            filename="test_file.jpg",
            prediction="Fake",
            confidence=85.5,
            risk="High",
            media_type="Image",
            fake_probability=85.5,
            real_probability=14.5,
            output_path=out_pdf
        )
        self.assertTrue(os.path.exists(out_pdf), "PDF report should be created")
        self.assertGreater(os.path.getsize(out_pdf), 1000, "PDF should not be empty")
        if os.path.exists(out_pdf):
            os.remove(out_pdf)
        print("  [PASS] 04 - Forensic PDF Report Assembly")

    def test_05_database_connectivity(self):
        """Test SQLite case history database"""
        from app import app, db, Investigation
        with app.app_context():
            count = Investigation.query.count()
            self.assertGreaterEqual(count, 0)
            print(f"  [PASS] 05 - SQLite Registry: {count} historical cases verified")

    def test_06_web_endpoints(self):
        """Test HTTP routes availability"""
        endpoints = ["/", "/dashboard", "/history", "/architecture", "/results", "/about", "/health"]
        for ep in endpoints:
            url = self.server_url + ep
            try:
                resp = urllib.request.urlopen(url, timeout=5)
                self.assertEqual(resp.status, 200, f"Endpoint {ep} should return 200")
            except Exception as e:
                self.fail(f"Failed to connect to {ep}: {e}")
        print("  [PASS] 06 - All Web UI Endpoints (HTTP 200 OK)")

    def test_07_rest_api_endpoint(self):
        """Test /api/predict JSON endpoint"""
        url = self.server_url + "/api/predict"
        boundary = "----TestApiBoundaryRunner"
        with open(self.sample_img, "rb") as f:
            file_bytes = f.read()

        body = bytearray()
        body.extend(f"--{boundary}\r\n".encode("utf-8"))
        body.extend(b'Content-Disposition: form-data; name="file"; filename="api_test.jpg"\r\n')
        body.extend(b"Content-Type: image/jpeg\r\n\r\n")
        body.extend(file_bytes)
        body.extend(b"\r\n")
        body.extend(f"--{boundary}--\r\n".encode("utf-8"))

        req = urllib.request.Request(url, data=bytes(body), method="POST")
        req.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                self.assertEqual(resp.status, 200)
                data = json.loads(resp.read().decode("utf-8"))
                self.assertTrue(data.get("success"))
                self.assertIn("prediction", data)
                self.assertIn("confidence", data)
                self.assertIn("report_download_url", data)
                print("  [PASS] 07 - Headless REST API (/api/predict) JSON contract verified")
        except Exception as e:
            self.fail(f"API request failed: {e}")

    def test_08_csv_export(self):
        """Test /export_csv route output"""
        url = self.server_url + "/export_csv"
        try:
            resp = urllib.request.urlopen(url, timeout=5)
            self.assertEqual(resp.status, 200)
            csv_content = resp.read().decode("utf-8")
            self.assertIn("ID,Filename,Prediction,Confidence (%),Risk Level,Date Analyzed", csv_content)
            print("  [PASS] 08 - CSV Case History Export verified")
        except Exception as e:
            self.fail(f"CSV export failed: {e}")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("  RUNNING DEEPFAKE INVESTIGATION SYSTEM AUTOMATED TEST SUITE")
    print("=" * 70 + "\n")
    unittest.main(verbosity=0)
