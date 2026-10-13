"""
=============================================================================
DEEPFAKE INVESTIGATION SYSTEM — BENCHMARK SAMPLE VIDEOS DOWNLOAD & CLI RUNNER
=============================================================================

This CLI utility manages the benchmark sample video dataset:
  - List available Authentic (Real) and Synthetic (Fake/Deepfake) benchmark videos
  - Export/download individual videos or batches into any local destination directory
  - Package videos into a self-contained ZIP archive with benchmark documentation
  - Directly run the deepfake video forensic detector on any sample video

Usage:
  python download_samples.py --list
  python download_samples.py --category all --outdir ./exported_videos
  python download_samples.py --category real --outdir ./real_samples
  python download_samples.py --category fake --zip fake_benchmarks.zip
  python download_samples.py --detect real_sample_1_authentic_speaker.mp4
  python download_samples.py --detect fake_sample_1_neural_faceswap.mp4
=============================================================================
"""

import os
import sys
import shutil
import zipfile
import argparse

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAMPLE_DIR = os.path.join(BASE_DIR, "sample_videos")

CATALOG = [
    {
        "filename": "real_sample_1_authentic_speaker.mp4",
        "category": "Real",
        "title": "Authentic Interview (Real Speaker)",
        "duration": "10s",
        "size_mb": 5.0,
        "description": "Verified authentic human speaker with natural eye micro-movements, facial kinematics, and organic illumination."
    },
    {
        "filename": "real_sample_2_conversational_face.mp4",
        "category": "Real",
        "title": "Natural Conversational Portrait",
        "duration": "10s",
        "size_mb": 5.0,
        "description": "Ground-truth authentic subject with natural spontaneous speech and zero digital neural manipulation."
    },
    {
        "filename": "fake_sample_1_neural_faceswap.mp4",
        "category": "Fake",
        "title": "Neural Face-Swap Deepfake",
        "duration": "10s",
        "size_mb": 10.9,
        "description": "High-resolution identity face-swap with boundary warp blending artifacts and subtle facial jitter."
    },
    {
        "filename": "fake_sample_2_expression_manipulation.mp4",
        "category": "Fake",
        "title": "Expression Reenactment Deepfake",
        "duration": "10s",
        "size_mb": 4.5,
        "description": "Target facial expression driven by source actor, causing synthetic mouth warping and skin texture smoothing."
    },
    {
        "filename": "fake_sample_3_deepfake_reenactment.mp4",
        "category": "Fake",
        "title": "DFDC Forensic Deepfake Benchmark",
        "duration": "10s",
        "size_mb": 2.8,
        "description": "Standardized Deepfake Detection Challenge manipulated clip with synthetic neural facial replacement."
    },
    {
        "filename": "fake_sample_4_ai_synthesis.mp4",
        "category": "Fake",
        "title": "Generative AI Facial Synthesis",
        "duration": "8s",
        "size_mb": 2.2,
        "description": "Fully AI-synthesized viral deepfake video with generative facial textures and altered temporal frequencies."
    },
    {
        "filename": "fake_sample_5_lipsync_tampering.mp4",
        "category": "Fake",
        "title": "Neural Lip-Sync Tampering",
        "duration": "10s",
        "size_mb": 3.5,
        "description": "Audio-driven synthetic lip sync deepfake demonstrating biometric mismatch between audio and lip articulators."
    }
]


def list_samples():
    print("=" * 80)
    print("DEEPFAKE INVESTIGATION SYSTEM — BENCHMARK SAMPLES")
    print("=" * 80)
    print(f"{'#':<3} {'Category':<8} {'Size':<9} {'Filename':<42}")
    print("-" * 80)
    
    total_size = 0.0
    for idx, item in enumerate(CATALOG, 1):
        vpath = os.path.join(SAMPLE_DIR, item["filename"])
        exists = os.path.exists(vpath)
        actual_size = (os.path.getsize(vpath) / (1024 * 1024)) if exists else 0.0
        total_size += actual_size
        status = f"{actual_size:.1f} MB" if exists else "MISSING"
        cat_badge = "[REAL]" if item["category"] == "Real" else "[FAKE]"
        print(f"{idx:<3} {cat_badge:<8} {status:<9} {item['filename']:<42}")
        print(f"    Title: {item['title']}")
        print(f"    Info : {item['description']}")
        print()

    print("-" * 80)
    print(f"Total benchmark dataset size: {total_size:.2f} MB ({len(CATALOG)} curated videos)")
    print("=" * 80)


def export_samples(category="all", outdir=None, zip_path=None):
    category = category.lower()
    selected = []
    for item in CATALOG:
        cat_match = (
            category == "all" or
            (category == "real" and item["category"].lower() == "real") or
            (category == "fake" and item["category"].lower() == "fake")
        )
        if cat_match:
            vpath = os.path.join(SAMPLE_DIR, item["filename"])
            if os.path.exists(vpath):
                selected.append((item, vpath))
            else:
                print(f"[WARN] File not found: {vpath}")

    if not selected:
        print("[ERROR] No matching sample videos found to export.")
        return

    # Export to directory
    if outdir:
        os.makedirs(outdir, exist_ok=True)
        print(f"\n[+] Exporting {len(selected)} sample videos to: {os.path.abspath(outdir)}")
        for item, vpath in selected:
            dest = os.path.join(outdir, item["filename"])
            shutil.copyfile(vpath, dest)
            size_mb = os.path.getsize(dest) / (1024 * 1024)
            print(f"    [+] Copied: {item['filename']} ({size_mb:.2f} MB)")
        print("[+] Export complete!")

    # Export to ZIP
    if zip_path:
        zip_dir = os.path.dirname(os.path.abspath(zip_path))
        if zip_dir:
            os.makedirs(zip_dir, exist_ok=True)
        print(f"\n[+] Building benchmark ZIP archive: {os.path.abspath(zip_path)}")
        readme_lines = [
            "============================================================",
            "DEEPFAKE INVESTIGATION SYSTEM - BENCHMARK SAMPLE DATASET",
            "============================================================\n",
            f"Exported category filter: {category.upper()}",
            f"Total videos in archive: {len(selected)}\n",
            "MANIFEST:"
        ]
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for item, vpath in selected:
                zf.write(vpath, arcname=item["filename"])
                size_mb = os.path.getsize(vpath) / (1024 * 1024)
                readme_lines.append(f"* [{item['category'].upper()}] {item['filename']} ({size_mb:.2f} MB)")
                readme_lines.append(f"  Title: {item['title']}")
                readme_lines.append(f"  Notes: {item['description']}\n")
                print(f"    [+] Archived: {item['filename']} ({size_mb:.2f} MB)")
            zf.writestr("BENCHMARK_README.txt", "\n".join(readme_lines))
        zip_size_mb = os.path.getsize(zip_path) / (1024 * 1024)
        print(f"[+] ZIP created successfully: {zip_path} ({zip_size_mb:.2f} MB)")


def detect_sample(filename_or_path):
    # Check if filename is in sample_videos/
    candidate = filename_or_path
    if not os.path.exists(candidate):
        candidate = os.path.join(SAMPLE_DIR, filename_or_path)

    if not os.path.exists(candidate):
        print(f"[ERROR] Sample video file not found: {filename_or_path}")
        print("Available sample videos:")
        for item in CATALOG:
            print(f"  - {item['filename']}")
        return

    print("\n" + "=" * 70)
    print("RUNNING FORENSIC DEEPFAKE DETECTOR ON BENCHMARK SAMPLE")
    print(f"Target Video: {os.path.abspath(candidate)}")
    print("=" * 70)

    try:
        from video_detector import detect_video
        result = detect_video(candidate)

        print("\n" + "-" * 70)
        print("INVESTIGATION FORENSIC VERDICT")
        print("-" * 70)
        print(f"Verdict        : {result[0]}")
        print(f"Confidence     : {result[1]:.2f}%")
        print(f"Forensic Risk  : {result[2]}")
        print(f"Fake Frames    : {result[3]}")
        print(f"Real Frames    : {result[4]}")
        print(f"Uncertain      : {result[5]}")
        print(f"Fake Ratio     : {result[6]:.2f}%")
        print(f"Real Ratio     : {result[7]:.2f}%")
        print(f"Avg Fake Prob  : {result[9]:.2f}%")
        print(f"Avg Real Prob  : {result[10]:.2f}%")
        print("=" * 70)
    except Exception as e:
        print(f"[ERROR] Detection failed: {e}")
        import traceback
        traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(
        description="Deepfake Investigation System — Benchmark Sample Videos Utility"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all verified Authentic and Synthetic benchmark sample videos"
    )
    parser.add_argument(
        "--category",
        choices=["all", "real", "fake"],
        default="all",
        help="Filter samples by category (all, real, or fake)"
    )
    parser.add_argument(
        "--outdir",
        type=str,
        help="Target folder to export/download sample videos into"
    )
    parser.add_argument(
        "--zip",
        type=str,
        help="Output path to export sample videos as a .zip archive"
    )
    parser.add_argument(
        "--detect",
        type=str,
        help="Run deepfake detection on a sample video by filename"
    )

    args = parser.parse_args()

    if len(sys.argv) == 1 or args.list:
        list_samples()
        return

    if args.detect:
        detect_sample(args.detect)
        return

    if args.outdir or args.zip:
        export_samples(category=args.category, outdir=args.outdir, zip_path=args.zip)
    else:
        list_samples()


if __name__ == "__main__":
    main()
