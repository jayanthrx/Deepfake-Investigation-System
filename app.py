# ============================================================
# app.py
# DeepFake Investigation System
# ============================================================

from flask import (
    Flask,
    render_template,
    request,
    send_from_directory,
    redirect,
    url_for,
    flash,
    Response,
    jsonify
)

import os
import sys
import uuid
import csv
import io
from datetime import datetime

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="backslashreplace")


# ============================================================
# IMPORT DETECTORS
# ============================================================

from predictor import predict_image, model as efficientnet_model
from video_detector import detect_video


# ============================================================
# OPTIONAL MODULES
# ============================================================

try:
    from pdf_report import generate_report
except Exception as e:
    print("PDF module could not be loaded:", e)
    generate_report = None


try:
    from database import db, Investigation
except Exception as e:
    print("Database module could not be loaded:", e)
    db = None
    Investigation = None


try:
    from gradcam import generate_heatmap
except Exception as e:
    print("GradCAM module could not be loaded:", e)
    generate_heatmap = None


# ============================================================
# FLASK APPLICATION
# ============================================================

app = Flask(__name__)

app.secret_key = "deepfake-investigation-system"


# ============================================================
# BASE DIRECTORY
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)


# ============================================================
# FOLDERS
# ============================================================

UPLOAD_FOLDER = os.path.join(
    BASE_DIR,
    "uploads"
)

IMAGE_FOLDER = os.path.join(
    UPLOAD_FOLDER,
    "images"
)

VIDEO_FOLDER = os.path.join(
    UPLOAD_FOLDER,
    "videos"
)

FRAME_FOLDER = os.path.join(
    UPLOAD_FOLDER,
    "video_frames"
)

REPORT_FOLDER = os.path.join(
    BASE_DIR,
    "reports"
)


# ============================================================
# CREATE FOLDERS
# ============================================================

os.makedirs(
    UPLOAD_FOLDER,
    exist_ok=True
)

os.makedirs(
    IMAGE_FOLDER,
    exist_ok=True
)

os.makedirs(
    VIDEO_FOLDER,
    exist_ok=True
)

os.makedirs(
    FRAME_FOLDER,
    exist_ok=True
)

os.makedirs(
    REPORT_FOLDER,
    exist_ok=True
)


# ============================================================
# FLASK CONFIG
# ============================================================

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

app.config["IMAGE_FOLDER"] = IMAGE_FOLDER

app.config["VIDEO_FOLDER"] = VIDEO_FOLDER

app.config["FRAME_FOLDER"] = FRAME_FOLDER

app.config["REPORT_FOLDER"] = REPORT_FOLDER

app.config["MAX_CONTENT_LENGTH"] = (
    500 * 1024 * 1024
)

app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///history.db"

app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

if db is not None:
    try:
        db.init_app(app)
        with app.app_context():
            db.create_all()
        print("Database initialized successfully.")
    except Exception as dbe:
        print("Database initialization error:", dbe)


# ============================================================
# ALLOWED FILE TYPES
# ============================================================

ALLOWED_IMAGES = {
    "jpg",
    "jpeg",
    "png",
    "bmp",
    "webp"
}


ALLOWED_VIDEOS = {
    "mp4",
    "avi",
    "mov",
    "mkv",
    "webm"
}


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_extension(filename):

    if not filename:
        return ""

    if "." not in filename:
        return ""

    return filename.rsplit(
        ".",
        1
    )[1].lower()


# ============================================================
# SAFE FLOAT
# ============================================================

def safe_float(
    value,
    default=0.0
):

    try:
        return float(value)

    except Exception:
        return default


# ============================================================
# SAFE INT
# ============================================================

def safe_int(
    value,
    default=0
):

    try:
        return int(value)

    except Exception:
        return default


# ============================================================
# NORMALIZE PROBABILITIES
# ============================================================

def normalize_probabilities(
    fake_probability,
    real_probability
):

    fake_probability = safe_float(
        fake_probability
    )

    real_probability = safe_float(
        real_probability
    )

    # ------------------------------------------
    # Keep values inside 0-100
    # ------------------------------------------

    fake_probability = max(
        0.0,
        min(
            100.0,
            fake_probability
        )
    )

    real_probability = max(
        0.0,
        min(
            100.0,
            real_probability
        )
    )

    total = (
        fake_probability
        + real_probability
    )

    # ------------------------------------------
    # If both are zero
    # ------------------------------------------

    if total <= 0:

        return (
            50.0,
            50.0
        )

    # ------------------------------------------
    # Normalize to 100
    # ------------------------------------------

    if abs(total - 100.0) > 0.01:

        fake_probability = (
            fake_probability
            / total
            * 100.0
        )

        real_probability = (
            real_probability
            / total
            * 100.0
        )

    return (
        round(fake_probability, 2),
        round(real_probability, 2)
    )


# ============================================================
# RISK CALCULATION
# ============================================================

def get_risk(
    prediction,
    confidence
):

    prediction = str(
        prediction
    ).strip().lower()

    confidence = safe_float(
        confidence
    )

    # ------------------------------------------
    # FAKE
    # ------------------------------------------

    if "fake" in prediction:

        if confidence >= 85:
            return "Very High"

        elif confidence >= 70:
            return "High"

        elif confidence >= 55:
            return "Medium"

        else:
            return "Low"

    # ------------------------------------------
    # REAL
    # ------------------------------------------

    if "real" in prediction:

        if confidence >= 55:
            return "Low"

        return "Medium"

    # ------------------------------------------
    # UNCERTAIN
    # ------------------------------------------

    return "Medium"


# ============================================================
# HOME PAGE
# ============================================================

@app.route("/")
def index():

    return render_template(
        "index.html"
    )


# ============================================================
# PREDICTION ROUTE
# ============================================================

@app.route(
    "/predict",
    methods=["POST"]
)
def predict():

    print()
    print("=" * 70)
    print("NEW INVESTIGATION REQUEST")
    print("=" * 70)


    # ========================================================
    # CHECK FILE
    # ========================================================

    if "file" not in request.files:

        flash(
            "No file selected."
        )

        return redirect(
            url_for("index")
        )


    file = request.files["file"]


    if file.filename == "":

        flash(
            "No file selected."
        )

        return redirect(
            url_for("index")
        )


    original_filename = file.filename

    extension = get_extension(
        original_filename
    )


    print(
        "Filename:",
        original_filename
    )

    print(
        "Extension:",
        extension
    )


    # ========================================================
    # IMAGE ANALYSIS
    # ========================================================

    if extension in ALLOWED_IMAGES:

        unique_name = (
            uuid.uuid4().hex
            + "."
            + extension
        )


        filepath = os.path.join(
            IMAGE_FOLDER,
            unique_name
        )


        file.save(
            filepath
        )


        print()
        print("=" * 70)
        print("IMAGE ANALYSIS STARTED")
        print("Image:", filepath)
        print("=" * 70)


        try:

            # =================================================
            # CALL IMAGE DETECTOR
            # =================================================

            result = predict_image(
                filepath
            )


            print()
            print("=" * 70)
            print("IMAGE DETECTOR RETURNED")
            print("=" * 70)

            print(
                result
            )


            # =================================================
            # DEFAULT VALUES
            # =================================================

            prediction = "Uncertain"

            confidence = 0.0

            risk = "Medium"

            fake_probability = 0.0

            real_probability = 0.0


            # =================================================
            # PARSE PREDICTOR RESULT
            # =================================================

            if isinstance(
                result,
                (tuple, list)
            ):

                result_length = len(
                    result
                )


                print(
                    "Number of returned values:",
                    result_length
                )


                # =============================================
                # CORRECT 5-VALUE FORMAT
                #
                # predictor.py returns:
                #
                # (
                #   label,
                #   confidence,
                #   risk,
                #   fake_probability,
                #   real_probability
                # )
                # =============================================

                if result_length >= 5:

                    prediction = str(
                        result[0]
                    )

                    confidence = safe_float(
                        result[1]
                    )

                    risk = str(
                        result[2]
                    )

                    fake_probability = safe_float(
                        result[3]
                    )

                    real_probability = safe_float(
                        result[4]
                    )


                # =============================================
                # 3-VALUE FORMAT
                # =============================================

                elif result_length == 3:

                    prediction = str(
                        result[0]
                    )

                    confidence = safe_float(
                        result[1]
                    )

                    risk = str(
                        result[2]
                    )


                    if prediction.lower() == "fake":

                        fake_probability = confidence

                        real_probability = (
                            100.0
                            - confidence
                        )

                    elif prediction.lower() == "real":

                        real_probability = confidence

                        fake_probability = (
                            100.0
                            - confidence
                        )

                    else:

                        fake_probability = confidence

                        real_probability = (
                            100.0
                            - confidence
                        )


                # =============================================
                # 2-VALUE FORMAT
                # =============================================

                elif result_length == 2:

                    prediction = str(
                        result[0]
                    )

                    confidence = safe_float(
                        result[1]
                    )

                    risk = get_risk(
                        prediction,
                        confidence
                    )


                    if prediction.lower() == "fake":

                        fake_probability = confidence

                        real_probability = (
                            100.0
                            - confidence
                        )

                    elif prediction.lower() == "real":

                        real_probability = confidence

                        fake_probability = (
                            100.0
                            - confidence
                        )

                    else:

                        fake_probability = confidence

                        real_probability = (
                            100.0
                            - confidence
                        )


                # =============================================
                # 1-VALUE FORMAT
                # =============================================

                elif result_length == 1:

                    prediction = str(
                        result[0]
                    )

                    confidence = 0.0

                    risk = get_risk(
                        prediction,
                        confidence
                    )


            else:

                prediction = str(
                    result
                )

                confidence = 0.0

                risk = get_risk(
                    prediction,
                    confidence
                )


            # =================================================
            # NORMALIZE PROBABILITIES
            # =================================================

            (
                fake_probability,
                real_probability
            ) = normalize_probabilities(
                fake_probability,
                real_probability
            )


            # =================================================
            # CORRECT RISK
            # =================================================

            if not risk or str(
                risk
            ).lower() in {
                "",
                "unknown",
                "none",
                "unknown risk"
            }:

                risk = get_risk(
                    prediction,
                    confidence
                )


            # =================================================
            # ROUND VALUES
            # =================================================

            confidence = round(
                confidence,
                2
            )

            fake_probability = round(
                fake_probability,
                2
            )

            real_probability = round(
                real_probability,
                2
            )


            # =================================================
            # TERMINAL OUTPUT
            # =================================================

            print()
            print("=" * 70)
            print("FINAL IMAGE RESULT")
            print("=" * 70)

            print(
                "Prediction:",
                prediction
            )

            print(
                "Confidence:",
                confidence,
                "%"
            )

            print(
                "Fake Probability:",
                fake_probability,
                "%"
            )

            print(
                "Real Probability:",
                real_probability,
                "%"
            )

            print(
                "Risk:",
                risk
            )

            print("=" * 70)


            # =================================================
            # GENERATE GRAD-CAM HEATMAP
            # =================================================

            heatmap_filename = None
            heatmap_url = None
            heatmap_path = None

            if generate_heatmap is not None:
                try:
                    heatmap_filename = "heatmap_" + unique_name
                    target_heatmap_path = os.path.join(
                        IMAGE_FOLDER,
                        heatmap_filename
                    )
                    hm_res = generate_heatmap(
                        efficientnet_model,
                        filepath,
                        output_path=target_heatmap_path
                    )
                    if hm_res and os.path.exists(hm_res):
                        heatmap_path = hm_res
                        heatmap_url = "/uploads/images/" + heatmap_filename
                        print("Grad-CAM Heatmap generated at:", heatmap_path)
                except Exception as hm_err:
                    print("Grad-CAM generation error:", hm_err)


            # =================================================
            # SAVE TO DATABASE
            # =================================================

            if db is not None and Investigation is not None:
                try:
                    inv_record = Investigation(
                        filename=original_filename,
                        result=prediction,
                        confidence=confidence,
                        risk=risk,
                        date=datetime.now().strftime("%d-%m-%Y %H:%M:%S")
                    )
                    db.session.add(inv_record)
                    db.session.commit()
                    print("Investigation saved to database successfully.")
                except Exception as db_err:
                    db.session.rollback()
                    print("Database save error:", db_err)


            # =================================================
            # GENERATE IMAGE PDF REPORT
            # =================================================

            report_filename = None


            if generate_report is not None:

                try:

                    report_name = (
                        "deepfake_report_"
                        + uuid.uuid4().hex
                        + ".pdf"
                    )


                    report_path = os.path.join(
                        REPORT_FOLDER,
                        report_name
                    )


                    print()
                    print(
                        "Generating image PDF report..."
                    )


                    generated = generate_report(

                        filename=original_filename,

                        result=prediction,

                        confidence=confidence,

                        risk=risk,

                        date=datetime.now().strftime(
                            "%d-%m-%Y %H:%M:%S"
                        ),

                        image_path=filepath,

                        heatmap_path=heatmap_path,

                        media_type="Image",

                        fake_probability=fake_probability,

                        real_probability=real_probability,

                        fake_frames=0,

                        real_frames=0,

                        uncertain_frames=0,

                        fake_percentage=0,

                        real_percentage=0,

                        uncertain_percentage=0,

                        average_fake_probability=(
                            fake_probability
                        ),

                        average_real_probability=(
                            real_probability
                        ),

                        output_path=report_path
                    )


                    # -----------------------------------------
                    # CHECK RETURNED FILE
                    # -----------------------------------------

                    if isinstance(
                        generated,
                        str
                    ):

                        generated_path = generated


                        # If relative path
                        if not os.path.isabs(
                            generated_path
                        ):

                            possible_path = os.path.join(
                                BASE_DIR,
                                generated_path
                            )

                            if os.path.exists(
                                possible_path
                            ):

                                generated_path = (
                                    possible_path
                                )

                            else:

                                generated_path = os.path.join(
                                    REPORT_FOLDER,
                                    generated_path
                                )


                        if os.path.exists(
                            generated_path
                        ):

                            report_filename = os.path.basename(
                                generated_path
                            )


                    # -----------------------------------------
                    # FALLBACK: CHECK OUR OUTPUT PATH
                    # -----------------------------------------

                    if report_filename is None:

                        if os.path.exists(
                            report_path
                        ):

                            report_filename = os.path.basename(
                                report_path
                            )


                    print(
                        "IMAGE PDF REPORT:",
                        report_filename
                    )


                except Exception as report_error:

                    print()
                    print(
                        "=" * 70
                    )

                    print(
                        "IMAGE PDF REPORT ERROR:"
                    )

                    print(
                        str(report_error)
                    )

                    print(
                        "=" * 70
                    )


            else:

                print(
                    "PDF generator is not available."
                )


            # =================================================
            # IMAGE RESULT PAGE
            # =================================================

            return render_template(

                "result.html",

                filename=original_filename,

                file_type="IMAGE",

                prediction=prediction,

                confidence=confidence,

                fake_probability=fake_probability,

                real_probability=real_probability,

                risk=risk,

                fake_frames=0,

                real_frames=0,

                uncertain_frames=0,

                fake_percentage=0,

                real_percentage=0,

                uncertain_percentage=0,

                average_fake_probability=(
                    fake_probability
                ),

                average_real_probability=(
                    real_probability
                ),

                uploaded_file=(
                    "/uploads/images/"
                    + unique_name
                ),

                heatmap_url=heatmap_url,

                video_url=None,

                report_filename=report_filename,

                date=datetime.now().strftime(
                    "%d-%m-%Y %H:%M:%S"
                ),

                error=None
            )


        except Exception as error:

            print()
            print("=" * 70)
            print("IMAGE ANALYSIS ERROR")
            print("=" * 70)

            print(
                str(error)
            )

            print("=" * 70)


            return render_template(

                "result.html",

                filename=original_filename,

                file_type="IMAGE",

                prediction="Error",

                confidence=0,

                fake_probability=0,

                real_probability=0,

                risk="Unknown",

                fake_frames=0,

                real_frames=0,

                uncertain_frames=0,

                fake_percentage=0,

                real_percentage=0,

                uncertain_percentage=0,

                average_fake_probability=0,

                average_real_probability=0,

                uploaded_file=None,

                video_url=None,

                report_filename=None,

                date=datetime.now().strftime(
                    "%d-%m-%Y %H:%M:%S"
                ),

                error=str(error)
            )


    # ========================================================
    # VIDEO ANALYSIS
    # ========================================================

    elif extension in ALLOWED_VIDEOS:

        unique_name = (
            uuid.uuid4().hex
            + "."
            + extension
        )


        filepath = os.path.join(
            VIDEO_FOLDER,
            unique_name
        )


        file.save(
            filepath
        )


        print()
        print("=" * 70)
        print("VIDEO ANALYSIS STARTED")
        print("Video:", filepath)
        print("=" * 70)


        try:

            # =================================================
            # CALL VIDEO DETECTOR
            # =================================================

            video_result = detect_video(
                filepath
            )


            print()
            print("=" * 70)
            print("VIDEO DETECTOR RETURNED")
            print("=" * 70)

            print(
                video_result
            )


            # =================================================
            # DEFAULT VALUES
            # =================================================

            video_prediction = "Uncertain"

            video_confidence = 0.0

            video_risk = "Medium"

            fake_frames = 0

            real_frames = 0

            uncertain_frames = 0

            fake_percentage = 0.0

            real_percentage = 0.0

            uncertain_percentage = 0.0

            average_fake_probability = 0.0

            average_real_probability = 0.0


            # =================================================
            # PARSE VIDEO RESULT
            # =================================================

            if isinstance(
                video_result,
                (tuple, list)
            ):

                result_length = len(
                    video_result
                )


                print(
                    "Number of returned values:",
                    result_length
                )


                # =============================================
                # 11 VALUES
                # =============================================

                if result_length == 11:

                    (
                        video_prediction,

                        video_confidence,

                        video_risk,

                        fake_frames,

                        real_frames,

                        uncertain_frames,

                        fake_percentage,

                        real_percentage,

                        uncertain_percentage,

                        average_fake_probability,

                        average_real_probability

                    ) = video_result


                # =============================================
                # 3 VALUES
                # =============================================

                elif result_length == 3:

                    (
                        video_prediction,

                        video_confidence,

                        video_risk

                    ) = video_result


                    video_confidence = safe_float(
                        video_confidence
                    )


                    if str(
                        video_prediction
                    ).lower() == "fake":

                        average_fake_probability = (
                            video_confidence
                        )

                        average_real_probability = (
                            100.0
                            - video_confidence
                        )


                    elif str(
                        video_prediction
                    ).lower() == "real":

                        average_real_probability = (
                            video_confidence
                        )

                        average_fake_probability = (
                            100.0
                            - video_confidence
                        )


                    else:

                        average_fake_probability = (
                            video_confidence
                        )

                        average_real_probability = (
                            100.0
                            - video_confidence
                        )


                # =============================================
                # 1 VALUE
                # =============================================

                elif result_length == 1:

                    video_prediction = str(
                        video_result[0]
                    )

                    video_confidence = 0.0

                    video_risk = "Medium"


                else:

                    raise ValueError(
                        "Unexpected number of values returned "
                        "by video detector: "
                        + str(result_length)
                    )


            else:

                raise ValueError(
                    "Video detector returned a non-list result: "
                    + str(video_result)
                )


            # =================================================
            # SAFE CONVERSIONS
            # =================================================

            video_prediction = str(
                video_prediction
            )

            video_confidence = safe_float(
                video_confidence
            )

            video_risk = str(
                video_risk
            )

            fake_frames = safe_int(
                fake_frames
            )

            real_frames = safe_int(
                real_frames
            )

            uncertain_frames = safe_int(
                uncertain_frames
            )

            fake_percentage = safe_float(
                fake_percentage
            )

            real_percentage = safe_float(
                real_percentage
            )

            uncertain_percentage = safe_float(
                uncertain_percentage
            )

            average_fake_probability = safe_float(
                average_fake_probability
            )

            average_real_probability = safe_float(
                average_real_probability
            )


            # =================================================
            # NORMALIZE VIDEO PROBABILITIES
            # =================================================

            (
                average_fake_probability,
                average_real_probability
            ) = normalize_probabilities(

                average_fake_probability,

                average_real_probability
            )


            # =================================================
            # FRAME PERCENTAGES
            # =================================================

            total_frames = (
                fake_frames
                + real_frames
                + uncertain_frames
            )


            if total_frames > 0:

                fake_percentage = (
                    fake_frames
                    / total_frames
                    * 100
                )

                real_percentage = (
                    real_frames
                    / total_frames
                    * 100
                )

                uncertain_percentage = (
                    uncertain_frames
                    / total_frames
                    * 100
                )


            # =================================================
            # VIDEO RISK
            # =================================================

            if (
                not video_risk
                or str(video_risk).lower()
                in {
                    "",
                    "unknown",
                    "none",
                    "unknown risk"
                }
            ):

                video_risk = get_risk(
                    video_prediction,
                    video_confidence
                )


            # =================================================
            # FINAL VIDEO RESULT
            # =================================================

            print()
            print("=" * 70)
            print("FINAL VIDEO RESULT")
            print("=" * 70)

            print(
                "Prediction:",
                video_prediction
            )

            print(
                "Confidence:",
                round(
                    video_confidence,
                    2
                ),
                "%"
            )

            print(
                "Fake Frames:",
                fake_frames
            )

            print(
                "Real Frames:",
                real_frames
            )

            print(
                "Uncertain Frames:",
                uncertain_frames
            )

            print(
                "Fake Percentage:",
                round(
                    fake_percentage,
                    2
                ),
                "%"
            )

            print(
                "Real Percentage:",
                round(
                    real_percentage,
                    2
                ),
                "%"
            )

            print(
                "Uncertain Percentage:",
                round(
                    uncertain_percentage,
                    2
                ),
                "%"
            )

            print(
                "Average Fake Probability:",
                round(
                    average_fake_probability,
                    2
                ),
                "%"
            )

            print(
                "Average Real Probability:",
                round(
                    average_real_probability,
                    2
                ),
                "%"
            )

            print(
                "Risk:",
                video_risk
            )

            print("=" * 70)


            # =================================================
            # SAVE TO DATABASE
            # =================================================

            if db is not None and Investigation is not None:
                try:
                    inv_record = Investigation(
                        filename=original_filename,
                        result=video_prediction,
                        confidence=round(video_confidence, 2),
                        risk=video_risk,
                        date=datetime.now().strftime("%d-%m-%Y %H:%M:%S")
                    )
                    db.session.add(inv_record)
                    db.session.commit()
                    print("Video investigation saved to database successfully.")
                except Exception as db_err:
                    db.session.rollback()
                    print("Database save error:", db_err)


            # =================================================
            # GENERATE VIDEO PDF REPORT
            # =================================================

            report_filename = None


            if generate_report is not None:

                try:

                    report_name = (
                        "deepfake_report_"
                        + uuid.uuid4().hex
                        + ".pdf"
                    )


                    report_path = os.path.join(
                        REPORT_FOLDER,
                        report_name
                    )


                    print()
                    print(
                        "Generating video PDF report..."
                    )


                    generated = generate_report(

                        filename=original_filename,

                        result=video_prediction,

                        confidence=video_confidence,

                        risk=video_risk,

                        date=datetime.now().strftime(
                            "%d-%m-%Y %H:%M:%S"
                        ),

                        image_path=None,

                        heatmap_path=None,

                        media_type="Video",

                        fake_probability=(
                            average_fake_probability
                        ),

                        real_probability=(
                            average_real_probability
                        ),

                        fake_frames=fake_frames,

                        real_frames=real_frames,

                        uncertain_frames=uncertain_frames,

                        fake_percentage=fake_percentage,

                        real_percentage=real_percentage,

                        uncertain_percentage=uncertain_percentage,

                        average_fake_probability=(
                            average_fake_probability
                        ),

                        average_real_probability=(
                            average_real_probability
                        ),

                        output_path=report_path
                    )


                    # -----------------------------------------
                    # CHECK RETURNED FILE
                    # -----------------------------------------

                    if isinstance(
                        generated,
                        str
                    ):

                        generated_path = generated


                        if not os.path.isabs(
                            generated_path
                        ):

                            possible_path = os.path.join(
                                BASE_DIR,
                                generated_path
                            )

                            if os.path.exists(
                                possible_path
                            ):

                                generated_path = (
                                    possible_path
                                )

                            else:

                                generated_path = os.path.join(
                                    REPORT_FOLDER,
                                    generated_path
                                )


                        if os.path.exists(
                            generated_path
                        ):

                            report_filename = os.path.basename(
                                generated_path
                            )


                    # -----------------------------------------
                    # FALLBACK
                    # -----------------------------------------

                    if report_filename is None:

                        if os.path.exists(
                            report_path
                        ):

                            report_filename = os.path.basename(
                                report_path
                            )


                    print(
                        "VIDEO PDF REPORT:",
                        report_filename
                    )


                except Exception as report_error:

                    print()
                    print(
                        "=" * 70
                    )

                    print(
                        "VIDEO PDF REPORT ERROR:"
                    )

                    print(
                        str(report_error)
                    )

                    print(
                        "=" * 70
                    )


            # =================================================
            # VIDEO RESULT PAGE
            # =================================================

            return render_template(

                "result.html",

                filename=original_filename,

                file_type="VIDEO",

                prediction=video_prediction,

                confidence=round(
                    video_confidence,
                    2
                ),

                fake_probability=(
                    average_fake_probability
                ),

                real_probability=(
                    average_real_probability
                ),

                risk=video_risk,

                fake_frames=fake_frames,

                real_frames=real_frames,

                uncertain_frames=uncertain_frames,

                fake_percentage=round(
                    fake_percentage,
                    2
                ),

                real_percentage=round(
                    real_percentage,
                    2
                ),

                uncertain_percentage=round(
                    uncertain_percentage,
                    2
                ),

                average_fake_probability=(
                    average_fake_probability
                ),

                average_real_probability=(
                    average_real_probability
                ),

                uploaded_file=(
                    "/uploads/videos/"
                    + unique_name
                ),

                video_url=(
                    "/uploads/videos/"
                    + unique_name
                ),

                report_filename=report_filename,

                date=datetime.now().strftime(
                    "%d-%m-%Y %H:%M:%S"
                ),

                error=None
            )


        except Exception as error:

            print()
            print("=" * 70)
            print("VIDEO ANALYSIS ERROR")
            print("=" * 70)

            print(
                str(error)
            )

            print("=" * 70)


            return render_template(

                "result.html",

                filename=original_filename,

                file_type="VIDEO",

                prediction="Error",

                confidence=0,

                fake_probability=0,

                real_probability=0,

                risk="Unknown",

                fake_frames=0,

                real_frames=0,

                uncertain_frames=0,

                fake_percentage=0,

                real_percentage=0,

                uncertain_percentage=0,

                average_fake_probability=0,

                average_real_probability=0,

                uploaded_file=None,

                video_url=None,

                report_filename=None,

                date=datetime.now().strftime(
                    "%d-%m-%Y %H:%M:%S"
                ),

                error=str(error)
            )


    # ========================================================
    # INVALID FILE
    # ========================================================

    else:

        flash(
            "Unsupported file type."
        )

        return redirect(
            url_for("index")
        )


# ============================================================
# SERVE UPLOADS
# ============================================================

@app.route(
    "/uploads/<path:filename>"
)
def uploaded_file(filename):

    return send_from_directory(
        UPLOAD_FOLDER,
        filename
    )


# ============================================================
# SERVE IMAGES
# ============================================================

@app.route(
    "/uploads/images/<filename>"
)
def uploaded_image(filename):

    return send_from_directory(
        IMAGE_FOLDER,
        filename
    )


# ============================================================
# SERVE VIDEOS
# ============================================================

@app.route(
    "/uploads/videos/<filename>"
)
def uploaded_video(filename):

    return send_from_directory(
        VIDEO_FOLDER,
        filename
    )


# ============================================================
# SERVE VIDEO FRAMES
# ============================================================

@app.route(
    "/uploads/video_frames/<filename>"
)
def uploaded_frame(filename):

    return send_from_directory(
        FRAME_FOLDER,
        filename
    )


# ============================================================
# DOWNLOAD PDF REPORT
# ============================================================

@app.route(
    "/reports/<filename>"
)
def download_report(filename):

    return send_from_directory(
        REPORT_FOLDER,
        filename,
        as_attachment=True
    )


@app.route(
    "/download/<path:filename>"
)
def download(filename):

    return send_from_directory(
        REPORT_FOLDER,
        filename,
        as_attachment=True
    )


# ============================================================
# ABOUT
# ============================================================

@app.route("/about")
def about():

    return render_template(
        "about.html"
    )


# ============================================================
# ARCHITECTURE
# ============================================================

@app.route("/architecture")
def architecture():

    return render_template(
        "architecture.html"
    )


# ============================================================
# RESULTS
# ============================================================

@app.route("/results")
def results():

    return render_template(
        "results.html"
    )


# ============================================================
# DASHBOARD
# ============================================================

@app.route("/dashboard")
def dashboard():

    records = []
    total = 0
    fake = 0
    real = 0
    uncertain = 0
    avg_confidence = 0.0

    if db is not None and Investigation is not None:
        try:
            records = Investigation.query.order_by(
                Investigation.id.desc()
            ).all()

            total = len(records)

            fake = Investigation.query.filter_by(
                result="Fake"
            ).count()

            real = Investigation.query.filter_by(
                result="Real"
            ).count()

            uncertain = Investigation.query.filter_by(
                result="Uncertain"
            ).count()

            if total > 0:
                avg_confidence = round(
                    sum(float(r.confidence) for r in records) / total,
                    2
                )
        except Exception as e:
            print("Dashboard query error:", e)

    return render_template(
        "dashboard.html",
        records=records,
        total=total,
        fake=fake,
        real=real,
        uncertain=uncertain,
        avg_confidence=avg_confidence
    )


# ============================================================
# HISTORY
# ============================================================

@app.route("/history")
def history():

    data = []

    if db is not None and Investigation is not None:
        try:
            records = Investigation.query.order_by(
                Investigation.id.desc()
            ).all()

            data = [
                [r.filename, r.result, round(float(r.confidence), 2), r.risk, r.date]
                for r in records
            ]
        except Exception as e:
            print("History query error:", e)

    return render_template(
        "history.html",
        data=data
    )


# ============================================================
# EXPORT CSV
# ============================================================

@app.route("/export_csv")
def export_csv():

    records = []
    if db is not None and Investigation is not None:
        try:
            records = Investigation.query.order_by(
                Investigation.id.desc()
            ).all()
        except Exception as e:
            print("CSV Export query error:", e)

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "ID",
        "Filename",
        "Prediction",
        "Confidence (%)",
        "Risk Level",
        "Date Analyzed"
    ])

    for r in records:
        writer.writerow([
            r.id,
            r.filename,
            r.result,
            round(float(r.confidence), 2),
            r.risk,
            r.date
        ])

    csv_data = output.getvalue()
    return Response(
        csv_data,
        mimetype="text/csv",
        headers={
            "Content-Disposition": "attachment; filename=deepfake_investigations.csv"
        }
    )


# ============================================================
# CLEAR HISTORY
# ============================================================

@app.route("/clear_history", methods=["POST", "GET"])
def clear_history():

    if db is not None and Investigation is not None:
        try:
            Investigation.query.delete()
            db.session.commit()
            flash("Investigation history cleared successfully.")
        except Exception as e:
            db.session.rollback()
            flash(f"Error clearing history: {e}")

    return redirect(
        url_for("dashboard")
    )


# ============================================================
# API PREDICT ENDPOINT (JSON REST API)
# ============================================================

@app.route("/api/predict", methods=["POST"])
def api_predict():

    if "file" not in request.files:
        return jsonify({
            "success": False,
            "error": "No file uploaded. Please provide a 'file' parameter."
        }), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({
            "success": False,
            "error": "No file selected."
        }), 400

    original_filename = file.filename
    extension = get_extension(original_filename)

    if extension in ALLOWED_IMAGES:
        unique_name = uuid.uuid4().hex + "." + extension
        image_path = os.path.join(IMAGE_FOLDER, unique_name)
        file.save(image_path)

        prediction, confidence, risk, fake_prob, real_prob = predict_image(image_path)

        heatmap_name = f"heatmap_{unique_name}"
        heatmap_path = os.path.join(IMAGE_FOLDER, heatmap_name)
        heatmap_generated = False
        if efficientnet_model is not None:
            try:
                res = generate_heatmap(
                    model=efficientnet_model,
                    img_path=image_path,
                    output_path=heatmap_path
                )
                if res is not None:
                    heatmap_generated = True
            except Exception as he:
                print("API Grad-CAM error:", he)

        report_name = f"deepfake_report_{uuid.uuid4().hex}.pdf"
        report_path = os.path.join(REPORT_FOLDER, report_name)
        if generate_report is not None:
            try:
                generate_report(
                    filename=original_filename,
                    prediction=prediction,
                    confidence=confidence,
                    risk=risk,
                    media_type="Image",
                    image_path=image_path,
                    heatmap_path=heatmap_path if heatmap_generated else None,
                    fake_probability=fake_prob,
                    real_probability=real_prob,
                    output_path=report_path
                )
            except Exception as pe:
                print("API PDF error:", pe)

        if db is not None and Investigation is not None:
            try:
                inv = Investigation(
                    filename=original_filename,
                    result=prediction,
                    confidence=round(float(confidence), 2),
                    risk=risk,
                    date=datetime.now().strftime("%d-%m-%Y %H:%M:%S")
                )
                db.session.add(inv)
                db.session.commit()
            except Exception as dbe:
                db.session.rollback()

        return jsonify({
            "success": True,
            "media_type": "Image",
            "filename": original_filename,
            "prediction": prediction,
            "confidence": round(float(confidence), 2),
            "risk": risk,
            "fake_probability": round(float(fake_prob), 2),
            "real_probability": round(float(real_prob), 2),
            "image_url": f"/uploads/images/{unique_name}",
            "heatmap_url": f"/uploads/images/{heatmap_name}" if heatmap_generated else None,
            "report_download_url": f"/download/{report_name}"
        })

    elif extension in ALLOWED_VIDEOS:
        unique_name = uuid.uuid4().hex + "." + extension
        video_path = os.path.join(VIDEO_FOLDER, unique_name)
        file.save(video_path)

        (
            prediction,
            confidence,
            risk,
            fake_frames,
            real_frames,
            uncertain_frames,
            fake_pct,
            real_pct,
            uncertain_pct,
            avg_fake_prob,
            avg_real_prob
        ) = detect_video(video_path)

        report_name = f"deepfake_report_{uuid.uuid4().hex}.pdf"
        report_path = os.path.join(REPORT_FOLDER, report_name)
        if generate_report is not None:
            try:
                generate_report(
                    filename=original_filename,
                    prediction=prediction,
                    confidence=confidence,
                    risk=risk,
                    media_type="Video",
                    fake_frames=fake_frames,
                    real_frames=real_frames,
                    uncertain_frames=uncertain_frames,
                    fake_percentage=fake_pct,
                    real_percentage=real_pct,
                    uncertain_percentage=uncertain_pct,
                    average_fake_probability=avg_fake_prob,
                    average_real_probability=avg_real_prob,
                    output_path=report_path
                )
            except Exception as pe:
                print("API Video PDF error:", pe)

        if db is not None and Investigation is not None:
            try:
                inv = Investigation(
                    filename=original_filename,
                    result=prediction,
                    confidence=round(float(confidence), 2),
                    risk=risk,
                    date=datetime.now().strftime("%d-%m-%Y %H:%M:%S")
                )
                db.session.add(inv)
                db.session.commit()
            except Exception as dbe:
                db.session.rollback()

        return jsonify({
            "success": True,
            "media_type": "Video",
            "filename": original_filename,
            "prediction": prediction,
            "confidence": round(float(confidence), 2),
            "risk": risk,
            "fake_frames": fake_frames,
            "real_frames": real_frames,
            "uncertain_frames": uncertain_frames,
            "fake_percentage": round(float(fake_pct), 2),
            "real_percentage": round(float(real_pct), 2),
            "uncertain_percentage": round(float(uncertain_pct), 2),
            "video_url": f"/uploads/videos/{unique_name}",
            "report_download_url": f"/download/{report_name}"
        })

    else:
        return jsonify({
            "success": False,
            "error": f"Unsupported media format '{extension}'. Allowed: {list(ALLOWED_IMAGES | ALLOWED_VIDEOS)}"
        }), 400


# ============================================================
# HEALTH CHECK
# ============================================================

@app.route("/health")
def health():

    return {
        "status": "running",
        "system": "DeepFake Investigation System"
    }


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 70)
    print("DEEPFAKE INVESTIGATION SYSTEM")
    print("=" * 70)

    print(
        "Server: http://127.0.0.1:5000"
    )

    print(
        "Reports Folder:",
        REPORT_FOLDER
    )

    print("=" * 70)


    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False
    )