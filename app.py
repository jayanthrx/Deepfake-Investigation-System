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
    flash
)

import os
import uuid
from datetime import datetime


# ============================================================
# IMPORT DETECTORS
# ============================================================

from predictor import predict_image
from video_detector import detect_video


# ============================================================
# OPTIONAL MODULES
# ============================================================

try:
    from pdf_report import generate_report
except Exception:
    generate_report = None

try:
    from database import db
except Exception:
    db = None

try:
    from gradcam import generate_heatmap
except Exception:
    generate_heatmap = None


# ============================================================
# FLASK APPLICATION
# ============================================================

app = Flask(__name__)

app.secret_key = "deepfake-investigation-system"


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


os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(IMAGE_FOLDER, exist_ok=True)
os.makedirs(VIDEO_FOLDER, exist_ok=True)
os.makedirs(FRAME_FOLDER, exist_ok=True)
os.makedirs(REPORT_FOLDER, exist_ok=True)


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


# ============================================================
# ALLOWED FILES
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

    if "." not in filename:
        return ""

    return filename.rsplit(
        ".",
        1
    )[1].lower()


def get_risk(prediction, confidence):

    prediction = str(
        prediction
    ).lower()

    try:
        confidence = float(
            confidence
        )
    except Exception:
        confidence = 0.0

    # ------------------------------------------
    # FAKE
    # ------------------------------------------

    if "fake" in prediction:

        if confidence >= 75:
            return "High"

        if confidence >= 55:
            return "Medium"

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


def safe_float(value, default=0.0):

    try:
        return float(value)
    except Exception:
        return default


def safe_int(value, default=0):

    try:
        return int(value)
    except Exception:
        return default


# ============================================================
# HOME
# ============================================================

@app.route("/")
def index():

    return render_template(
        "index.html"
    )


# ============================================================
# PREDICT
# ============================================================

@app.route(
    "/predict",
    methods=["POST"]
)
def predict():

    print()
    print("=" * 60)
    print("NEW INVESTIGATION REQUEST")
    print("=" * 60)


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

        file.save(filepath)


        print()
        print("=" * 60)
        print("IMAGE ANALYSIS STARTED")
        print(filepath)
        print("=" * 60)


        try:

            result = predict_image(
                filepath
            )


            print(
                "IMAGE DETECTOR RETURNED:",
                result
            )


            # ------------------------------------------------
            # DEFAULT VALUES
            # ------------------------------------------------

            prediction = "Uncertain"
            confidence = 0.0
            fake_probability = 0.0
            real_probability = 0.0
            risk = "Medium"


            # ------------------------------------------------
            # RESULT PARSING
            # ------------------------------------------------

            if isinstance(
                result,
                (tuple, list)
            ):

                result_length = len(result)


                # --------------------------------------------
                # 5 VALUES
                # --------------------------------------------

                if result_length >= 5:

                    prediction = result[0]

                    confidence = safe_float(
                        result[1]
                    )

                    fake_probability = safe_float(
                        result[2]
                    )

                    real_probability = safe_float(
                        result[3]
                    )

                    risk = str(
                        result[4]
                    )


                # --------------------------------------------
                # 3 VALUES
                # --------------------------------------------

                elif result_length == 3:

                    prediction = result[0]

                    confidence = safe_float(
                        result[1]
                    )

                    risk = str(
                        result[2]
                    )


                    if str(
                        prediction
                    ).lower() == "fake":

                        fake_probability = confidence

                        real_probability = (
                            100.0 - confidence
                        )

                    elif str(
                        prediction
                    ).lower() == "real":

                        real_probability = confidence

                        fake_probability = (
                            100.0 - confidence
                        )

                    else:

                        # For uncertain result,
                        # confidence represents
                        # the dominant probability.

                        fake_probability = confidence

                        real_probability = (
                            100.0 - confidence
                        )


                # --------------------------------------------
                # 2 VALUES
                # --------------------------------------------

                elif result_length == 2:

                    prediction = result[0]

                    confidence = safe_float(
                        result[1]
                    )

                    risk = get_risk(
                        prediction,
                        confidence
                    )


                    if str(
                        prediction
                    ).lower() == "fake":

                        fake_probability = confidence

                        real_probability = (
                            100.0 - confidence
                        )

                    else:

                        real_probability = confidence

                        fake_probability = (
                            100.0 - confidence
                        )


                else:

                    prediction = str(
                        result[0]
                    )


            else:

                prediction = str(
                    result
                )


            # ------------------------------------------------
            # PRINT RESULT
            # ------------------------------------------------

            print()
            print("=" * 60)
            print("IMAGE RESULT")
            print("=" * 60)

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

            print("=" * 60)


            # ------------------------------------------------
            # IMAGE RESULT PAGE
            # ------------------------------------------------

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

                average_fake_probability=fake_probability,

                average_real_probability=real_probability,

                uploaded_file=(
                    "/uploads/images/"
                    + unique_name
                ),

                video_url=None,

                report_filename=None,

                date=datetime.now().strftime(
                    "%d-%m-%Y %H:%M:%S"
                ),

                error=None
            )


        except Exception as error:

            print(
                "IMAGE ERROR:",
                error
            )


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

        file.save(filepath)


        print()
        print("=" * 60)
        print("VIDEO ANALYSIS STARTED")
        print(filepath)
        print("=" * 60)


        try:

            # =================================================
            # CALL VIDEO DETECTOR
            # =================================================

            video_result = detect_video(
                filepath
            )


            print()
            print("=" * 60)
            print("VIDEO DETECTOR RETURNED")
            print("=" * 60)

            print(
                video_result
            )


            # =================================================
            # DEFAULT VIDEO VALUES
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
            #
            # IMPORTANT:
            #
            # Your detector currently returns:
            #
            # 11 VALUES
            #
            # OR sometimes:
            #
            # 3 VALUES
            #
            # This code handles BOTH.
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


                    # -----------------------------------------
                    # IMPORTANT FIX
                    #
                    # Your detector returns:
                    #
                    # ('Uncertain', 62.76, 'Medium')
                    #
                    # For an uncertain video we use the
                    # confidence as the dominant probability.
                    # -----------------------------------------

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


                # =============================================
                # UNEXPECTED
                # =============================================

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
            # CONVERT VALUES SAFELY
            # =================================================

            video_confidence = safe_float(
                video_confidence
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
            # IF 11-VALUE RESULT HAS FRAME DATA
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
            # DETERMINE RISK IF EMPTY
            # =================================================

            if not video_risk:

                video_risk = get_risk(
                    video_prediction,
                    video_confidence
                )


            # =================================================
            # FINAL VIDEO RESULT
            # =================================================

            print()
            print("=" * 60)
            print("FINAL VIDEO RESULT")
            print("=" * 60)

            print(
                "Prediction:",
                video_prediction
            )

            print(
                "Confidence:",
                video_confidence,
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

            print("=" * 60)


            # =================================================
            # GENERATE PDF REPORT
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


                    # -----------------------------------------
                    # YOUR CURRENT pdf_report.py SIGNATURE
                    # -----------------------------------------

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

                        real_frames=real_frames
                    )


                    # -----------------------------------------
                    # YOUR pdf_report.py creates its own file
                    # -----------------------------------------

                    if isinstance(
                        generated,
                        str
                    ):

                        generated_path = generated


                        # If only filename was returned
                        if not os.path.isabs(
                            generated_path
                        ):

                            generated_path = os.path.join(
                                BASE_DIR,
                                generated_path
                            )


                        if os.path.exists(
                            generated_path
                        ):

                            report_filename = (
                                os.path.basename(
                                    generated_path
                                )
                            )


                    # -----------------------------------------
                    # FALLBACK: LOOK FOR GENERATED REPORT
                    # -----------------------------------------

                    if report_filename is None:

                        possible_report = os.path.join(
                            REPORT_FOLDER,
                            "report_"
                            + original_filename.split(".")[0]
                            + ".pdf"
                        )


                        if os.path.exists(
                            possible_report
                        ):

                            report_filename = (
                                os.path.basename(
                                    possible_report
                                )
                            )


                    print(
                        "PDF REPORT:",
                        report_filename
                    )


                except Exception as report_error:

                    print()
                    print(
                        "PDF REPORT ERROR:",
                        report_error
                    )


            # =================================================
            # RENDER VIDEO RESULT
            # =================================================

            return render_template(

                "result.html",

                filename=original_filename,

                file_type="VIDEO",

                prediction=video_prediction,

                confidence=video_confidence,

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

                fake_percentage=fake_percentage,

                real_percentage=real_percentage,

                uncertain_percentage=uncertain_percentage,

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
            print("=" * 60)
            print("VIDEO ANALYSIS ERROR")
            print("=" * 60)

            print(
                str(error)
            )

            print("=" * 60)


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


@app.route(
    "/uploads/images/<filename>"
)
def uploaded_image(filename):

    return send_from_directory(
        IMAGE_FOLDER,
        filename
    )


@app.route(
    "/uploads/videos/<filename>"
)
def uploaded_video(filename):

    return send_from_directory(
        VIDEO_FOLDER,
        filename
    )


@app.route(
    "/uploads/video_frames/<filename>"
)
def uploaded_frame(filename):

    return send_from_directory(
        FRAME_FOLDER,
        filename
    )


# ============================================================
# REPORT DOWNLOAD
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
    print("=" * 60)
    print("DEEPFAKE INVESTIGATION SYSTEM")
    print("=" * 60)

    print(
        "Server: http://127.0.0.1:5000"
    )

    print("=" * 60)


    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False
    )