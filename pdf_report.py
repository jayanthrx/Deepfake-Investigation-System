from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Image,
    Table,
    TableStyle
)

from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4

import os
import re
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="backslashreplace")


def sanitize_pdf_text(text):
    if text is None:
        return ""
    # Strip emojis and characters unprintable by standard PDF fonts
    clean = "".join(c for c in str(text) if ord(c) < 128 or (160 <= ord(c) <= 255))
    return clean if clean.strip() else "Investigation_File"


# ============================================================
# REPORT FOLDER
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

REPORT_FOLDER = os.path.join(
    BASE_DIR,
    "reports"
)

os.makedirs(
    REPORT_FOLDER,
    exist_ok=True
)


# ============================================================
# SAFE FILE NAME
# ============================================================

def safe_filename(filename):

    name = os.path.splitext(
        os.path.basename(filename)
    )[0]

    name = re.sub(
        r"[^a-zA-Z0-9_-]+",
        "_",
        name
    )

    return name[:80]


# ============================================================
# GENERATE REPORT
# ============================================================

def generate_report(
        filename,
        result=None,
        confidence=0,
        risk="Unknown",
        date=None,
        image_path=None,
        heatmap_path=None,
        media_type="Video",
        fake_probability=0,
        real_probability=0,
        fake_frames=0,
        real_frames=0,
        uncertain_frames=0,
        fake_percentage=0,
        real_percentage=0,
        uncertain_percentage=0,
        average_fake_probability=0,
        average_real_probability=0,
        prediction=None,
        output_path=None
):

    # --------------------------------------------------------
    # Support both "result" and "prediction"
    # --------------------------------------------------------

    if prediction is not None:

        result = prediction

    if result is None:

        result = "Uncertain"


    # --------------------------------------------------------
    # Date
    # --------------------------------------------------------

    if date is None:

        from datetime import datetime

        date = datetime.now().strftime(
            "%d-%m-%Y %H:%M:%S"
        )


    # --------------------------------------------------------
    # Convert values
    # --------------------------------------------------------

    try:
        confidence = float(
            confidence
        )
    except Exception:
        confidence = 0.0


    try:
        fake_probability = float(
            fake_probability
        )
    except Exception:
        fake_probability = 0.0


    try:
        real_probability = float(
            real_probability
        )
    except Exception:
        real_probability = 0.0


    try:
        fake_frames = int(
            fake_frames
        )
    except Exception:
        fake_frames = 0


    try:
        real_frames = int(
            real_frames
        )
    except Exception:
        real_frames = 0


    try:
        uncertain_frames = int(
            uncertain_frames
        )
    except Exception:
        uncertain_frames = 0


    try:
        fake_percentage = float(
            fake_percentage
        )
    except Exception:
        fake_percentage = 0.0


    try:
        real_percentage = float(
            real_percentage
        )
    except Exception:
        real_percentage = 0.0


    try:
        uncertain_percentage = float(
            uncertain_percentage
        )
    except Exception:
        uncertain_percentage = 0.0


    try:
        average_fake_probability = float(
            average_fake_probability
        )
    except Exception:
        average_fake_probability = 0.0


    try:
        average_real_probability = float(
            average_real_probability
        )
    except Exception:
        average_real_probability = 0.0


    # ========================================================
    # REPORT FILE
    # ========================================================

    if output_path:

        report_path = output_path

        report_name = os.path.basename(
            report_path
        )

    else:

        report_name = (
            "report_"
            + safe_filename(filename)
            + ".pdf"
        )

        report_path = os.path.join(
            REPORT_FOLDER,
            report_name
        )


    # ========================================================
    # PDF DOCUMENT
    # ========================================================

    doc = SimpleDocTemplate(
        report_path,
        pagesize=A4,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40
    )


    styles = getSampleStyleSheet()

    content = []


    # ========================================================
    # TITLE
    # ========================================================

    content.append(
        Paragraph(
            "Deepfake Investigation Report",
            styles["Title"]
        )
    )

    content.append(
        Spacer(
            1,
            20
        )
    )


    # ========================================================
    # SUMMARY
    # ========================================================

    content.append(
        Paragraph(
            "Investigation Summary",
            styles["Heading2"]
        )
    )

    content.append(
        Spacer(
            1,
            10
        )
    )


    summary_data = [

        [
            "File Name",
            sanitize_pdf_text(filename)
        ],

        [
            "Media Type",
            sanitize_pdf_text(media_type)
        ],

        [
            "Analysis Date",
            sanitize_pdf_text(date)
        ],

        [
            "Final Prediction",
            sanitize_pdf_text(result)
        ],

        [
            "Confidence",
            f"{confidence:.2f}%"
        ],

        [
            "Risk Level",
            sanitize_pdf_text(risk)
        ]

    ]


    summary_table = Table(
        summary_data,
        colWidths=[
            150,
            330
        ]
    )


    summary_table.setStyle(
        TableStyle([

            (
                "BACKGROUND",
                (0, 0),
                (0, -1),
                colors.lightgrey
            ),

            (
                "TEXTCOLOR",
                (0, 0),
                (-1, -1),
                colors.black
            ),

            (
                "FONTNAME",
                (0, 0),
                (0, -1),
                "Helvetica-Bold"
            ),

            (
                "GRID",
                (0, 0),
                (-1, -1),
                0.5,
                colors.grey
            ),

            (
                "VALIGN",
                (0, 0),
                (-1, -1),
                "TOP"
            ),

            (
                "LEFTPADDING",
                (0, 0),
                (-1, -1),
                8
            ),

            (
                "RIGHTPADDING",
                (0, 0),
                (-1, -1),
                8
            ),

            (
                "TOPPADDING",
                (0, 0),
                (-1, -1),
                8
            ),

            (
                "BOTTOMPADDING",
                (0, 0),
                (-1, -1),
                8
            )

        ])
    )


    content.append(
        summary_table
    )


    content.append(
        Spacer(
            1,
            25
        )
    )


    # ========================================================
    # PROBABILITIES
    # ========================================================

    content.append(
        Paragraph(
            "Model Probability Analysis",
            styles["Heading2"]
        )
    )

    content.append(
        Spacer(
            1,
            10
        )
    )


    probability_data = [

        [
            "Measurement",
            "Value"
        ],

        [
            "Fake Probability",
            f"{fake_probability:.2f}%"
        ],

        [
            "Real Probability",
            f"{real_probability:.2f}%"
        ],

        [
            "Average Fake Probability",
            f"{average_fake_probability:.2f}%"
        ],

        [
            "Average Real Probability",
            f"{average_real_probability:.2f}%"
        ]

    ]


    probability_table = Table(
        probability_data,
        colWidths=[
            300,
            180
        ]
    )


    probability_table.setStyle(
        TableStyle([

            (
                "BACKGROUND",
                (0, 0),
                (-1, 0),
                colors.lightgrey
            ),

            (
                "FONTNAME",
                (0, 0),
                (-1, 0),
                "Helvetica-Bold"
            ),

            (
                "GRID",
                (0, 0),
                (-1, -1),
                0.5,
                colors.grey
            ),

            (
                "ALIGN",
                (1, 1),
                (1, -1),
                "RIGHT"
            ),

            (
                "TOPPADDING",
                (0, 0),
                (-1, -1),
                8
            ),

            (
                "BOTTOMPADDING",
                (0, 0),
                (-1, -1),
                8
            )

        ])
    )


    content.append(
        probability_table
    )


    content.append(
        Spacer(
            1,
            25
        )
    )


    # ========================================================
    # VIDEO FRAME ANALYSIS
    # ========================================================

    content.append(
        Paragraph(
            "Video Frame Analysis",
            styles["Heading2"]
        )
    )

    content.append(
        Spacer(
            1,
            10
        )
    )


    frame_data = [

        [
            "Category",
            "Frames",
            "Percentage"
        ],

        [
            "Fake",
            str(fake_frames),
            f"{fake_percentage:.2f}%"
        ],

        [
            "Real",
            str(real_frames),
            f"{real_percentage:.2f}%"
        ],

        [
            "Uncertain",
            str(uncertain_frames),
            f"{uncertain_percentage:.2f}%"
        ]

    ]


    frame_table = Table(
        frame_data,
        colWidths=[
            200,
            140,
            140
        ]
    )


    frame_table.setStyle(
        TableStyle([

            (
                "BACKGROUND",
                (0, 0),
                (-1, 0),
                colors.lightgrey
            ),

            (
                "FONTNAME",
                (0, 0),
                (-1, 0),
                "Helvetica-Bold"
            ),

            (
                "GRID",
                (0, 0),
                (-1, -1),
                0.5,
                colors.grey
            ),

            (
                "ALIGN",
                (1, 1),
                (-1, -1),
                "CENTER"
            ),

            (
                "TOPPADDING",
                (0, 0),
                (-1, -1),
                8
            ),

            (
                "BOTTOMPADDING",
                (0, 0),
                (-1, -1),
                8
            )

        ])
    )


    content.append(
        frame_table
    )


    content.append(
        Spacer(
            1,
            25
        )
    )


    # ========================================================
    # IMAGE
    # ========================================================

    if (
        image_path
        and
        os.path.exists(image_path)
    ):

        content.append(
            Paragraph(
                "Analyzed Image",
                styles["Heading2"]
            )
        )

        content.append(
            Spacer(
                1,
                10
            )
        )

        try:

            content.append(
                Image(
                    image_path,
                    width=250,
                    height=250
                )
            )

        except Exception as image_error:

            print(
                "PDF image error:",
                image_error
            )


        content.append(
            Spacer(
                1,
                20
            )
        )


    # ========================================================
    # HEATMAP
    # ========================================================

    if (
        heatmap_path
        and
        os.path.exists(heatmap_path)
    ):

        content.append(
            Paragraph(
                "Grad-CAM Heatmap",
                styles["Heading2"]
            )
        )

        content.append(
            Spacer(
                1,
                10
            )
        )

        try:

            content.append(
                Image(
                    heatmap_path,
                    width=250,
                    height=250
                )
            )

        except Exception as heatmap_error:

            print(
                "PDF heatmap error:",
                heatmap_error
            )


        content.append(
            Spacer(
                1,
                20
            )
        )


    # ========================================================
    # CONCLUSION
    # ========================================================

    content.append(
        Paragraph(
            "Investigation Conclusion",
            styles["Heading2"]
        )
    )

    content.append(
        Spacer(
            1,
            10
        )
    )


    if str(result).lower() == "fake":

        conclusion = (
            "The analysis indicates that the submitted "
            "media contains characteristics associated "
            "with potentially manipulated or synthetic "
            "content. Further forensic verification is "
            "recommended."
        )

    elif str(result).lower() == "real":

        conclusion = (
            "The analysis indicates that the submitted "
            "media appears consistent with authentic "
            "content. The result should still be considered "
            "an automated assessment."
        )

    else:

        conclusion = (
            "The analysis produced an uncertain result. "
            "The available evidence was not sufficiently "
            "strong to confidently classify the media as "
            "real or fake."
        )


    content.append(
        Paragraph(
            conclusion,
            styles["Normal"]
        )
    )


    content.append(
        Spacer(
            1,
            30
        )
    )


    # ========================================================
    # FOOTER
    # ========================================================

    content.append(
        Paragraph(
            "This report was automatically generated by "
            "the AI Deepfake Investigation System.",
            styles["Italic"]
        )
    )


    # ========================================================
    # BUILD
    # ========================================================

    doc.build(
        content
    )


    print(
        "PDF Report Saved:",
        report_path
    )


    return report_name