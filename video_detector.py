# ============================================================
# video_detector.py
# DeepFake Investigation System
# ============================================================

import cv2
import os
import sys
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="backslashreplace")

from predictor import predict_image


# ============================================================
# CONFIGURATION
# ============================================================

SAMPLE_EVERY = 10

TEMP_FOLDER = "uploads"

os.makedirs(
    TEMP_FOLDER,
    exist_ok=True
)


# ============================================================
# EMPTY RESULT
# ============================================================

def empty_result(
    prediction="Unknown",
    confidence=0.0,
    risk="Unknown"
):

    return (
        prediction,
        round(float(confidence), 2),
        risk,

        0,
        0,
        0,

        0.0,
        0.0,
        0.0,

        0.0,
        0.0
    )


# ============================================================
# VIDEO DETECTOR
# ============================================================

def detect_video(video_path):

    print()
    print("=" * 70)
    print("VIDEO ANALYSIS STARTED")
    print("Video:", video_path)
    print("=" * 70)


    # ========================================================
    # CHECK FILE
    # ========================================================

    if not os.path.exists(video_path):

        print(
            "ERROR: Video file does not exist."
        )

        return empty_result()


    # ========================================================
    # OPEN VIDEO
    # ========================================================

    cap = cv2.VideoCapture(
        video_path
    )


    if not cap.isOpened():

        print(
            "ERROR: Could not open video."
        )

        return empty_result()


    # ========================================================
    # STORAGE
    # ========================================================

    predictions = []

    fake_probabilities = []

    real_probabilities = []

    confidences = []


    fake_count = 0

    real_count = 0

    uncertain_count = 0


    frame_count = 0


    # ========================================================
    # READ VIDEO
    # ========================================================

    while True:

        ret, frame = cap.read()


        if not ret:
            break


        frame_count += 1


        # ====================================================
        # ANALYZE EVERY 10TH FRAME
        # ====================================================

        if frame_count % SAMPLE_EVERY != 0:

            continue


        # ====================================================
        # TEMPORARY FRAME
        # ====================================================

        temp_frame = os.path.join(

            TEMP_FOLDER,

            f"_video_frame_{frame_count}.jpg"
        )


        success = cv2.imwrite(

            temp_frame,

            frame
        )


        if not success:

            print(
                "Could not save frame:",
                frame_count
            )

            continue


        try:

            # =================================================
            # IMAGE PREDICTION
            # =================================================

            result = predict_image(
                temp_frame
            )


            # =================================================
            # EXPECTED PREDICTOR FORMAT
            #
            # (
            #     prediction,
            #     confidence,
            #     risk,
            #     fake_probability,
            #     real_probability
            # )
            # =================================================

            if isinstance(
                result,
                (tuple, list)
            ):

                # ---------------------------------------------
                # 5 VALUES
                # ---------------------------------------------

                if len(result) >= 5:

                    prediction = str(
                        result[0]
                    )

                    confidence = float(
                        result[1]
                    )

                    risk = str(
                        result[2]
                    )

                    fake_probability = float(
                        result[3]
                    )

                    real_probability = float(
                        result[4]
                    )


                # ---------------------------------------------
                # 3 VALUES
                # ---------------------------------------------

                elif len(result) == 3:

                    prediction = str(
                        result[0]
                    )

                    confidence = float(
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


                # ---------------------------------------------
                # 2 VALUES
                # ---------------------------------------------

                elif len(result) == 2:

                    prediction = str(
                        result[0]
                    )

                    confidence = float(
                        result[1]
                    )

                    risk = "Medium"


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


                # ---------------------------------------------
                # 1 VALUE
                # ---------------------------------------------

                elif len(result) == 1:

                    prediction = str(
                        result[0]
                    )

                    confidence = 50.0

                    risk = "Medium"

                    fake_probability = 50.0

                    real_probability = 50.0


                else:

                    raise ValueError(
                        "Empty predictor result."
                    )


            else:

                prediction = str(
                    result
                )

                confidence = 50.0

                risk = "Medium"

                fake_probability = 50.0

                real_probability = 50.0


            # =================================================
            # LIMIT VALUES
            # =================================================

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


            # =================================================
            # NORMALIZE
            # =================================================

            probability_total = (

                fake_probability
                +
                real_probability
            )


            if probability_total > 0:

                fake_probability = (

                    fake_probability
                    /
                    probability_total
                    *
                    100.0
                )


                real_probability = (

                    real_probability
                    /
                    probability_total
                    *
                    100.0
                )


            # =================================================
            # SAVE RESULTS
            # =================================================

            predictions.append(
                prediction
            )


            fake_probabilities.append(
                fake_probability
            )


            real_probabilities.append(
                real_probability
            )


            confidences.append(
                confidence
            )


            # =================================================
            # COUNT PREDICTION
            # =================================================

            prediction_lower = (
                prediction.lower()
            )


            if "fake" in prediction_lower:

                fake_count += 1


            elif "real" in prediction_lower:

                real_count += 1


            else:

                uncertain_count += 1


            # =================================================
            # PRINT FRAME RESULT
            # =================================================

            print(
                "======================"
            )

            print(
                "File:",
                temp_frame
            )

            print(
                "Prediction:",
                prediction
            )

            print(
                "Confidence:",
                round(
                    confidence,
                    2
                ),
                "%"
            )

            print(
                "Fake Probability:",
                round(
                    fake_probability,
                    2
                ),
                "%"
            )

            print(
                "Real Probability:",
                round(
                    real_probability,
                    2
                ),
                "%"
            )

            print(
                "Risk:",
                risk
            )

            print(
                "======================"
            )


        except Exception as e:

            print(
                "Frame prediction error:",
                e
            )


        finally:

            # ===============================================
            # DELETE TEMPORARY FRAME
            # ===============================================

            try:

                if os.path.exists(
                    temp_frame
                ):

                    os.remove(
                        temp_frame
                    )

            except Exception:

                pass


    # ========================================================
    # RELEASE VIDEO
    # ========================================================

    cap.release()


    # ========================================================
    # CHECK RESULTS
    # ========================================================

    analyzed_frames = len(
        predictions
    )


    if analyzed_frames == 0:

        print(
            "ERROR: No frames were analyzed."
        )

        return empty_result()


    # ========================================================
    # PERCENTAGES
    # ========================================================

    fake_percentage = (

        fake_count
        /
        analyzed_frames
        *
        100.0
    )


    real_percentage = (

        real_count
        /
        analyzed_frames
        *
        100.0
    )


    uncertain_percentage = (

        uncertain_count
        /
        analyzed_frames
        *
        100.0
    )


    # ========================================================
    # AVERAGE PROBABILITIES
    # ========================================================

    average_fake_probability = float(

        np.mean(
            fake_probabilities
        )
    )


    average_real_probability = float(

        np.mean(
            real_probabilities
        )
    )


    # ========================================================
    # AVERAGE CONFIDENCE
    # ========================================================

    average_confidence = float(

        np.mean(
            confidences
        )
    )


    # Top-K / anomaly forensic aggregation
    sorted_fake_probs = sorted(fake_probabilities, reverse=True)
    top_k = min(5, len(sorted_fake_probs))
    top_fake_avg = float(np.mean(sorted_fake_probs[:top_k])) if top_k > 0 else 0.0

    if fake_count > real_count or fake_percentage >= 18.0 or (fake_count >= 3 and top_fake_avg >= 68.0):
        final_prediction = "Fake"
        final_confidence = max(average_fake_probability, top_fake_avg)
    elif real_count > fake_count:
        final_prediction = "Real"
        final_confidence = average_real_probability
    else:
        if average_fake_probability > average_real_probability:
            final_prediction = "Fake"
            final_confidence = average_fake_probability
        elif average_real_probability > average_fake_probability:
            final_prediction = "Real"
            final_confidence = average_real_probability
        else:
            final_prediction = "Uncertain"
            final_confidence = average_confidence


    # ========================================================
    # UNCERTAINTY CHECK
    # ========================================================

    probability_difference = abs(

        average_fake_probability
        -
        average_real_probability
    )


    if probability_difference < 5:

        final_prediction = "Uncertain"

        final_confidence = max(

            average_fake_probability,

            average_real_probability
        )


    # ========================================================
    # RISK LEVEL
    # ========================================================

    if final_prediction == "Fake":

        if (

            fake_percentage >= 70

            and

            average_fake_probability >= 80
        ):

            risk_level = "Very High"


        elif (

            fake_percentage >= 50

            or

            average_fake_probability >= 70
        ):

            risk_level = "High"


        elif (

            fake_percentage >= 30

            or

            average_fake_probability >= 60
        ):

            risk_level = "Medium"


        else:

            risk_level = "Low"


    elif final_prediction == "Real":

        if average_real_probability >= 60:

            risk_level = "Low"

        else:

            risk_level = "Medium"


    else:

        risk_level = "Medium"


    # ========================================================
    # ROUND VALUES
    # ========================================================

    final_confidence = round(
        final_confidence,
        2
    )


    fake_percentage = round(
        fake_percentage,
        2
    )


    real_percentage = round(
        real_percentage,
        2
    )


    uncertain_percentage = round(
        uncertain_percentage,
        2
    )


    average_fake_probability = round(
        average_fake_probability,
        2
    )


    average_real_probability = round(
        average_real_probability,
        2
    )


    # ========================================================
    # FINAL TERMINAL OUTPUT
    # ========================================================

    print()
    print("=" * 70)

    print(
        "FINAL VIDEO RESULT"
    )

    print("=" * 70)


    print(
        "Prediction:",
        final_prediction
    )


    print(
        "Confidence:",
        final_confidence,
        "%"
    )


    print(
        "Fake Frames:",
        fake_count
    )


    print(
        "Real Frames:",
        real_count
    )


    print(
        "Uncertain Frames:",
        uncertain_count
    )


    print(
        "Analyzed Frames:",
        analyzed_frames
    )


    print(
        "Fake Percentage:",
        fake_percentage,
        "%"
    )


    print(
        "Real Percentage:",
        real_percentage,
        "%"
    )


    print(
        "Uncertain Percentage:",
        uncertain_percentage,
        "%"
    )


    print(
        "Average Fake Probability:",
        average_fake_probability,
        "%"
    )


    print(
        "Average Real Probability:",
        average_real_probability,
        "%"
    )


    print(
        "Risk:",
        risk_level
    )


    print("=" * 70)


    # ========================================================
    # IMPORTANT:
    # RETURN EXACTLY 11 VALUES
    # ========================================================

    return (

        final_prediction,

        final_confidence,

        risk_level,

        fake_count,

        real_count,

        uncertain_count,

        fake_percentage,

        real_percentage,

        uncertain_percentage,

        average_fake_probability,

        average_real_probability
    )