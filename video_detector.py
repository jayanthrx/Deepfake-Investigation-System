# video_detector.py

import cv2
import os
import numpy as np

from predictor import predict_image


def detect_video(video_path):
    """
    Analyze a video by sampling frames.

    Returns exactly 3 values:
        prediction, confidence, risk_level
    """

    if not os.path.exists(video_path):
        return "Error", 0.0, "Unknown"

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return "Error", 0.0, "Unknown"

    predictions = []

    frame_count = 0
    sample_every = 10

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        frame_count += 1

        # Analyze every 10th frame
        if frame_count % sample_every != 0:
            continue

        # Save temporary frame
        temp_frame = os.path.join(
            "uploads",
            f"_video_frame_{frame_count}.jpg"
        )

        cv2.imwrite(temp_frame, frame)

        try:
            result = predict_image(temp_frame)

            # predictor.py may return:
            # prediction, confidence, risk
            if isinstance(result, tuple):

                if len(result) >= 3:
                    prediction = result[0]
                    confidence = float(result[1])
                    risk = result[2]

                elif len(result) == 2:
                    prediction = result[0]
                    confidence = float(result[1])
                    risk = "Medium"

                else:
                    prediction = result[0]
                    confidence = 50.0
                    risk = "Medium"

            else:
                prediction = str(result)
                confidence = 50.0
                risk = "Medium"

            predictions.append(
                (
                    str(prediction),
                    confidence,
                    str(risk)
                )
            )

        except Exception as e:
            print("Frame prediction error:", e)

        # Delete temporary frame
        try:
            if os.path.exists(temp_frame):
                os.remove(temp_frame)
        except:
            pass

    cap.release()

    # No frames analyzed
    if len(predictions) == 0:
        return "Unknown", 0.0, "Unknown"

    # ------------------------------------------------
    # Calculate final video result
    # ------------------------------------------------

    fake_count = 0
    real_count = 0

    confidences = []

    for prediction, confidence, risk in predictions:

        prediction_lower = prediction.lower()

        if "fake" in prediction_lower:
            fake_count += 1
        elif "real" in prediction_lower:
            real_count += 1

        confidences.append(confidence)

    average_confidence = float(
        np.mean(confidences)
    )

    total_frames = fake_count + real_count

    if total_frames == 0:
        return "Unknown", average_confidence, "Medium"

    fake_percentage = (
        fake_count / total_frames
    ) * 100

    real_percentage = (
        real_count / total_frames
    ) * 100

    # ------------------------------------------------
    # Final classification
    # ------------------------------------------------

    if fake_percentage >= 60:

        final_prediction = "Fake"

        if fake_percentage >= 80:
            risk_level = "High"
        else:
            risk_level = "Medium"

    elif real_percentage >= 60:

        final_prediction = "Real"

        if real_percentage >= 80:
            risk_level = "Low"
        else:
            risk_level = "Medium"

    else:

        final_prediction = "Uncertain"
        risk_level = "Medium"

    return (
        final_prediction,
        round(average_confidence, 2),
        risk_level
    )