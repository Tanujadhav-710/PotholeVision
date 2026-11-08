
from flask import Flask, render_template, request, send_from_directory
from ultralytics import YOLO
from werkzeug.utils import secure_filename
from datetime import datetime
import os
import cv2
import uuid
import json
import subprocess

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "outputs"
MODEL_PATH = "models/best.pt"
HISTORY_FILE = "history.json"

app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

model = YOLO(MODEL_PATH)

IMAGE_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}
VIDEO_EXTENSIONS = {"mp4", "avi", "mov"}

FFMPEG_EXE = "ffmpeg"


# --------------------------------
# HISTORY FUNCTIONS
# --------------------------------

def load_history():

    if not os.path.exists(HISTORY_FILE):
        return {
            "images_processed": 0,
            "videos_processed": 0,
            "total_detections": 0,
            "confidence_values": [],
            "history": []
        }

    try:
        with open(HISTORY_FILE, "r") as file:
            data = json.load(file)

        return data

    except (json.JSONDecodeError, OSError):
        return {
            "images_processed": 0,
            "videos_processed": 0,
            "total_detections": 0,
            "confidence_values": [],
            "history": []
        }


def save_history(data):

    temp_file = HISTORY_FILE + ".tmp"

    with open(temp_file, "w") as file:
        json.dump(data, file, indent=4)

    os.replace(temp_file, HISTORY_FILE)


def record_detection(media_type, filename, output_name,
                     count, confidences):

    data = load_history()

    if media_type == "image":
        data["images_processed"] += 1
    else:
        data["videos_processed"] += 1

    data["total_detections"] += count

    data["confidence_values"].extend(confidences)

    average_confidence = (
        sum(confidences) / len(confidences)
        if confidences else 0
    )

    entry = {
        "filename": filename,
        "output_file": output_name,
        "type": media_type,
        "detections": count,
        "average_confidence": round(average_confidence, 2),
        "date": datetime.now().strftime("%d-%m-%Y %I:%M %p")
    }

    data["history"].insert(0, entry)

    # Keep the latest 50 records
    data["history"] = data["history"][:50]

    save_history(data)


def get_dashboard_stats():

    data = load_history()

    values = data.get("confidence_values", [])

    average_confidence = (
        sum(values) / len(values)
        if values else 0
    )

    return {
        "images_processed": data.get("images_processed", 0),
        "videos_processed": data.get("videos_processed", 0),
        "total_detections": data.get("total_detections", 0),
        "average_confidence": round(average_confidence, 2),
        "history": data.get("history", [])
    }


# --------------------------------
# HOME PAGE
# --------------------------------

@app.route("/")
def home():

    stats = get_dashboard_stats()

    return render_template(
        "index.html",
        stats=stats
    )


# --------------------------------
# DETECTION ROUTE
# --------------------------------

@app.route("/detect", methods=["POST"])
def detect():

    if "file" not in request.files:
        return "No file uploaded", 400

    file = request.files["file"]

    if not file.filename:
        return "Please select a file", 400

    filename = secure_filename(file.filename)
    extension = filename.rsplit(".", 1)[-1].lower()

    if extension not in IMAGE_EXTENSIONS | VIDEO_EXTENSIONS:
        return "Unsupported file type", 400

    unique_name = uuid.uuid4().hex + "_" + filename
    input_path = os.path.join(UPLOAD_FOLDER, unique_name)

    file.save(input_path)

    # --------------------------------
    # IMAGE DETECTION
    # --------------------------------

    if extension in IMAGE_EXTENSIONS:

        results = model.predict(
            source=input_path,
            conf=0.25,
            imgsz=640,
            device="cpu",
            verbose=False
        )

        output_name = "detected_" + unique_name
        output_path = os.path.join(OUTPUT_FOLDER, output_name)

        annotated_image = results[0].plot()
        success = cv2.imwrite(output_path, annotated_image)

        if not success:
            return "Unable to save detected image", 500

        detections = []
        confidences = []

        for box in results[0].boxes:

            confidence = round(float(box.conf[0]) * 100, 2)
            confidences.append(confidence)

            detections.append({
                "class": model.names[int(box.cls[0])],
                "confidence": confidence
            })

        record_detection(
            "image",
            filename,
            output_name,
            len(detections),
            confidences
        )

        return render_template(
            "index.html",
            stats=get_dashboard_stats(),
            result_file=output_name,
            count=len(detections),
            detections=detections,
            media_type="image"
        )

    # --------------------------------
    # VIDEO DETECTION
    # --------------------------------

    elif extension in VIDEO_EXTENSIONS:

        cap = cv2.VideoCapture(input_path)

        if not cap.isOpened():
            return "Unable to read video", 400

        fps = cap.get(cv2.CAP_PROP_FPS)

        if not fps or fps <= 0:
            fps = 25

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if width <= 0 or height <= 0:
            cap.release()
            return "Invalid video dimensions", 400

        width = width - (width % 2)
        height = height - (height % 2)

        output_name = "detected_" + uuid.uuid4().hex + ".mp4"
        output_path = os.path.join(OUTPUT_FOLDER, output_name)

        writer = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height)
        )

        if not writer.isOpened():
            cap.release()
            return "Unable to create output video", 500

        total_detections = 0
        frame_number = 0
        last_annotated_frame = None
        confidences = []

        try:

            while True:

                success, frame = cap.read()

                if not success:
                    break

                frame_number += 1

                # Detect on every third frame
                if frame_number % 3 == 0 or last_annotated_frame is None:

                    result = model.predict(
                        frame,
                        conf=0.25,
                        imgsz=640,
                        device="cpu",
                        verbose=False
                    )[0]

                    total_detections += len(result.boxes)

                    for box in result.boxes:
                        confidences.append(
                            round(float(box.conf[0]) * 100, 2)
                        )

                    print(
                        "Frame:", frame_number,
                        "Detections:", len(result.boxes)
                    )

                    last_annotated_frame = result.plot()

                annotated_frame = cv2.resize(
                    last_annotated_frame,
                    (width, height)
                )

                writer.write(annotated_frame)

        finally:
            cap.release()
            writer.release()

        # --------------------------------
        # H.264 VIDEO CONVERSION
        # --------------------------------

        h264_path = output_path.replace(
            ".mp4", "_h264.mp4"
        )

        try:

            subprocess.run(
                [
                    FFMPEG_EXE,
                    "-y",
                    "-i", output_path,
                    "-c:v", "libx264",
                    "-preset", "fast",
                    "-pix_fmt", "yuv420p",
                    "-an",
                    "-movflags", "+faststart",
                    h264_path
                ],
                check=True,
                capture_output=True,
                text=True
            )

            os.replace(h264_path, output_path)

            print("H.264 conversion successful")

        except FileNotFoundError:
            return "FFmpeg executable not found. Check installation path.", 500

        except subprocess.CalledProcessError as error:
            print(error.stderr)

            if os.path.exists(h264_path):
                os.remove(h264_path)

            return "Video conversion failed", 500

        print("Total detections:", total_detections)
        print("Output size:", os.path.getsize(output_path))

        record_detection(
            "video",
            filename,
            output_name,
            total_detections,
            confidences
        )

        return render_template(
            "index.html",
            stats=get_dashboard_stats(),
            result_file=output_name,
            count=total_detections,
            detections=[],
            media_type="video"
        )

    return "Unsupported file type", 400


# --------------------------------
# OUTPUT FILE ROUTE
# --------------------------------

@app.route("/outputs/<path:filename>")
def output_file(filename):

    return send_from_directory(
        OUTPUT_FOLDER,
        filename,
        conditional=True
    )


# --------------------------------
# RUN APPLICATION
# --------------------------------

if __name__ == "__main__":
    import webbrowser
    from threading import Timer

    Timer(1.5, lambda: webbrowser.open("http://127.0.0.1:5000")).start()

    app.run(debug=True, use_reloader=False)
    

   