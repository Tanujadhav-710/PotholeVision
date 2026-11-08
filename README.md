# PotholeVision – Road Surface Monitoring System

PotholeVision is a web-based road surface monitoring application that uses YOLO26n object detection to identify potholes in road images and videos. It provides an interface for uploading media, viewing detected potholes, and reviewing detection statistics.

## Project Overview

The application integrates a pretrained YOLO26n model with a Flask web application. It processes uploaded images and videos, visualizes detected potholes using bounding boxes, and generates annotated output files.

## Features

- **Image Detection:** Upload road images and detect potholes.
- **Video Detection:** Process uploaded videos and visualize detected potholes.
- **Bounding-Box Visualization:** Display detected potholes with confidence scores.
- **Detection Statistics:** Display detection counts and processing history.
- **Dashboard:** View image and video processing statistics.
- **Video Optimization:** Run inference on every third frame to reduce processing time while maintaining the original video playback duration.
- **Video Conversion:** Use FFmpeg to convert processed videos into H.264 format for browser compatibility.

## Technologies Used

| Component | Technology |
|---|---|
| Programming Language | Python |
| Backend | Flask |
| Object Detection | YOLO26n |
| Model Framework | Ultralytics |
| Image Processing | OpenCV |
| Frontend | HTML, CSS |
| Video Processing | FFmpeg |
| Data Storage | JSON |

## Project Structure

```text
PotholeVision/
│
├── app.py
├── requirements.txt
├── .gitignore
│
├── models/
│   └── best.pt
│
├── templates/
│   └── index.html
│
├── static/
├── uploads/
├── outputs/
└── history.json
```

*Note: Generated folders and model weights may be excluded from the public repository.*

## How It Works

1. The user uploads an image or video through the web interface.
2. Flask receives and validates the uploaded file.
3. The YOLO26n model performs pothole detection.
4. Bounding boxes and confidence scores are generated.
5. For video processing, inference is performed on every third frame to reduce computational workload.
6. OpenCV generates the annotated output.
7. FFmpeg converts the processed video into H.264 format.
8. Detection results and statistics are displayed on the dashboard.

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/Tanujadhav-710/PotholeVision.git
cd PotholeVision
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure the YOLO model

Place the authorized YOLO26n model weights at:

```text
models/best.pt
```

The model weights are not included in this repository. Their source and redistribution permissions must be verified before sharing them.

### 5. Install FFmpeg

Install FFmpeg and configure its executable path in `app.py` to match your system.

### 6. Run the application

```bash
python app.py
```

Open the application at:

```text
http://127.0.0.1:5000
```

## My Contributions

My primary contribution was developing the Flask-based web application around an existing trained YOLO26n model.

- Developed the Flask backend and application routes.
- Integrated YOLO26n model inference.
- Implemented image and video upload functionality.
- Modified the detection and video-processing pipeline.
- Implemented bounding-box visualization and detection-result generation.
- Optimized video inference by processing every third frame.
- Integrated FFmpeg for H.264 video conversion.
- Developed the HTML/CSS interface and dashboard.
- Implemented JSON-based detection history and statistics.

**Model development clarification:** The YOLO26n architecture and pretrained model were not developed from scratch as part of my contribution. The work focused on application development and model integration.

## Future Enhancements

- Real-time pothole detection using live camera feeds.
- GPS-based pothole location mapping.
- Pothole severity classification.
- Road condition monitoring and reporting.
- Integration with smart transportation systems.

## Project Status

Developed as an academic project to explore computer vision, web application development, and automated road surface monitoring.
