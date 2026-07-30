# DepthVision

DepthVision is an AI-powered computer vision web application that combines Monocular Depth Estimation and YOLOv8 Object Detection to analyze scenes and identify high-risk object pairs based on their relative spatial distances.



---

## Features

* Monocular Depth Estimation using a custom CNN architecture
* YOLOv8 Object Detection
* 3D Coordinate Generation
* Object-to-Object Distance Calculation
* High Risk Pair Detection
* Annotated Image Visualization
* Interactive Web Interface
* FastAPI REST API
* Documentation & About Pages
* Browser-Based Image Upload and Analysis

---

## System Workflow

Input Image

↓

Depth Estimation Model

↓

YOLOv8 Object Detection

↓

3D Coordinate Generation

↓

Distance Calculation

↓

Risk Analysis

↓

Annotated Visualization Generation

↓

Web Interface Display

---

## Technologies Used

### Backend

* Python
* FastAPI
* PyTorch
* YOLOv8 (Ultralytics)
* OpenCV
* NumPy
* Pandas

### Frontend

* HTML5
* CSS3
* JavaScript

### Machine Learning

* Custom Encoder-Decoder CNN
* Monocular Depth Estimation
* Object Detection
* Distance Analytics

---

## Project Architecture

User

↓

Web Browser

↓

HTML / CSS / JavaScript Frontend

↓

FastAPI Backend

↓

DepthVision CNN + YOLOv8

↓

Distance & Risk Analysis Engine

↓

Annotated Results

---

## Folder Structure

DepthVision/

├── backend/

│   ├── config.py

│   ├── model.py

│   ├── predictor.py

│   ├── detector.py

│   └── distance_calculator.py

│

├── frontend/

│   ├── index.html

│   ├── about.html

│   ├── docs.html

│   │

│   ├── css/

│   │   └── style.css

│   │

│   └── js/

│       └── app.js

│

├── models/

│   ├── best_depthvision_model.pth

│   └── yolov8n.pt

│

├── notebooks/

│

├── main.py

│

├── requirements.txt

│

└── README.md

---

## Installation

Clone the repository:

```bash
git clone <repository-url>
cd DepthVision
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Run Application

Start the FastAPI server:

```bash
uvicorn main:app --reload
```

Open your browser:

```text
http://127.0.0.1:8000
```

API Documentation:

```text
http://127.0.0.1:8000/docs
```

---

## Application Pages

### Home

* Upload image
* Run analysis
* View annotated results
* View detected objects
* View high-risk pairs

### About

Provides project overview, architecture, and technology stack.

### Documentation

Provides workflow explanation, supported formats, outputs, and limitations.

### API Documentation

Interactive Swagger UI generated automatically by FastAPI.

---

## Output

### Annotated Image

* Bounding Boxes
* Object Labels
* High Risk Connection Lines
* Distance Labels

### Detected Objects Table

* Object Class
* Confidence Score
* Estimated Depth

### High Risk Pairs Table

* Object Pair
* Relative Distance
* Risk Analysis

---

## Future Improvements

* Real-World Distance Calibration
* Video Stream Analysis
* Live Camera Support
* Multi-Object Tracking
* Collision Warning System
* Autonomous Navigation Applications
* Docker Deployment
* AWS Cloud Deployment

---

## Developer

Lakshay Nehra

Computer Vision | Artificial Intelligence | Machine Learning | FastAPI
