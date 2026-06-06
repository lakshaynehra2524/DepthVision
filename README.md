# DepthVision

DepthVision is an AI-powered computer vision system that combines Monocular Depth Estimation and YOLO Object Detection to analyze scenes and identify high-risk object pairs based on their relative distances.

The system predicts scene depth from a single image, detects objects, generates 3D coordinates, calculates distances between detected objects, and visualizes potential risk zones through an interactive Streamlit dashboard.

---

## Features

* Monocular Depth Estimation
* YOLOv8 Object Detection
* 3D Coordinate Generation
* Object-to-Object Distance Calculation
* High Risk Pair Detection
* Interactive Streamlit Dashboard
* Annotated Visualization Output

---

## Project Workflow

Input Image

↓

Depth Prediction

↓

Object Detection

↓

3D Coordinate Generation

↓

Distance Calculation

↓

High Risk Pair Analysis

↓

Visualization Dashboard

---

## Technologies Used

* Python
* PyTorch
* YOLOv8
* OpenCV
* NumPy
* Pandas
* Streamlit

---

## Folder Structure

DepthVision/

├── app.py

├── backend/

│ ├── predictor.py

│ ├── detector.py

│ ├── distance_calculator.py

│ ├── model.py

│ └── config.py

├── models/

│ ├── best_depthvision_model.pth

│ └── yolov8n.pt

├── dataset/

├── outputs/

├── notebooks/

└── requirements.txt

---

## Installation

pip install -r requirements.txt

---

## Run Application

streamlit run app.py

---

## Output

* Detected Objects Table
* High Risk Pairs Table
* Annotated Image Visualization
* Relative Distance Analysis

---

## Future Improvements

* Real-world distance calibration
* Video stream support
* Real-time camera integration
* Collision warning system
* Autonomous driving applications
