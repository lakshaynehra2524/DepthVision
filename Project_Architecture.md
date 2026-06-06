# DepthVision Architecture

## System Overview

DepthVision combines Deep Learning and Computer Vision techniques to estimate depth and calculate distances between detected objects from a single image.

---

## Architecture Flow

Input Image

↓

Depth Estimation Model

↓

Depth Map

↓

YOLO Object Detection

↓

Object Centers Extraction

↓

Depth Assignment

↓

3D Coordinate Generation

↓

Distance Calculation

↓

Risk Analysis

↓

Visualization Dashboard

---

## Module Description

### predictor.py

Responsible for generating depth maps using the trained DepthVision model.

---

### detector.py

Loads YOLOv8 and performs object detection.

Outputs:

* Bounding Boxes
* Class Labels
* Confidence Scores

---

### distance_calculator.py

Handles:

* Center Extraction
* Coordinate Scaling
* Depth Assignment
* 3D Coordinate Creation
* Distance Calculation
* High Risk Pair Detection

---

### app.py

Main Streamlit application.

Responsible for:

* User Interface
* Image Upload
* Visualization
* Results Display

---

## Model Inputs

RGB Image

---

## Model Outputs

Depth Map

Detected Objects

Object Distances

High Risk Pairs

Annotated Visualization

---

## Risk Analysis Logic

Objects with the smallest relative distances are classified as High Risk Pairs.

The nearest object pairs are highlighted on the visualization output and displayed in the dashboard.
