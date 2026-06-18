from fastapi import FastAPI
from fastapi import UploadFile
from fastapi import File
import cv2
import base64
import numpy as np

import tempfile

from PIL import Image

from backend.predictor import predict_depth

from backend.detector import (
    detect_objects,
    load_yolo_model,
    get_bounding_boxes
)

from backend.distance_calculator import (
    extract_object_centers,
    scale_coordinates,
    attach_depth_values,
    create_3d_coordinates,
    generate_distance_table,
    get_high_risk_pairs,
    get_object_pair_info
)
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles


app = FastAPI(
    title="DepthVision API",
    description="Monocular Depth Estimation + Object Distance Analysis",
    version="1.0"
)

app.mount(
    "/frontend",
    StaticFiles(directory="frontend"),
    name="frontend"
)


@app.get("/")
def home():
    return FileResponse("frontend/index.html")

@app.get("/about")
def about():
    return FileResponse("frontend/about.html")

@app.get("/docs-info")
def docs_info():
    return FileResponse("frontend/docs.html")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False,suffix=".jpg") as temp_file:
        temp_file.write(await file.read())
        image_path = temp_file.name

    depth_map = predict_depth(image_path)

    results = detect_objects(image_path)

    yolo_model = load_yolo_model()

    detected_objects = (extract_object_centers(results,yolo_model))
    boxes_data = get_bounding_boxes(results,yolo_model)

    image = Image.open(image_path)

    original_width, original_height = (image.size)
    detected_objects = (scale_coordinates(
            detected_objects,
            original_width,
            original_height))

    detected_objects = (attach_depth_values(
            detected_objects,
            depth_map))

    detected_objects = (create_3d_coordinates(detected_objects))

    distance_df = (generate_distance_table(detected_objects))

    if distance_df.empty:
        high_risk_pairs = []
        pair_info = []
    else:
        high_risk_pairs = (get_high_risk_pairs(distance_df,top_n=5))

        pair_info = get_object_pair_info(detected_objects,high_risk_pairs)

        high_risk_pairs = (high_risk_pairs.to_dict(orient="records"))

    image_bgr = cv2.imread(image_path)
    annotated_image = image_bgr.copy()

    # Draw Bounding Boxes
    for box in boxes_data:
        cv2.rectangle(
            annotated_image,
            (box["x1"], box["y1"]),
            (box["x2"], box["y2"]),
            (0,255,0),
            2
        )

        cv2.putText(
            annotated_image,
            box["class"],
            (box["x1"], box["y1"] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0,255,0),
            2
        )


    # Draw Risk Lines

    for pair in pair_info:

        cv2.line(
            annotated_image,
            (pair["x1"], pair["y1"]),
            (pair["x2"], pair["y2"]),
            (0,0,255),
            4
        )

        mid_x = int(
            (pair["x1"] + pair["x2"]) / 2
        )

        mid_y = int(
            (pair["y1"] + pair["y2"]) / 2
        )

        cv2.putText(
            annotated_image,
            str(pair["distance"]),
            (mid_x, mid_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0,0,255),
            3
        )


    _, buffer = cv2.imencode(
        ".jpg",
        annotated_image
    )

    image_base64 = base64.b64encode(
        buffer
    ).decode("utf-8")


    display_objects = []

    for obj in detected_objects:

        display_objects.append({

            "class":
            obj["class"],

            "confidence":
            round(
                obj["confidence"],
                2
            ),

            "depth":
            round(
                obj["depth"],
                4
            )
        })


    return {

        "annotated_image":
        image_base64,

        "objects":
        display_objects,

        "high_risk_pairs":
        high_risk_pairs
    }