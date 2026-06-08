from ultralytics import YOLO

from backend.config import (
    YOLO_MODEL_PATH
)

def load_yolo_model():

    model = YOLO(
        YOLO_MODEL_PATH
    )

    return model

def detect_objects(
    image_path
):

    model = load_yolo_model()

    results = model.predict(
        image_path,
        conf=0.3,
        verbose=False
    )

    return results


def get_bounding_boxes(
    results,
    yolo_model
):

    boxes_data = []

    for box in results[0].boxes:

        x1, y1, x2, y2 = (
            box.xyxy[0]
            .cpu()
            .numpy()
        )

        class_id = int(
            box.cls[0]
        )

        class_name = (
            yolo_model.names[class_id]
        )

        boxes_data.append({

            "x1": int(x1),
            "y1": int(y1),

            "x2": int(x2),
            "y2": int(y2),

            "class": class_name
        })

    return boxes_data