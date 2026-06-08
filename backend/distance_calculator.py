import math

import pandas as pd

def extract_object_centers(
    results,
    yolo_model
):

    detected_objects = []

    for box in results[0].boxes:

        x1, y1, x2, y2 = (
            box.xyxy[0]
            .cpu()
            .numpy()
        )

        center_x = int(
            (x1 + x2) / 2
        )

        center_y = int(
            (y1 + y2) / 2
        )

        class_id = int(
            box.cls[0]
        )

        class_name = (
            yolo_model.names[class_id]
        )

        confidence = float(
            box.conf[0]
        )

        detected_objects.append({

            "class": class_name,

            "confidence": confidence,

            "center_x": center_x,

            "center_y": center_y
        })

    return detected_objects

def scale_coordinates(
    detected_objects,
    original_width,
    original_height
):

    for obj in detected_objects:

        obj["depth_x"] = int(
            obj["center_x"]
            * 512
            / original_width
        )

        obj["depth_y"] = int(
            obj["center_y"]
            * 256
            / original_height
        )

    return detected_objects


def attach_depth_values(
    detected_objects,
    depth_map
):

    for obj in detected_objects:

        x = obj["depth_x"]

        y = obj["depth_y"]

        obj["depth"] = float(
            depth_map[y, x]
        )

    return detected_objects


def create_3d_coordinates(
    detected_objects
):

    for obj in detected_objects:

        obj["X"] = float(
            obj["depth_x"]
        )

        obj["Y"] = float(
            obj["depth_y"]
        )

        obj["Z"] = float(
            obj["depth"]
        )

    return detected_objects


def calculate_distance(
    obj1,
    obj2
):

    return math.sqrt(

        (obj2["X"] - obj1["X"])**2 +

        (obj2["Y"] - obj1["Y"])**2 +

        (obj2["Z"] - obj1["Z"])**2
    )


def generate_distance_table(
    detected_objects
):

    results = []

    for i in range(
        len(detected_objects)
    ):

        for j in range(
            i + 1,
            len(detected_objects)
        ):

            distance = (
                calculate_distance(
                    detected_objects[i],
                    detected_objects[j]
                )
            )
            results.append({

                "Object_1":
                detected_objects[i]["class"],

                "Object_2":
                detected_objects[j]["class"],

                "X1":
                detected_objects[i]["center_x"],

                "Y1":
                detected_objects[i]["center_y"],

                "X2":
                detected_objects[j]["center_x"],

                "Y2":
                detected_objects[j]["center_y"],

                "Distance":
                distance
            })

    return pd.DataFrame(
        results
    )


def get_high_risk_pairs(
    distance_df,
    top_n=10
):

    return (

        distance_df

        .sort_values(
            "Distance"
        )

        .head(top_n)
    )


def get_object_pair_info(
    detected_objects,
    high_risk_pairs
):

    pair_info = []

    for _, row in high_risk_pairs.iterrows():

        pair_info.append({

            "x1": row["X1"],
            "y1": row["Y1"],

            "x2": row["X2"],
            "y2": row["Y2"],

            "distance": round(
                row["Distance"],
                2
            )
        })

    return pair_info