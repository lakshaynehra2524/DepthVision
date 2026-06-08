import streamlit as st
import tempfile

from PIL import Image
import cv2
import numpy as np
import pandas as pd

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

# Background
import base64
def set_background():

    image_path = "background_img/Img_01.png"
        
    with open(image_path , "rb") as image_file:

        encoded = base64.b64encode(
            image_file.read()
        ).decode()

    st.markdown(
        f"""
        <style>

        .stApp {{
            background-image: url("data:image/png;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}

        </style>
        """,
        unsafe_allow_html=True
    )

st.set_page_config(
    page_title="DepthVision",
    page_icon="🚨",
    layout="wide"
)

set_background()

st.markdown(
    """
    <h1 style='color:#8B5CF6;'>
        DepthVision
    </h1>
    """,
    unsafe_allow_html=True
)

st.subheader(
    "AI-Based Object Distance & Risk Detection"
)

left_col, center_col, right_col = st.columns(
    [1, 4, 1.5]
)

with left_col:

    st.header("Upload Image")

    uploaded_file = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png"]
    )

    run_analysis = st.button(
        "Run Analysis"
    )



with right_col:

    st.header(
        "High Risk Pairs"
    )

st.divider()

st.header(
    "All Detected Objects"
)

if uploaded_file is not None and run_analysis:

    with st.spinner(
        "Running Analysis..."
    ):

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".jpg"
        ) as temp_file:

            temp_file.write(
                uploaded_file.read()
            )

            image_path = temp_file.name

        depth_map = predict_depth(
            image_path
        )

        results = detect_objects(
            image_path
        )

        yolo_model = load_yolo_model()
        boxes_data = get_bounding_boxes(
            results,
            yolo_model
        )

        detected_objects = extract_object_centers(
            results,
            yolo_model
        )

        image = Image.open(
            image_path
        )

        original_width, original_height = image.size

        detected_objects = scale_coordinates(
            detected_objects,
            original_width,
            original_height
        )

        detected_objects = attach_depth_values(
            detected_objects,
            depth_map
        )

        detected_objects = create_3d_coordinates(
            detected_objects
        )

        distance_df = generate_distance_table(
            detected_objects
        )

        high_risk_pairs = get_high_risk_pairs(
            distance_df,
            top_n=5
        )
        pair_info = get_object_pair_info(
            detected_objects,
            high_risk_pairs
        )

        image_bgr = cv2.imread(
            image_path
        )
        annotated_image = image_bgr.copy()
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

        annotated_image = cv2.cvtColor(
            annotated_image,
            cv2.COLOR_BGR2RGB
        )

        with center_col:

            st.header(
                "High Risk Pairs Visualization"
            )

            st.image(
                annotated_image,
                width=1200
            )
        with right_col:

            st.subheader(
                "Top High Risk Pairs"
            )

            display_pairs = high_risk_pairs[
                ["Object_1", "Object_2", "Distance"]
            ].copy()

            display_pairs["Distance"] = (
                display_pairs["Distance"]
                .round(1)
            )

            display_pairs["Risk"] = display_pairs[
                "Distance"
            ].apply(
                lambda x:
                "🔴 High" if x < 10 else
                "🟠 Medium" if x < 25 else
                "🟢 Low"
            )

            st.dataframe(
                display_pairs,
                width=1200
            )

        display_objects = pd.DataFrame(
            detected_objects
        )

        display_objects = display_objects[
            [
                "class",
                "confidence",
                "depth"
            ]
        ]

        display_objects["confidence"] = (
            display_objects["confidence"]
            .round(2)
        )

        display_objects["depth"] = (
            display_objects["depth"]
            .round(4)
        )

        st.dataframe(
            display_objects,
            width=1200
        )

        st.success(
            f"Analysis Complete | Objects Detected: {len(detected_objects)}"
        )


