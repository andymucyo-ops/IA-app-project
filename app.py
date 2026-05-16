import numpy as np
import streamlit as st
import cv2 as cv
import matplotlib.pyplot as plt
from streamlit.runtime.uploaded_file_manager import UploadedFile

from metrics.metrics import compute_canny_metrics, compute_harris_metrics, compute_sift_metrics
from processing.detectors import detect_canny, detect_harris, detect_sift
from processing.matching import match_features
from utils.visualization import draw_feature_matches, draw_heatmap, draw_heatmap_overlay, draw_keypoints, draw_response_histogram, draw_sift_heatmap, draw_sift_heatmap_overlay, draw_sift_keypoints
from utils.image_io import load_from_sample, load_from_upload



def main():
    st.title("🔍 Feature Extraction Explorer")

    #initializing state variables
    if "has_run" not in st.session_state:
        st.session_state.has_run = False
    if "last_image_key" not in st.session_state:
        st.session_state.last_image_key = None
    if "last_algo" not in st.session_state:
        st.session_state.last_algo = None
    input_image = None
    image_key = None
    harris_keypoints = None
    harris_response_map = None
    canny_result = None
    sift_keypoints = None
    sift_descriptors = None
    metrics = None
    exec_time = 0.0

    #=================================================================
    # sidebar
    #=================================================================
    with st.sidebar:
        st.subheader("Input selection")
        input_selector = st.radio(
                                "Image input",
                                ("Upload file", "Sample images"),
                                label_visibility="collapsed"
                                )
        st.subheader("Detection algorithm")
        algo_selector = st.radio(
                                "Detector",
                                ("Harris", "Canny", "SIFT"),
                                label_visibility="collapsed"
                                )

        match algo_selector:
            case "Harris":
                st.subheader("Parameters")
                block_size: int = st.slider(
                        "Block size", 
                        min_value=2, 
                        max_value=10, 
                        value=2
                        )
                ksize: int = st.slider("ksize", 
                                       min_value=3, 
                                       max_value=7, 
                                       value=3, 
                                       step=2
                                       )
                k: float = st.slider(
                        "k", 
                        min_value=0.01, 
                        max_value=0.10, 
                        value=0.04
                        )
                threshold_ratio: float = st.slider(
                        "Threshold ratio", 
                        min_value=0.01, 
                        max_value=0.1, 
                        value=0.01
                        )
                radius: int = st.slider(
                        "Radius", 
                        min_value=2, 
                        max_value=10
                        )

            case "Canny":
                st.subheader("Parameters")
                canny_sigma: float = st.slider(
                        "Sigma",
                        min_value=0.5,
                        max_value=5.0,
                        value=1.0,
                        step=0.1
                        )
                low_threshold: int = st.slider(
                        "Low threshold",
                        min_value=0,
                        max_value=100,
                        value=50
                        )
                high_threshold: int = st.slider(
                        "High threshold",
                        min_value=0,
                        max_value=300,
                        value=150
                        )

            case "SIFT":
                st.subheader("Parameters")
                nfeatures: int = st.slider(
                        "nfeatures",
                        min_value=0,
                        max_value=1000,
                        value=0
                        )
                contrast_threshold: float = st.slider(
                        "Contrast threshold",
                        min_value=0.01,
                        max_value=0.1,
                        value=0.04
                        )
                edge_threshold: int = st.slider(
                        "Edge threshold",
                        min_value=5,
                        max_value=30,
                        value=10
                        )
                sift_sigma: float = st.slider(
                        "Sigma",
                        min_value=1.2,
                        max_value=2.0,
                        value=1.6
                        )
        run_button = st.button("Run", type="primary")
    
    #=================================================================
    # main area 
    #=================================================================
    if input_selector == "Upload file":
        uploaded_image: UploadedFile | None = st.file_uploader(
                "Uploaded image",
                type=["jpg", "png"]
                )
        image_key = uploaded_image.name if uploaded_image is not None else None
        if uploaded_image is not None:
            input_image: np.ndarray | None = load_from_upload(uploaded_image)
    else:
        match algo_selector:
            case "Harris":
                sample_selector = st.selectbox(
                "Sample images",
                (
                    "-- Choose your sample", 
                    "Building", 
                    "Checkerboard",
                    )
                )
                image_key = sample_selector
                match sample_selector:
                    case "Building":
                        input_image = load_from_sample("Building.jpeg")
                    case "Checkerboard":
                        input_image = load_from_sample("Checkerboard.jpg")
                    case _:
                        input_image = None
   
            case "Canny":
                sample_selector = st.selectbox(
                                    "Sample images",
                                    (
                                        "-- Choose your sample", 
                                        "Building",
                                        "Building 2", 
                                        "Checkered Flag",
                                        )
                                    )
                image_key = sample_selector
                match sample_selector:
                    case "Building":
                        input_image = load_from_sample("Building.jpg")
                    case "Building 2":
                        input_image = load_from_sample("Building2.jpg")
                    case "Checkered Flag":
                        input_image = load_from_sample("Checkered_flag.jpg")
                    case _:
                        input_image = None
 
            case "SIFT":
                sample_selector = st.selectbox(
                                    "Sample images",
                                    (
                                        "-- Choose your sample", 
                                        "Building",
                                        "Building 2", 
                                        "Checkered Flag",
                                        )
                                    )
                image_key = sample_selector
                match sample_selector:
                    case "Building":
                        input_image = load_from_sample("Building.jpg")
                    case "Building 2":
                        input_image = load_from_sample("Building2.jpg")
                    case "Checkered Flag":
                        input_image = load_from_sample("Checkered_flag.jpg")
                    case _:
                        input_image = None
    
    #=================================================================
    # Image display and processing
    #=================================================================
    left_column, right_column = st.columns(2)
    with left_column:
        st.subheader("Input image")
        if input_image is not None:
            st.image(input_image)
    # --- detect if context changed since last Run ---
    context_changed = (
        image_key != st.session_state.last_image_key or
        algo_selector != st.session_state.last_algo
    )
    if context_changed:
        st.session_state.has_run = False
    # --- handle Run button ---
    if run_button:
        if input_image is None:
            st.warning("Please select or upload an image first")
        else:
            st.session_state.has_run = True
            st.session_state.last_image_key = image_key
            st.session_state.last_algo = algo_selector

    with right_column:
        st.subheader("Processed output")
        image_slot = st.empty()
        match algo_selector:
            case "Harris":
                if st.session_state.has_run and input_image is not None:
                    (harris_keypoints, harris_response_map), exec_time = detect_harris(
                            input_image,
                            block_size,
                            ksize,
                            k,
                            threshold_ratio
                            )
                    processed_image: np.ndarray = draw_keypoints(
                            input_image,
                            harris_keypoints,
                            radius=radius
                            )
                    image_slot.image(processed_image)
            case "Canny":
                if st.session_state.has_run and input_image is not None:
                    canny_result, exec_time = detect_canny(
                            input_image,
                            canny_sigma,
                            low_threshold,
                            high_threshold
                            )
                    # step-by-step toggle — unique to Canny
                    step = st.radio(
                            "Pipeline step",
                            ("blurred", "gradient", "nms", "edges"),
                            horizontal=True
                            )
                    
                    image_slot.image(canny_result[step], clamp=True)
            case "SIFT":
                if st.session_state.has_run and input_image is not None:
                    (sift_keypoints, sift_descriptors), exec_time = detect_sift(
                            input_image,
                            nfeatures,
                            contrast_threshold,
                            edge_threshold,
                            sift_sigma
                            )
                    processed_image = draw_sift_keypoints(input_image, sift_keypoints)
                    image_slot.image(processed_image)


    #=================================================================
    # Process visualization 
    #=================================================================

    match algo_selector:
        case "Harris":
            tab_heatmap, tab_histogram = st.tabs(
            ["Response heatmap", "Distribution"]
            )
            with tab_heatmap:
                activate_overlay: bool = st.checkbox("Show overlay")
                if harris_response_map is not None:
                    if activate_overlay: 
                        heatmap: np.ndarray = draw_heatmap_overlay(
                                input_image,
                                harris_response_map
                                ) 
                    else:
                        heatmap = draw_heatmap(
                                harris_response_map
                                )
                    st.image(heatmap)
            with tab_histogram:
                if harris_response_map is not None:
                    fig = draw_response_histogram(harris_response_map)
                    st.pyplot(fig)
                    plt.close(fig)
                    

        case "Canny":
            tab_heatmap, tab_histogram = st.tabs(
            ["Response heatmap", "Distribution"]
            )
            with tab_heatmap:
                if canny_result is not None:
                    st.image(canny_result["gradient"], clamp=True)

            with tab_histogram:
                if canny_result is not None:
                    fig = draw_response_histogram(canny_result["gradient"])
                    st.pyplot(fig)
                    plt.close(fig)
                    
        case "SIFT":
            tab_heatmap, tab_histogram, tab_matching = st.tabs(
            ["Response heatmap", "Distribution", "Matching"]
                    )
            with tab_heatmap:
                 activate_overlay: bool = st.checkbox("Show overlay")
                 if sift_keypoints is not None:
                    if activate_overlay: 
                        heatmap: np.ndarray = draw_sift_heatmap_overlay(
                                input_image,
                                sift_keypoints
                                ) 
                    else:
                        heatmap = draw_sift_heatmap(
                                input_image,
                                sift_keypoints
                                )
                    st.image(heatmap)

            with tab_histogram:
                if sift_keypoints is not None:
                    response = [kp.response for kp in sift_keypoints]
                    fig = draw_response_histogram(response)
                    st.pyplot(fig)
                    plt.close(fig)
            with tab_matching:
                if input_image is not None:
                    match input_selector:
                        case "Upload file":
                            st.info("upload a second image to see feature matching")
                            second_image_upload: UploadedFile | None = st.file_uploader(
                            "Uploaded image",
                            type=["jpg", "png"],
                            key="second_image"
                            )

                            if second_image_upload is not None and sift_keypoints is not None:
                                second_image: np.ndarray = load_from_upload(second_image_upload)
                        case "Sample images":
                            matching_sample = st.selectbox(
                                    "distorted sample image to be matched",
                                    ("Building2 cropped")
                                    )
                            match matching_sample:
                                case "Building2 cropped":
                                    second_image = load_from_sample(
                                            "sift_matching/cropped_building2.jpg"
                                            )
                    if second_image is not None:
                        (sift_keypoints2, sift_descriptors2), _ = detect_sift(
                                second_image, 
                                nfeatures, 
                                contrast_threshold, 
                                edge_threshold, 
                                sift_sigma
                                )
                        max_matches: int = st.slider(
                                "Max number of matches",
                                min_value=1,
                                max_value=50,
                                value=25
                                )
                        matches: list[cv.DMatch] = match_features(
                                sift_descriptors, 
                                sift_descriptors2,
                                max_matches
                                ) 
                        match_viz: np.ndarray = draw_feature_matches(
                                input_image, sift_keypoints,
                                second_image, sift_keypoints2,
                                matches
                                )
                        st.image(match_viz)
                        


    #=================================================================
    # Metrics section
    #=================================================================
    st.subheader("Metrics")

    match algo_selector:
        case "Harris":
            if harris_keypoints is not None:
                metrics = compute_harris_metrics(
                        harris_keypoints, 
                        harris_response_map,
                        exec_time
                        )
        case "Canny":
            if canny_result is not None:
                metrics = compute_canny_metrics(
                        canny_result["edges"], 
                        canny_result["gradient"],
                        exec_time
                        )
        case "SIFT":
            if sift_keypoints is not None:
                metrics = compute_sift_metrics(
                        sift_keypoints, 
                        exec_time
                        )

    col1, col2, col3 = st.columns(3)
    if metrics is not None:
        for col, (label, value) in zip([col1, col2, col3], metrics.items()):
            with col:
                st.metric(label, value)


if __name__ == "__main__":
    main()
