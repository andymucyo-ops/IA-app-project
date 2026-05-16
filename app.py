import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from streamlit.runtime.uploaded_file_manager import UploadedFile

from processing.detectors import detect_canny, detect_harris
from utils.visualization import draw_heatmap, draw_heatmap_overlay, draw_keypoints, draw_response_histogram
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
    harris_response_map = None
    canny_result = None

    #=================================================================
    # sidebar
    #=================================================================
    with st.sidebar:
        input_selector = st.radio(
                                "Image input",
                                ("Upload file", "Sample images")
                                )
        algo_selector = st.selectbox(
                                "Detector",
                                ("Canny", "Harris", "SIFT")
                                )

        match algo_selector:
            case "Canny":
                st.subheader("Parameters")
                sigma: float = st.slider(
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
            case "SIFT":
                pass
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
                st.info("SIFT detection not implemented yet")
    
    #=================================================================
    # Image display
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


        match algo_selector:
            case "Harris":
                if st.session_state.has_run and input_image is not None:
                    keypoints, harris_response_map = detect_harris(
                            input_image,
                            block_size,
                            ksize,
                            k,
                            threshold_ratio
                            )
                    processed_image: np.ndarray = draw_keypoints(
                            input_image,
                            keypoints,
                            radius=radius
                            )
                    st.image(processed_image)
            case "Canny":
                if st.session_state.has_run and input_image is not None:
                    canny_result: dict = detect_canny(
                            input_image,
                            sigma,
                            low_threshold,
                            high_threshold
                            )
                    image_slot = st.empty()
                    # step-by-step toggle — unique to Canny
                    step = st.radio(
                            "Pipeline step",
                            ("blurred", "gradient", "nms", "edges"),
                            horizontal=True
                            )
                    
                    image_slot.image(canny_result[step], clamp=True)

    #=================================================================
    # Diognostic and metrics
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
            st.info("Not implemented yet")





if __name__ == "__main__":
    main()
