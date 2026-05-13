import numpy as np
import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile

from processing.detectors import detect_harris
from utils.visualization import draw_keypoints
from utils.image_io import load_from_sample, load_from_upload



def main():
    st.title("🔍 Feature Extraction Explorer")

    # sidebar
    input_image = None
    with st.sidebar:
        input_selector = st.radio(
                                "Image input",
                                ("Upload file", "Sample images")
                                )
        algo_selector = st.selectbox(
                                "Detector",
                                ("Canny", "Harris", "SIFT")
                                )
        # st.info("Parameters will appear here")
        match algo_selector:
            case "Canny":
                pass
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
     
    # main area 
    left_column, right_column = st.columns(2)
    with left_column:
        if input_selector == "Upload file":
            uploaded_image: UploadedFile | None = st.file_uploader(
                    "Uploaded image",
                    type=["jpg", "png"]
                    )
            if uploaded_image is not None:
                input_image: np.ndarray | None = load_from_upload(uploaded_image)
                if input_image is not None:
                    st.image(input_image)
        else:
            sample_selector = st.selectbox(
                    "Sample images",
                    (
                        "-- Choose your sample", 
                        "Building", 
                        "Goat with glasses", 
                        "Building 2",
                        "Checkerboard",
                        # "Polygones"
                        )
                    )
            # st.info("Sample image placeholder")
            match sample_selector:
                case "Building":
                    input_image = load_from_sample("Building.jpeg")
                case "Goat with glasses":
                    input_image = load_from_sample("goat.JPG")
                case "Building 2":
                    input_image = load_from_sample("building2.jpg")
                case "Checkerboard":
                    input_image = load_from_sample("Checkerboard.jpg")
                # case "Polygones":
                    # input_image = load_from_sample("polygones.jpg")
                case _:
                    input_image = None
            if input_image is not None:
                st.image(input_image)
        
    with right_column:

        st.subheader("Processed output")
        if algo_selector == "Harris" and run_button:
            if input_image is None:
                st.error("Please select or upload an image")
            else:
                keypoints = detect_harris(
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

        # st.info("Processed output placeholder")

    # placeholders for dignostic and info selection
    # TODO: Diagnostic viewer
    # TODO: Metrics info




if __name__ == "__main__":
    main()
