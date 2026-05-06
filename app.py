import numpy as np
import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile

from utils.image_io import load_from_sample, load_from_upload




def main():
    st.title("🔍 Feature Extraction Explorer")

    #sidebar
    sidebar = st.sidebar
    input_selector = sidebar.radio(
                            "Image input",
                            ("Upload file", "Sample images")
                            )
    algo_selector = sidebar.selectbox(
                            "Algorithm",
                            ("Canny", "Harris", "SIFT")
                            )
    with sidebar:
        st.info("Parameters will appear here")
    sidebar.button("Run")
     
    #main area 
    left_column, right_column = st.columns(2)
    with left_column:
        # st.info("Original image placeholder")
        if input_selector =="Upload file":
            uploaded_image: UploadedFile = st.file_uploader(
                    "Uploaded image",
                    type=["jpg", "png"]
                    )
            if uploaded_image is not None:
                processed_image: np.ndarray = load_from_upload(uploaded_image)
                st.image(processed_image)
        else:
            sample_selector = st.selectbox(
                    "Sample images",
                    ("Building", "Goat with glasses")
                    )
            # st.info("Sample image placeholder")
            match sample_selector:
                case "Building":
                    sample_image: np.ndarray = load_from_sample("Building.jpeg")
                case "Goat with glasses":
                    sample_image: np.ndarray = load_from_sample("goat.JPG")
            st.image(sample_image)
        
    with right_column:
        st.info("Processed output placeholder")

    #placeholders for dignostic and info selection
    diognostic_viewer = st.info("Diognostic viewer")
    metrics_info = st.info("Metrics info")




if __name__ == "__main__":
    main()
