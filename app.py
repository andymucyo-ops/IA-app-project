import numpy as np
import streamlit as st
import cv2 as cv
import matplotlib.pyplot as plt
from streamlit.runtime.uploaded_file_manager import UploadedFile

from metrics.metrics import compute_canny_metrics, compute_harris_metrics, compute_sift_metrics
from processing.detectors import detect_canny, detect_harris, detect_sift
from processing.matching import match_features
from utils.helpers import ALGO_FULL_NAME
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
    # Theoretical  and parameters guide expanders 
    match algo_selector:
        case "Harris":
            with st.expander(f"📖 About {ALGO_FULL_NAME['Harris']}"):
                st.markdown(
                    "Detects **corners** — points where intensity changes significantly "
                    "in all directions. Corners are more distinctive than edges (change "
                    "in one direction only) or flat regions (no change), making them "
                    "reliable landmarks for matching and tracking."
                )
                st.markdown("For each pixel, Harris builds the **structure tensor** $M$ "
                            "over a local window:")
                st.latex(r"""
                    M = \sum_{x,y} w(x,y)
                    \begin{bmatrix}
                    I_x^2 & I_x I_y \\
                    I_x I_y & I_y^2
                    \end{bmatrix}
                """)
                st.markdown(
                    "where $I_x$, $I_y$ are the image gradients and $w$ is a Gaussian "
                    "weighting window. The corner response score $R$ is then:"
                )
                st.latex(r"R = \det(M) - k \cdot (\text{tr}(M))^2")
                st.markdown(
                    "- $R \gg 0$ → **corner**\n"
                    "- $R \ll 0$ → **edge**\n"
                    "- $|R| \\approx 0$ → **flat region**"
                )

            with st.expander("🎛️ Parameter guide"):
                st.markdown(
                    "- **Block size** — neighbourhood window size per pixel. "
                    "Larger = smoother, less sensitive to noise, misses fine corners.\n"
                    "- **ksize** — Sobel kernel size for gradient computation (must be odd). "
                    "Larger = smoother gradients, less precise localisation.\n"
                    "- **k** — sensitivity coefficient in the $R$ formula. "
                    "Higher = fewer but more confident corners. Typical range: 0.04–0.06.\n"
                    "- **Threshold ratio** — fraction of max response used as cutoff. "
                    "Increase to keep only the strongest corners.\n"
                    "- **Radius** — display size of drawn circles. Visual only, does not affect detection."
                )

        case "Canny":
            with st.expander(f"📖 About {ALGO_FULL_NAME['Canny']}"):
                st.markdown(
                    "Detects **edges** — boundaries between regions of different intensity. "
                    "Canny is a multi-stage pipeline designed to find the strongest, "
                    "thinnest edges while suppressing noise, making it one of the most "
                    "widely used edge detectors in computer vision."
                )
                st.markdown("**Step 1 — Gaussian blur**: smooths the image to reduce noise "
                            "before gradient computation. Controlled by $\\sigma$:")
                st.latex(r"G(x,y) = \frac{1}{2\pi\sigma^2} e^{-\frac{x^2+y^2}{2\sigma^2}}")
                st.markdown("**Step 2 — Gradient magnitude**: Sobel filters compute intensity "
                            "change in $x$ and $y$ directions:")
                st.latex(r"\|\nabla I\| = \sqrt{G_x^2 + G_y^2}")
                st.markdown(
                    "**Step 3 — Non-maximum suppression**: thins edges to single-pixel width "
                    "by keeping only local gradient maxima along the gradient direction.\n\n"
                    "**Step 4 — Hysteresis thresholding**: two thresholds determine final edges — "
                    "pixels above $T_{high}$ are strong edges, pixels between $T_{low}$ and "
                    "$T_{high}$ are kept only if connected to a strong edge. "
                    "Pixels below $T_{low}$ are discarded."
                )

            with st.expander("🎛️ Parameter guide"):
                st.markdown(
                    "- **Sigma** ($\\sigma$) — controls Gaussian blur strength before edge detection. "
                    "Increase for noisy images; decrease to preserve fine detail.\n"
                    "- **Low threshold** — lower bound of the hysteresis window. "
                    "Decrease to recover more weak edges connected to strong ones.\n"
                    "- **High threshold** — upper bound of the hysteresis window. "
                    "Increase to keep only the strongest, most confident edges. "
                    "A ratio of $T_{high} / T_{low} \\approx 2$–$3$ is generally recommended.\n"
                    "- **Pipeline step toggle** — use the radio buttons in the output area "
                    "to step through each stage of the pipeline and observe its effect."
                )
        case "SIFT":
            with st.expander(f"📖 About {ALGO_FULL_NAME['SIFT']}"):
                st.markdown(
                    "Detects and describes **keypoints** — distinctive local features that "
                    "remain recognisable across changes in scale, rotation, and illumination. "
                    "Unlike Harris (corners only) or Canny (edges only), SIFT produces a full "
                    "**detection + description** pipeline, making it directly usable for "
                    "image matching and registration."
                )
                st.markdown(
                    "**Step 1 — Scale-space extrema detection**: the image is convolved with "
                    "Gaussians at increasing scales. Differences of Gaussians (DoG) approximate "
                    "the Laplacian and reveal blob-like structures at each scale:"
                )
                st.latex(r"D(x,y,\sigma) = G(x,y,k\sigma) * I(x,y) - G(x,y,\sigma) * I(x,y)")
                st.markdown(
                    "Keypoints are localised at **extrema** (minima and maxima) of $D$ across "
                    "scale and space."
                )
                st.markdown(
                    "**Step 2 — Keypoint filtering**: low-contrast candidates and responses "
                    "along edges are discarded using the contrast threshold and edge threshold.\n\n"
                    "**Step 3 — Orientation assignment**: a dominant gradient orientation is "
                    "computed for each keypoint from its local neighbourhood, giving SIFT "
                    "**rotation invariance**.\n\n"
                    "**Step 4 — Descriptor computation**: a 128-dimensional histogram of "
                    "gradient orientations is built around each keypoint, forming a compact "
                    "and distinctive fingerprint used for matching."
                )

            with st.expander("🎛️ Parameter guide"):
                st.markdown(
                    "- **nfeatures** — maximum number of keypoints to retain. "
                    "Set to 0 for no limit. Reduce to keep only the strongest features "
                    "and speed up computation.\n"
                    "- **Contrast threshold** — filters out low-contrast keypoints. "
                    "Increase to keep only highly distinctive features; "
                    "decrease to detect more keypoints in low-contrast regions.\n"
                    "- **Edge threshold** — filters out keypoints localised along edges "
                    "rather than corners. Increase to be more permissive; "
                    "decrease to reject more edge responses.\n"
                    "- **Sigma** ($\\sigma$) — Gaussian blur applied to the input before "
                    "scale-space construction. Higher values assume more pre-existing blur "
                    "in the image. Default 1.6 is optimal for most images.\n"
                    "- **Matching tab** — upload a second image to see SIFT descriptors "
                    "matched across the two views. Adjust max matches with the slider."
                )

            # Image input
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
                                                # "Building",
                                                "Building 2", 
                                                # "Checkered Flag",
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
