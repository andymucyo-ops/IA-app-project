# 🔍 Feature Extraction Explorer

An interactive educational Streamlit app for exploring keypoint-based feature extraction algorithms. Built for the Image Analysis course (Assignment 2), deployed on Hugging Face Spaces.

***

## Project Summary

Feature Extraction Explorer lets you interactively explore three foundational computer vision algorithms along the **Detect → Describe → Match** pipeline:

| Algorithm | Role |
|---|---|
| **Harris Corner Detector** | Corner-level feature extraction |
| **Canny Edge Detector** | Edge-level feature extraction with step-by-step pipeline view |
| **SIFT** | Full keypoint detection, description, and matching |

For each algorithm, the app provides:
- Real-time parameter tuning with instant visual feedback
- Diagnostic views (response heatmaps)
- Quantitative metrics (keypoint count, mean response strength, execution time)
- In-app theoretical explanations and parameter guides

***

## Local Run Instructions

### Prerequisites

This project uses [`uv`](https://docs.astral.sh/uv/) as the package manager.

```bash
# Clone the repository
git clone <your-repo-url>
cd feature-extraction-demo

# Install dependencies
uv sync

# Run the app
uv run streamlit run app.py
```

The app will open at `http://localhost:8501`.

***

## Hugging Face Space

**Live demo**: _URL to be added after deployment_

***

## Screenshots

_Screenshots to be added after deployment._

***

## Known Limitations

- **Large image handling**: images are resized to a maximum dimension before processing to keep the app responsive. Very high-resolution uploads may lose fine detail after resizing.
- **SIFT matching for uploaded images**: when using the file upload input, the matching tab requires the user to manually upload a second image. Sample image users can select from pre-built modified pairs.
- **NMS step is approximated**: the Canny pipeline step viewer shows an approximation of the Non-Maximum Suppression output, since OpenCV's `cv2.Canny` applies NMS and hysteresis internally as a single operation.
- **No `st.session_state` caching on descriptors**: SIFT descriptors are recomputed on every parameter change. For large images with `nfeatures=0`, this may cause a slight delay.
- **Matching tab is SIFT-only**: feature matching is not available for Harris or Canny, as those algorithms do not produce descriptors.

***

## Design Choices

See [`docs/design_choices.md`](docs/design_choices.md) for a discussion of key implementation decisions.

***

## Project Structure

```
feature-extraction-demo/
│
├── app.py                        # Main Streamlit entry point
│
├── processing/
│   ├── detectors.py              # Canny, Harris, SIFT detection logic
│   ├── descriptors.py            # SIFT descriptor computation
│   └── matching.py               # Brute-force feature matcher
│
├── metrics/
│   └── metrics.py                # Keypoint count, response distribution, exec time
│
├── utils/
│   ├── image_io.py               # Image loading, validation, normalisation
│   ├── visualization.py          # Keypoint overlays, heatmaps, histograms, match viz
│   └── helpers.py                # Shared utilities (timer decorator, normalisation)
│
├── sample_images/                # Built-in sample images per algorithm
├── docs/
│   └── design_choices.md
├── requirements.txt
└── README.md
```
