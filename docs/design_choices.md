# Design Choices — Feature Extraction Explorer

This document explains the key implementation decisions made during the development of the Feature Extraction Explorer app.

***

## General Philosophy

The overarching principle throughout development was to **leverage OpenCV's optimized implementations** rather than reimplementing algorithms from scratch. This kept the codebase focused on the educational and interactive layer of the app — parameter exploration, visualization, and explanation — rather than low-level algorithm implementation. Where OpenCV abstracts away intermediate pipeline steps, the app compensates by exposing internal outputs (response maps, gradient magnitude) to give the user meaningful insight into what the algorithm is doing.

***

## Harris Corner Detector

**Using `cv2.cornerHarris` rather than a manual implementation.**

The Harris detector was implemented using OpenCV's `cv2.cornerHarris` function rather than computing the structure tensor and response matrix manually. The motivation was purely practical: reimplementing Harris from scratch would not have added educational value for the user and would have introduced unnecessary complexity and potential for numerical bugs.

The key design decision was to **not treat `cv2.cornerHarris` as a black box**. The function returns the raw response map — a float32 matrix of per-pixel corner strength values — which is retained and exposed in the diagnostic tab as a colour-mapped heatmap. This gives the user a direct visual understanding of where the algorithm assigns high corner response, making the internal mechanics of the detector tangible without requiring a manual implementation.

***

## Canny Edge Detector

**Using `cv2.Canny` with approximated NMS intermediate output.**

The same reasoning applies to Canny: OpenCV provides a well-tested, optimized implementation via `cv2.Canny`, and reimplementing the full pipeline (Gaussian blur → Sobel gradients → NMS → hysteresis) manually was not justified given the project scope.

The primary educational feature of the Canny module is the **step-by-step pipeline toggle**, which lets the user observe the output at each stage:

- **Blurred** — output of `cv2.GaussianBlur`
- **Gradient magnitude** — computed via Sobel filters and `np.hypot`, normalized to `[0, 255]`
- **NMS (approximated)** — since `cv2.Canny` applies Non-Maximum Suppression and hysteresis internally as a single operation, the NMS step is approximated by thresholding the gradient magnitude at `low_threshold`. This is not bit-for-bit identical to OpenCV's internal NMS but produces a visually meaningful "before hysteresis" state that serves the educational purpose.
- **Final edges** — direct output of `cv2.Canny`

***

## SIFT

**Density heatmap over uniform keypoint scatter.**

For the SIFT diagnostic heatmap, a **response-weighted density map** was chosen over a simple uniform dot plot. Each detected keypoint contributes a filled circle to a blank canvas, with radius proportional to `kp.size` (keypoint scale) and intensity proportional to `kp.response` (keypoint strength). The resulting map is Gaussian-blurred and colour-mapped.

This approach was chosen for two reasons:

- It provides a **spatially intuitive view** of where the most significant features are concentrated, which complements the rich-keypoint overlay (scaled circles with orientation lines) already shown in the main output panel.
- The overlay mode — blending the density map with the original image via `cv2.addWeighted` — makes it immediately clear which image regions drive the most SIFT activity, which is more informative than individual dots at scale.

**Brute-force matching over FLANN.**

Feature matching uses OpenCV's `BFMatcher` with `NORM_L2` distance and `crossCheck=True`, which retains only mutually best matches. FLANN was considered but not used — for the image sizes and match counts relevant to this demo (≤ 50 matches), brute-force matching is fast enough and requires no index-tuning parameters, keeping the implementation simple and transparent.

***

## State Management

**Recomputation over caching.**

A deliberate choice was made **not** to cache detection results in `st.session_state`. Instead, the detection function is called on every Streamlit rerun when `has_run` is `True`, meaning results are recomputed each time the user moves a parameter slider.

This was chosen for simplicity — caching results would have required invalidating the cache on parameter changes, adding complexity without a meaningful UX benefit given that all three detectors run well within interactive latency on the image sizes used. The `@timer` decorator records execution time in milliseconds, making any performance impact visible to the user directly in the metrics panel.

`st.session_state` is used only for the minimal control flow needed to implement the **run-once-then-live-update** behaviour: pressing Run once triggers detection, after which slider adjustments update the output without requiring another button press. Switching algorithm or image resets the state and requires pressing Run again.

***

## Module Structure

The codebase follows a strict single-responsibility structure:

| Module | Responsibility |
|---|---|
| `app.py` | Streamlit layout and widget definitions only — no CV logic |
| `processing/detectors.py` | Detection functions returning raw outputs |
| `processing/matching.py` | Feature matching |
| `processing/descriptors.py` | SIFT descriptor computation |
| `utils/visualization.py` | All drawing and plotting functions |
| `metrics/metrics.py` | Quantitative metric computation |
| `utils/image_io.py` | Image loading, validation, normalisation |
| `utils/helpers.py` | Shared utilities (timer decorator, normalisation helper) |

This separation ensures each module can be read, tested, and refactored independently.
