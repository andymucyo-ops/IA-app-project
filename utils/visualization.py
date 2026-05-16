import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from utils.helpers import _normalize_255

def draw_keypoints(
        image: np.ndarray,
        keypoints: list[tuple[int, int]],
        color: tuple[int, int, int] = (255, 0, 0),
        radius: int = 2 
        ) -> np.ndarray:
    """
    Draws circles on the detected corners following Harris Corner Detection.
    returns
    --------
    canvas: input image with circles drawn on the detected corners 
    """
    if image.ndim == 2:
        canvas: np.ndarray = cv.cvtColor(
                (image * 255).astype(np.uint8),
                cv.COLOR_GRAY2RGB
                )
    else:
        canvas: np.ndarray = (image * 255).astype(np.uint8).copy()

    for (x, y) in keypoints:
        cv.circle(canvas, (x, y), radius, color, thickness=2)

    return canvas

def draw_heatmap(response_map: np.ndarray) -> np.ndarray:
    response_map_uint8: np.ndarray = _normalize_255(response_map).astype(np.uint8)
    heatmap_BGR: np.ndarray = cv.applyColorMap(response_map_uint8, cv.COLORMAP_JET)
    heatmap_RBG: np.ndarray = cv.cvtColor(heatmap_BGR, cv.COLOR_BGR2RGB) 

    return heatmap_RBG.astype(np.uint8)

def draw_heatmap_overlay(image: np.ndarray, response_map: np.ndarray) -> np.ndarray:
    response_map_uint8: np.ndarray = _normalize_255(response_map).astype(np.uint8)
    uint8_image: np.ndarray = (image * 255).astype(np.uint8)
    heatmap_BGR: np.ndarray = cv.applyColorMap(response_map_uint8, cv.COLORMAP_JET)
    heatmap_RGB: np.ndarray = cv.cvtColor(heatmap_BGR, cv.COLOR_BGR2RGB) 
    blended: np.ndarray = cv.addWeighted(uint8_image, 0.5, heatmap_RGB, 0.5, 0)

    return blended

def draw_sift_heatmap(image: np.ndarray, keypoints: list[cv.KeyPoint]) -> np.ndarray:
    height, width = image.shape[:2]
    density_map: np.ndarray = np.zeros((height, width), dtype=np.float32)

    for kp in keypoints:
        x,y = int(kp.pt[0]), int(kp.pt[1])
        scale = kp.size
        cv.circle(density_map, (x,y), radius=int(scale), color=kp.response, thickness=-1)

    blurred_density = cv.GaussianBlur(density_map, (0, 0), sigmaX=20)
    normalized = _normalize_255(blurred_density).astype(np.uint8)
    heatmap_BGR: np.ndarray = cv.applyColorMap(normalized, cv.COLORMAP_JET)
    heatmap_RGB: np.ndarray = cv.cvtColor(heatmap_BGR, cv.COLOR_BGR2RGB) 

    return heatmap_RGB

def draw_sift_heatmap_overlay(image: np.ndarray, keypoints: list[cv.KeyPoint]) -> np.ndarray:
    uint8_image: np.ndarray = (image * 255).astype(np.uint8)
    height, width = image.shape[:2]
    density_map: np.ndarray = np.zeros((height, width), dtype=np.float32)

    for kp in keypoints:
        x,y = int(kp.pt[0]), int(kp.pt[1])
        scale = kp.size
        cv.circle(density_map, (x,y), radius=int(scale), color=kp.response, thickness=-1)

    blurred_density = cv.GaussianBlur(density_map, (0, 0), sigmaX=20)
    normalized = _normalize_255(blurred_density).astype(np.uint8)
    heatmap_BGR: np.ndarray = cv.applyColorMap(normalized, cv.COLORMAP_JET)
    heatmap_RGB: np.ndarray = cv.cvtColor(heatmap_BGR, cv.COLOR_BGR2RGB) 
    blended = cv.addWeighted(uint8_image, 0.5, heatmap_RGB, 0.5, 0)

    return blended


def draw_response_histogram(response_map: np.ndarray) -> plt.Figure:
    response_map_1D: np.ndarray = np.ravel(response_map)
    fig, ax = plt.subplots()
    ax.hist(response_map_1D, 
            bins=80, 
            color="steelblue", 
            edgecolor=None
            )
    ax.set_xlabel("Response values")
    ax.set_ylabel("Pixel count")
    ax.set_title("Response strength distribution")

    return fig

def draw_sift_keypoints(
        image,
        keypoints
        ):
    uint8_image = (image * 255).astype(np.uint8)
    output: np.ndarray = uint8_image.copy()
    cv.drawKeypoints(
            uint8_image,
            keypoints,
            output,
            flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
            )
    return output
    
def draw_feature_matching(
        image1: np.ndarray,
        keypoints1: list[cv.KeyPoint],
        image2: np.ndarray,
        keypoints2: list[cv.KeyPoint],
        matches: list[cv.DMatch]
        ) -> np.ndarray:
    uint8_image1: np.ndarray = (image1 * 255).astype(np.uint8)
    uint8_image2: np.ndarray = (image2 * 255).astype(np.uint8)

    output: np.ndarray = cv.drawMatches(
            uint8_image1, keypoints1,
            uint8_image2, keypoints2,
            matches,
            outImg=None,
            flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
            )

    return cv.cvtColor(output, cv.COLOR_BGR2RGB)
