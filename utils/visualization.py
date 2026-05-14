import numpy as np
import cv2 as cv

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
