import numpy as np
import cv2

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
        canvas: np.ndarray = cv2.cvtColor(
                (image * 255).astype(np.uint8),
                cv2.COLOR_GRAY2RGB
                )
    else:
        canvas: np.ndarray = (image * 255).astype(np.uint8).copy()

    for (x, y) in keypoints:
        cv2.circle(canvas, (x, y), radius, color, thickness=2)

    return canvas
