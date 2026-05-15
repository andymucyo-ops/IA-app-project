import cv2 as cv
import numpy as np
from utils.helpers import _RGB_to_grayscale, _normalize_255


def detect_harris(
        image: np.ndarray,
        block_size: int = 2,
        ksize: int = 3,
        k: float = 0.04,
        threshold_ratio: float = 0.01
        ) -> tuple[list[tuple[int, int]], np.ndarray]:
        """    
        Detect Harris corners in a grayscale float32 image.
        Returns    
        -------    
        np.ndarray a grid containing the normalized values of each pixel 
        List of tuple (int, int) representing the location of the corners
        """
        image_f32 = _RGB_to_grayscale(image)
        harris_response_map = cv.cornerHarris(image_f32, block_size, ksize, k) 
        threshold = threshold_ratio * harris_response_map.max()
        corner_mask = harris_response_map > threshold

        rows, cols= np.where(corner_mask)
        keypoints = list(zip(cols, rows))

        return keypoints, harris_response_map

def detect_canny(
        image: np.ndarray,
        sigma: float,
        low_threshold: float,
        high_threshold: float
        ) -> dict[str,np.ndarray]:
    """
    Detects edges of object on a Gray uint8 image represented as np.ndarray
    Returns.
    -------
    Dictonnary of np.ndarray representing each step of Canny edge detection:
    {
        "blurred":   np.ndarray,   after Gaussian blur, shape (H, W), uint8
        "gradient":  np.ndarray,   gradient magnitude map, shape (H, W), uint8
        "nms":       np.ndarray,   after non-maximum suppression, shape (H, W), uint8
        "edges":     np.ndarray,   final binary edge map, shape (H, W), uint8
    }
    """
    gray: np.ndarray = _RGB_to_grayscale(image)
    uint8_gray: np.ndarray = (gray * 255).astype(np.uint8) 

    #Gaussian blur:
    kernel_size: int = 2 * int(np.ceil(sigma * 3)) + 1
    blurred: np.ndarray = cv.GaussianBlur(uint8_gray, (kernel_size, kernel_size), sigma)

    #Gradient magnitude 
    grad_x: np.ndarray = cv.Sobel(
            blurred, 
            cv.CV_64F, 
            dx=1,
            dy=0,
            ksize=3
            )
    grad_y: np.ndarray = cv.Sobel(
            blurred, 
            cv.CV_64F, 
            dx=0,
            dy=1,
            ksize=3
            )
    gradient_magnitude = np.hypot(grad_x, grad_y)
    gradient_uint8: np.ndarray = _normalize_255(gradient_magnitude).astype(np.uint8) 

    #NMS + Hysteresis
    edges: np.ndarray = cv.Canny(blurred, low_threshold, high_threshold)

    #approximate NMS intermediary since cv.Canny directly applies NMS and Hysteresis
    nms_approx: np.ndarray = (gradient_magnitude > low_threshold).astype(np.uint8) * 255


    return {
        "blurred": blurred,
        "gradient": gradient_uint8,
        "nms": nms_approx,
        "edges": edges,
        }

