import numpy as np
import cv2 as cv
from utils.helpers import _RGB_to_grayscale

def compute_sift_descriptor(
        image: np.ndarray,
        keypoints: list[cv.KeyPoint]
        ) -> np.ndarray:
    gray: np.ndarray = _RGB_to_grayscale(image)
    uint8_gray: np.ndarray = (gray * 255).astype(np.uint8)

    sift: cv.SIFT = cv.SIFT_create() 

    _, descriptors = sift.compute(uint8_gray, keypoints)

    return descriptors
