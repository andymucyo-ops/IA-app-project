import cv2
import numpy as np

def _RGB_to_grayscale(image: np.ndarray) -> np.ndarray:
    grayscale = 0.299 * image[:,:,0] + 0.587 * image[:,:,1] + 0.114 * image[:,:,2]
    return grayscale.astype(np.float32)

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
        # image_f32 = cv2.(image.astype(np.float32),cv2.COLOR_GRAY2RGB)
        image_f32 = _RGB_to_grayscale(image)
        harris_response_map = cv2.cornerHarris(image_f32, block_size, ksize, k) 
        threshold = threshold_ratio * harris_response_map.max()
        corner_mask = harris_response_map > threshold

        rows, cols= np.where(corner_mask)
        keypoints = list(zip(cols, rows))

        return keypoints, harris_response_map
