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
        ) -> list[tuple[int, int]]:
        """    
        Detect Harris corners in a grayscale float32 image.
        Returns    
        -------    
        np.ndarray        
        List of tuple (int, int) representing the location of the corners
        """
        image_f32 = _RGB_to_grayscale(image)
        harris_response_map = cv2.cornerHarris(image_f32, block_size, ksize, k) 
        threshold = threshold_ratio * harris_response_map.max()
        corner_mask = harris_response_map > threshold

        rows, cols= np.where(corner_mask)
        keypoints = list(zip(cols, rows))

        return keypoints

def draw_keypoints(
        image: np.ndarray,
        keypoints: list[tuple[int, int]],
        color: tuple[int, int, int] = (255, 0, 0),
        radius: int = 2 
        ) -> np.ndarray:
    if image.ndim == 2:
        canvas: np.ndarray = cv2.cvtColor(
                (image * 255).astype(np.uint8),
                cv2.COLOR_GRAY2RGB
                )
    else:
        canvas: np.ndarray = (image * 255).astype(np.uint8).copy()

    for (x, y) in keypoints:
        cv2.circle(canvas, (x, y), radius, color, thickness=1)

    return canvas
