import numpy as np
from PIL import Image


MAX_DIMENSION = 1024  # configurable constant at top of file

def _resize_if_needed(img: Image.Image) -> Image.Image:
    """
    Resize input if needed for imahge_io to cap max size of image
    """
    w, h = img.size
    if max(w, h) > MAX_DIMENSION:
        scale = MAX_DIMENSION / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    return img

def _normalize(image_array: np.ndarray) -> np.ndarray:
    """
    helper function that normalizes the image array (values in [0, 1])
    """
    denom: np.float32 = (image_array.max() - image_array.min())
    if denom > 0:
        normalized: np.ndarray = (image_array - image_array.min()) / denom
    else:
        normalized: np.ndarray = np.zeros_like(image_array)
    return normalized
 
def _RGB_to_grayscale(image: np.ndarray) -> np.ndarray:
    """
    converts RGB image to grayscale
    """
    grayscale = 0.299 * image[:,:,0] + 0.587 * image[:,:,1] + 0.114 * image[:,:,2]
    return grayscale.astype(np.float32)

def _normalize_255(image_array: np.ndarray) -> np.ndarray:
    """
    helper function that normalizes the image array (values in [0, 255])
    """
    denom: np.float32 = (image_array.max() - image_array.min())
    if denom > 0:
        normalized: np.ndarray = (image_array - image_array.min()) / denom
    else:
        normalized: np.ndarray = np.zeros_like(image_array)
    return normalized * 255

