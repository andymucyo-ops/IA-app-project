import io
import os
from PIL import UnidentifiedImageError
import PIL.Image
import numpy as np
from streamlit.runtime.uploaded_file_manager import UploadedFile

def load_from_upload(uploaded_file: None | UploadedFile) -> np.ndarray | None:
    """
    Args:
        uploaded_file: A Streamlit UploadedFile object.
    Returns:
        Normalized image as a float32 numpy array (values in [0, 1]), or None if the image is not valid.
    """
    if uploaded_file is None:
        return None
    raw_bytes: bytes = uploaded_file.getvalue()

    try:
        with PIL.Image.open(io.BytesIO(raw_bytes)) as img:
            if img.mode != "RGB":
                img = img.convert("RGB")
            image_array: np.ndarray = np.array(img, dtype=np.float32)
    except UnidentifiedImageError:
        print("Error: File isn't of the correct format")
        return None
    except TypeError:
        print("Error: File isn't in the correct format")
        return None
    except ValueError:
        print("Error: something went wrong in the file characteristics")
        return None

    denom: float = (image_array.max() - image_array.min())
    if denom > 0:
        normalized: np.ndarray = (image_array - image_array.min()) / denom
    else:
        normalized: np.ndarray = np.zeros_like(image_array)
        
    return normalized
    
def load_from_sample(image_name: str | None = None) -> np.ndarray:
    """
    Args:
        image_name: name of one of the sample image.
    Returns:
        Normalized image as a float32 numpy array (values in [0, 1]), or None if the image is not valid.
    """
    if image_name is None:
        return None
    else:
        image_path: str = os.path.join("./sample_images/", image_name)

    try:
        with PIL.Image.open(image_path) as img:
            if img.mode != "RGB":
                img = img.convert("RGB")
            image_array: np.ndarray = np.array(img, dtype=np.float32)
    except FileNotFoundError:
        print("Error: File not found")
        return None
    except UnidentifiedImageError:
        print("Error: File isn't of the correct format")
        return None
    except TypeError:
        print("Error: File isn't in the correct format")
        return None
    except ValueError:
        print("Error: something went wrong in the file characteristics")
        return None

    denom: float = (image_array.max() - image_array.min())
    if denom > 0:
        normalized: np.ndarray = (image_array - image_array.min()) / denom
    else:
        normalized: np.ndarray = np.zeros_like(image_array)
        
    return normalized
    

