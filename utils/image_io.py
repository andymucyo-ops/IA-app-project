import io
import os
from PIL import UnidentifiedImageError, Image
import numpy as np
import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile
from utils.helpers import _normalize, _resize_if_needed


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
        with Image.open(io.BytesIO(raw_bytes)) as img:
            if img.mode != "RGB":
                img = img.convert("RGB")
            img = _resize_if_needed(img)
            image_array: np.ndarray = np.array(img, dtype=np.float32)
    except UnidentifiedImageError:
        st.error("Error: File isn't of the correct format")
        return None
    except TypeError:
        st.error("Error: File isn't in the correct format")
        return None
    except ValueError:
        st.error("Error: something went wrong in the file characteristics")
        return None

    return _normalize(image_array)
    
def load_from_sample(image_name: str | None = None) -> np.ndarray | None:
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
        with Image.open(image_path) as img:
            if img.mode != "RGB":
                img = img.convert("RGB")
            img = _resize_if_needed(img)
            image_array: np.ndarray = np.array(img, dtype=np.float32)
    except FileNotFoundError:
        st.error("Error: File not found")
        return None
    except UnidentifiedImageError:
        st.error("Error: File isn't of the correct format")
        return None
    except TypeError:
        st.error("Error: File isn't in the correct format")
        return None
    except ValueError:
        st.error("Error: something went wrong in the file characteristics")
        return None

    return _normalize(image_array)
    

