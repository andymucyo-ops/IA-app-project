import cv2 as cv
import numpy as np

def compute_harris_metrics(
        keypoints: list[tuple],
        response_map: np.ndarray,
        exec_time: float
        ) ->  dict:
    count: int = len(keypoints)
    mean_response: float = np.mean(response_map[response_map > 0])

    return {
            "Keypoints count" : count,
            "Mean response" : round(mean_response, 4),
            "Exec time (ms)": round(exec_time, 3)
            }

def compute_canny_metrics(
        edge_map: np.ndarray,
        gradient_map: np.ndarray,
        exec_time: float
        ) -> dict:
    edge_pixel_count: int = np.count_nonzero(edge_map)
    mean_gradient: float = np.mean(gradient_map[gradient_map > 0])

    return {
            "Edge pixels": edge_pixel_count,
            "Mean gradient": round(mean_gradient, 4),
            "Exec time (ms)": round(exec_time, 3)           
            }
    
def compute_sift_metrics(
        keypoints: list[cv.KeyPoint],
        exec_time: float
        ) -> dict:
    count: int = len(keypoints)
    responses: list = [kp.response for kp in keypoints]
    mean_response: float = np.mean(responses) if count == 0 else 0 

    return {
            "Keypoints count" : count,
            "Mean response" : round(mean_response, 4),
            "Exec time (ms)": round(exec_time, 3)
            }

   
