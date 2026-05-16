import numpy as np
import cv2 as cv


def match_features(
        desc1: np.ndarray,
        desc2: np.ndarray,
        max_matches: int = 50
        ) -> list[cv.DMatch]: 
    matcher: cv.BFMatcher = cv.BFMatcher(cv.NORM_L2, crossCheck=True)
    matches = matcher.match(desc1, desc2)
    matches = sorted(matches, key=lambda m: m.distance)

    return matches[:max_matches]
