import cv2
import numpy as np

def dilate_mask(mask, k=21):
    kernel = np.ones((k, k), np.uint8)
    return cv2.dilate(mask.astype(np.uint8), kernel, iterations=1)

def heuristic_complete(mask, label):
    label = label.lower()
    if any(x in label for x in ["tree", "bush", "leaf", "canopy"]):
        return dilate_mask(mask, 31)
    if any(x in label for x in ["temple", "pagoda", "building", "house", "roof"]):
        return dilate_mask(mask, 17)
    if any(x in label for x in ["mountain", "hill", "cliff"]):
        return dilate_mask(mask, 25)
    return dilate_mask(mask, 11)