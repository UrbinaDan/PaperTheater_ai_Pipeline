import cv2
import numpy as np
from scipy.ndimage import binary_fill_holes

def remove_small_components(mask, min_area=500):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    out = np.zeros_like(mask, dtype=np.uint8)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            out[labels == i] = 1
    return out

def smooth_mask(mask, kernel_size=5):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    out = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
    out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, kernel)
    return out

def fill_holes(mask):
    return binary_fill_holes(mask).astype(np.uint8)

def cleanup_mask(mask, min_area=500, kernel_size=5):
    mask = remove_small_components(mask, min_area)
    mask = smooth_mask(mask, kernel_size)
    mask = fill_holes(mask)
    return mask