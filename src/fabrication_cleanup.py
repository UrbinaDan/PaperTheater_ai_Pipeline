from typing import Dict, Any, List
import numpy as np
import cv2


def ensure_binary_mask(mask: np.ndarray) -> np.ndarray:
    """
    Convert mask to binary uint8 with values {0, 255}.
    """
    if mask.dtype == np.bool_:
        return (mask.astype(np.uint8)) * 255

    out = (mask > 0).astype(np.uint8) * 255
    return out


def remove_small_components(mask: np.ndarray, min_area: int = 200) -> np.ndarray:
    """
    Remove connected foreground components smaller than min_area.
    """
    binary = ensure_binary_mask(mask)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

    cleaned = np.zeros_like(binary)

    for label_idx in range(1, num_labels):  # skip background
        area = stats[label_idx, cv2.CC_STAT_AREA]
        if area >= min_area:
            cleaned[labels == label_idx] = 255

    return cleaned


def fill_small_holes(mask: np.ndarray, max_hole_area: int = 200) -> np.ndarray:
    """
    Fill holes inside foreground regions if the hole area is small enough.
    """
    binary = ensure_binary_mask(mask)

    inverted = cv2.bitwise_not(binary)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(inverted, connectivity=8)

    filled = binary.copy()
    h, w = binary.shape

    for label_idx in range(1, num_labels):
        area = stats[label_idx, cv2.CC_STAT_AREA]

        ys, xs = np.where(labels == label_idx)
        touches_border = (
            np.any(xs == 0) or np.any(xs == w - 1) or
            np.any(ys == 0) or np.any(ys == h - 1)
        )

        # only fill interior holes, not exterior background
        if (not touches_border) and area <= max_hole_area:
            filled[labels == label_idx] = 255

    return filled


def smooth_mask(mask: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """
    Light morphological smoothing: close then open.
    """
    binary = ensure_binary_mask(mask)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    smoothed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    smoothed = cv2.morphologyEx(smoothed, cv2.MORPH_OPEN, kernel)

    return smoothed


def fabrication_cleanup_mask(
    mask: np.ndarray,
    min_component_area: int = 200,
    max_hole_area: int = 200,
    smooth_kernel_size: int = 5,
) -> np.ndarray:
    """
    Safe fabrication-oriented cleanup pipeline.
    """
    cleaned = ensure_binary_mask(mask)
    cleaned = remove_small_components(cleaned, min_area=min_component_area)
    cleaned = fill_small_holes(cleaned, max_hole_area=max_hole_area)
    cleaned = smooth_mask(cleaned, kernel_size=smooth_kernel_size)
    return cleaned > 0


def cleanup_rendered_layers_for_fabrication(
    rendered_layers: List[Dict[str, Any]],
    min_component_area: int = 200,
    max_hole_area: int = 200,
    smooth_kernel_size: int = 5,
) -> List[Dict[str, Any]]:
    """
    Apply fabrication cleanup to each rendered layer mask.
    Returns a new list with cleaned masks.
    """
    cleaned_layers = []

    for layer in rendered_layers:
        cleaned_mask = fabrication_cleanup_mask(
            layer["mask"],
            min_component_area=min_component_area,
            max_hole_area=max_hole_area,
            smooth_kernel_size=smooth_kernel_size,
        )

        cleaned_layers.append({
            "name": layer["name"],
            "order": layer["order"],
            "object_ids": layer["object_ids"],
            "mask": cleaned_mask,
        })

    return cleaned_layers