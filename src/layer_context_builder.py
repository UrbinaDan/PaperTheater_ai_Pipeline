from typing import Dict, Any, List, Tuple
import numpy as np


def combine_masks_by_object_ids(
    object_ids: List[str],
    object_mask_map: Dict[str, np.ndarray]
) -> np.ndarray:
    """
    Combine object masks into a single boolean mask.
    """
    if not object_ids:
        raise ValueError("object_ids is empty")

    first_id = object_ids[0]
    if first_id not in object_mask_map:
        raise KeyError(f"Missing object mask for {first_id}")

    combined = np.zeros_like(object_mask_map[first_id], dtype=bool)

    for object_id in object_ids:
        if object_id not in object_mask_map:
            raise KeyError(f"Missing object mask for {object_id}")
        combined |= (object_mask_map[object_id] > 0)

    return combined


def build_front_occlusion_mask(
    layer_index: int,
    layers: List[Dict[str, Any]],
    object_mask_map: Dict[str, np.ndarray]
) -> np.ndarray:
    """
    Build the union of all masks from layers in front of the current layer.
    """
    current_layer = layers[layer_index]
    current_object_ids = current_layer["object_ids"]

    first_id = current_object_ids[0]
    base_shape = object_mask_map[first_id].shape
    front_mask = np.zeros(base_shape, dtype=bool)

    for later_layer in layers[layer_index + 1:]:
        later_mask = combine_masks_by_object_ids(
            later_layer["object_ids"],
            object_mask_map
        )
        front_mask |= later_mask

    return front_mask


def subtract_front_occlusion(
    ownership_mask: np.ndarray,
    front_occlusion_mask: np.ndarray
) -> np.ndarray:
    """
    Visible export mask = ownership minus front layers.
    """
    return ownership_mask & (~front_occlusion_mask)


def mask_to_bbox(mask: np.ndarray) -> List[int]:
    """
    Compute [x1, y1, x2, y2] from a boolean mask.
    Returns full-image bbox if mask is empty.
    """
    ys, xs = np.where(mask > 0)

    if len(xs) == 0 or len(ys) == 0:
        h, w = mask.shape
        return [0, 0, w - 1, h - 1]

    x1 = int(xs.min())
    y1 = int(ys.min())
    x2 = int(xs.max())
    y2 = int(ys.max())
    return [x1, y1, x2, y2]


def gather_layer_objects(
    layer: Dict[str, Any],
    scene_repr: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """
    Return scene_repr object records referenced by layer["object_ids"].
    """
    object_map = {obj["id"]: obj for obj in scene_repr["objects"]}
    out = []

    for object_id in layer["object_ids"]:
        if object_id not in object_map:
            raise KeyError(f"Layer references missing object id: {object_id}")
        out.append(object_map[object_id])

    return out


def summarize_depth(layer_objects: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Aggregate simple depth stats for the layer.
    """
    depth_medians = [float(obj["depth_median"]) for obj in layer_objects]
    depth_means = [float(obj["depth_mean"]) for obj in layer_objects]

    return {
        "depth_median_mean": float(np.mean(depth_medians)),
        "depth_mean_mean": float(np.mean(depth_means)),
        "depth_median_min": float(np.min(depth_medians)),
        "depth_median_max": float(np.max(depth_medians)),
    }


def extract_caption_text(scene_repr: Dict[str, Any]) -> str:
    """
    Normalize caption into a plain string.
    """
    caption = scene_repr.get("caption", "")

    if isinstance(caption, str):
        return caption

    if isinstance(caption, dict):
        # Prefer the only or first value if Florence stored it inside a dict.
        for _, value in caption.items():
            if isinstance(value, str):
                return value

    return ""


def build_layer_contexts(
    scene_repr: Dict[str, Any],
    layer_plan: Dict[str, Any],
    object_mask_map: Dict[str, np.ndarray]
) -> List[Dict[str, Any]]:
    """
    Build realization contexts for each layer.

    Each context contains:
    - ownership_mask
    - front_occlusion_mask
    - visible_mask
    - bbox
    - layer metadata
    - scene caption
    - depth summary
    """
    layers = sorted(layer_plan["layers"], key=lambda x: x["order"])
    caption_text = extract_caption_text(scene_repr)

    layer_contexts: List[Dict[str, Any]] = []

    for layer_index, layer in enumerate(layers):
        layer_objects = gather_layer_objects(layer, scene_repr)

        ownership_mask = combine_masks_by_object_ids(
            layer["object_ids"],
            object_mask_map
        )

        front_occlusion_mask = build_front_occlusion_mask(
            layer_index,
            layers,
            object_mask_map
        )

        visible_mask = subtract_front_occlusion(
            ownership_mask,
            front_occlusion_mask
        )

        ownership_bbox = mask_to_bbox(ownership_mask)
        visible_bbox = mask_to_bbox(visible_mask)

        labels = [obj["label"] for obj in layer_objects]
        depth_summary = summarize_depth(layer_objects)

        front_layer_names = [l["name"] for l in layers[layer_index + 1:]]
        rear_layer_names = [l["name"] for l in layers[:layer_index]]

        layer_contexts.append({
            "layer_name": layer["name"],
            "order": layer["order"],
            "object_ids": layer["object_ids"],
            "labels": labels,
            "layer_objects": layer_objects,
            "ownership_mask": ownership_mask,
            "front_occlusion_mask": front_occlusion_mask,
            "visible_mask": visible_mask,
            "ownership_bbox": ownership_bbox,
            "visible_bbox": visible_bbox,
            "scene_caption": caption_text,
            "depth_summary": depth_summary,
            "front_layer_names": front_layer_names,
            "rear_layer_names": rear_layer_names,
        })

    return layer_contexts