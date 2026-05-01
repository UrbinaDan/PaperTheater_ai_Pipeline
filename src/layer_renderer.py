#Renderer
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
from PIL import Image


def load_mask(mask_path: str) -> np.ndarray:
    """
    Load a mask image as a boolean numpy array.
    Assumes nonzero pixels are foreground.
    """
    mask_img = Image.open(mask_path).convert("L")
    mask = np.array(mask_img) > 0
    return mask


def save_mask(mask: np.ndarray, out_path: str) -> None:
    """
    Save a boolean mask as a uint8 PNG.
    """
    out = (mask.astype(np.uint8)) * 255
    Image.fromarray(out).save(out_path)


def build_object_mask_map(results: List[Dict[str, Any]]) -> Dict[str, np.ndarray]:
    """
    Build a map:
        object_id -> loaded mask
    from a branch result list like openai_results or amodal_results.
    """
    object_mask_map = {}

    for obj in results:
        object_id = obj["id"]
        mask_path = obj["mask_path"]
        object_mask_map[object_id] = load_mask(mask_path)

    return object_mask_map


def render_layer_masks(
    layer_plan: Dict[str, Any],
    object_mask_map: Dict[str, np.ndarray]
) -> List[Dict[str, Any]]:
    """
    Render one merged mask per layer using object_ids from the layer plan.
    """
    rendered_layers = []

    for layer in layer_plan["layers"]:
        layer_name = layer["name"]
        object_ids = layer["object_ids"]
        order = layer["order"]

        if not object_ids:
            raise ValueError(f"Layer {layer_name} has no object_ids")

        first_mask = object_mask_map[object_ids[0]]
        layer_mask = np.zeros_like(first_mask, dtype=bool)

        for object_id in object_ids:
            if object_id not in object_mask_map:
                raise KeyError(f"Missing mask for object_id: {object_id}")
            layer_mask |= object_mask_map[object_id]

        rendered_layers.append({
            "name": layer_name,
            "order": order,
            "object_ids": object_ids,
            "mask": layer_mask,
        })

    return rendered_layers


def make_layer_preview(
    image: np.ndarray,
    mask: np.ndarray
) -> np.ndarray:
    """
    Create an RGBA preview where masked pixels keep RGB
    and everything else becomes transparent.
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must be an RGB numpy array of shape (H, W, 3)")

    alpha = (mask.astype(np.uint8)) * 255
    rgba = np.dstack([image, alpha])
    return rgba


def save_rendered_layers(
    rendered_layers: List[Dict[str, Any]],
    image: np.ndarray,
    output_dir: str
) -> List[Dict[str, Any]]:
    """
    Save layer masks and transparent previews.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved = []

    for layer in rendered_layers:
        layer_name = layer["name"]
        order = layer["order"]
        mask = layer["mask"]

        mask_path = output_dir / f"{order:02d}_{layer_name}_mask.png"
        preview_path = output_dir / f"{order:02d}_{layer_name}_preview.png"

        save_mask(mask, str(mask_path))

        preview = make_layer_preview(image, mask)
        Image.fromarray(preview).save(preview_path)

        saved.append({
            "name": layer_name,
            "order": order,
            "object_ids": layer["object_ids"],
            "mask_path": str(mask_path),
            "preview_path": str(preview_path),
        })

    return saved