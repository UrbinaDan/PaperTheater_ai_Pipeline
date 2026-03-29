from pathlib import Path
from typing import Dict, Any, Tuple
import json

import numpy as np
from PIL import Image


def ensure_bool_mask(mask: np.ndarray) -> np.ndarray:
    return (mask > 0)


def crop_array(arr: np.ndarray, bbox: list, pad: int = 32) -> Tuple[np.ndarray, list]:
    """
    Crop array to bbox with padding.
    Returns cropped array and actual crop bbox.
    """
    x1, y1, x2, y2 = bbox
    h, w = arr.shape[:2]

    x1p = max(0, x1 - pad)
    y1p = max(0, y1 - pad)
    x2p = min(w - 1, x2 + pad)
    y2p = min(h - 1, y2 + pad)

    if arr.ndim == 2:
        cropped = arr[y1p:y2p + 1, x1p:x2p + 1]
    else:
        cropped = arr[y1p:y2p + 1, x1p:x2p + 1, :]

    return cropped, [x1p, y1p, x2p, y2p]


def paste_crop_back(
    full_shape: tuple,
    crop_bbox: list,
    crop_img: np.ndarray
) -> np.ndarray:
    """
    Paste a cropped image/mask back into full-frame canvas.
    """
    y1 = crop_bbox[1]
    y2 = crop_bbox[3] + 1
    x1 = crop_bbox[0]
    x2 = crop_bbox[2] + 1

    if len(full_shape) == 2:
        canvas = np.zeros(full_shape, dtype=crop_img.dtype)
        canvas[y1:y2, x1:x2] = crop_img
    else:
        canvas = np.zeros(full_shape, dtype=crop_img.dtype)
        canvas[y1:y2, x1:x2, :] = crop_img

    return canvas


def mask_to_rgba(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Apply alpha from mask to RGB image.
    """
    alpha = (mask.astype(np.uint8)) * 255
    return np.dstack([image, alpha])


def save_mask(mask: np.ndarray, out_path: str) -> None:
    out = (mask.astype(np.uint8)) * 255
    Image.fromarray(out).save(out_path)


def validate_generated_image(generated: np.ndarray, expected_shape: tuple) -> None:
    if not isinstance(generated, np.ndarray):
        raise TypeError("Generated output is not a numpy array")

    if generated.ndim != 3 or generated.shape[2] != 3:
        raise ValueError(f"Generated output must be RGB HxWx3, got {generated.shape}")

    if generated.shape[:2] != expected_shape[:2]:
        raise ValueError(
            f"Generated crop shape {generated.shape[:2]} does not match expected {expected_shape[:2]}"
        )


def realize_single_layer_experimental(
    image: np.ndarray,
    layer_context: Dict[str, Any],
    prompt: str,
    output_dir: str,
    openai_realize_fn,
    model_name: str,
    pad: int = 32,
) -> Dict[str, Any]:
    """
    Experimental realization for one layer.

    openai_realize_fn signature:
        generated_crop = openai_realize_fn(
            image_crop=image_crop,
            ownership_mask_crop=ownership_mask_crop,
            front_occlusion_mask_crop=front_occlusion_mask_crop,
            layer_context=layer_context,
            prompt=prompt,
            model_name=model_name,
        )
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    layer_name = layer_context["layer_name"]
    order = layer_context["order"]

    ownership_mask = ensure_bool_mask(layer_context["ownership_mask"])
    front_occlusion_mask = ensure_bool_mask(layer_context["front_occlusion_mask"])
    visible_mask = ensure_bool_mask(layer_context["visible_mask"])

    # Crop around the ownership region
    image_crop, crop_bbox = crop_array(image, layer_context["ownership_bbox"], pad=pad)
    ownership_mask_crop, _ = crop_array(ownership_mask.astype(np.uint8), layer_context["ownership_bbox"], pad=pad)
    front_occlusion_mask_crop, _ = crop_array(front_occlusion_mask.astype(np.uint8), layer_context["ownership_bbox"], pad=pad)
    visible_mask_crop, _ = crop_array(visible_mask.astype(np.uint8), layer_context["ownership_bbox"], pad=pad)

    ownership_mask_crop = ownership_mask_crop > 0
    front_occlusion_mask_crop = front_occlusion_mask_crop > 0
    visible_mask_crop = visible_mask_crop > 0

    # Save debug inputs
    crop_image_path = output_dir / f"{order:02d}_{layer_name}_input_crop.png"
    ownership_mask_path = output_dir / f"{order:02d}_{layer_name}_ownership_mask_crop.png"
    front_mask_path = output_dir / f"{order:02d}_{layer_name}_front_occlusion_mask_crop.png"
    visible_mask_path = output_dir / f"{order:02d}_{layer_name}_visible_mask_crop.png"
    prompt_path = output_dir / f"{order:02d}_{layer_name}_prompt.txt"
    meta_path = output_dir / f"{order:02d}_{layer_name}_context.json"

    Image.fromarray(image_crop).save(crop_image_path)
    save_mask(ownership_mask_crop, str(ownership_mask_path))
    save_mask(front_occlusion_mask_crop, str(front_mask_path))
    save_mask(visible_mask_crop, str(visible_mask_path))

    with open(prompt_path, "w", encoding="utf-8") as f:
        f.write(prompt)

    serializable_context = {
        "layer_name": layer_context["layer_name"],
        "order": layer_context["order"],
        "object_ids": layer_context["object_ids"],
        "labels": layer_context["labels"],
        "ownership_bbox": layer_context["ownership_bbox"],
        "visible_bbox": layer_context["visible_bbox"],
        "depth_summary": layer_context["depth_summary"],
        "front_layer_names": layer_context["front_layer_names"],
        "rear_layer_names": layer_context["rear_layer_names"],
        "scene_caption": layer_context["scene_caption"],
        "crop_bbox": crop_bbox,
    }

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(serializable_context, f, indent=2)

    # Generate crop
    generated_crop = openai_realize_fn(
        image_crop=image_crop,
        ownership_mask_crop=ownership_mask_crop,
        front_occlusion_mask_crop=front_occlusion_mask_crop,
        layer_context=layer_context,
        prompt=prompt,
        model_name=model_name,
    )

    validate_generated_image(generated_crop, image_crop.shape)

    # Raw full-frame generation
    raw_full = paste_crop_back(image.shape, crop_bbox, generated_crop)

    # Final export uses visible mask, not ownership mask
    final_visible_crop_rgba = mask_to_rgba(generated_crop, visible_mask_crop)
    raw_visible_crop_rgba = mask_to_rgba(image_crop, visible_mask_crop)

    raw_full_path = output_dir / f"{order:02d}_{layer_name}_generated_full.png"
    raw_crop_path = output_dir / f"{order:02d}_{layer_name}_generated_crop.png"
    final_visible_crop_path = output_dir / f"{order:02d}_{layer_name}_generated_visible_crop.png"
    original_visible_crop_path = output_dir / f"{order:02d}_{layer_name}_original_visible_crop.png"

    Image.fromarray(raw_full).save(raw_full_path)
    Image.fromarray(generated_crop).save(raw_crop_path)
    Image.fromarray(final_visible_crop_rgba).save(final_visible_crop_path)
    Image.fromarray(raw_visible_crop_rgba).save(original_visible_crop_path)

    return {
        "layer_name": layer_name,
        "order": order,
        "crop_bbox": crop_bbox,
        "prompt_path": str(prompt_path),
        "context_path": str(meta_path),
        "input_crop_path": str(crop_image_path),
        "ownership_mask_crop_path": str(ownership_mask_path),
        "front_occlusion_mask_crop_path": str(front_mask_path),
        "visible_mask_crop_path": str(visible_mask_path),
        "generated_full_path": str(raw_full_path),
        "generated_crop_path": str(raw_crop_path),
        "generated_visible_crop_path": str(final_visible_crop_path),
        "original_visible_crop_path": str(original_visible_crop_path),
    }