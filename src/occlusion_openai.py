import os
import io
import base64
import tempfile
import requests
import numpy as np
from PIL import Image


def save_temp_image(arr):
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    Image.fromarray(arr).save(tmp.name)
    return tmp.name


def mask_to_rgba(mask):
    """
    Convert a binary mask to RGBA where alpha=255 means editable region.
    """
    mask = (mask > 0).astype(np.uint8)
    h, w = mask.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[..., 3] = mask * 255
    return rgba


def resize_with_padding_to_square(image: np.ndarray, mask: np.ndarray, size: int = 1024):
    """
    Resize image and mask to a square canvas without distortion.
    Returns:
        square_image, square_mask, meta
    meta can be used to map the output back to the original size.
    """
    h, w = image.shape[:2]

    scale = min(size / w, size / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    image_pil = Image.fromarray(image)
    mask_rgba = mask_to_rgba(mask)
    mask_pil = Image.fromarray(mask_rgba)

    image_resized = image_pil.resize((new_w, new_h), Image.Resampling.LANCZOS)
    mask_resized = mask_pil.resize((new_w, new_h), Image.Resampling.NEAREST)

    square_image = Image.new("RGB", (size, size), (0, 0, 0))
    square_mask = Image.new("RGBA", (size, size), (0, 0, 0, 0))

    offset_x = (size - new_w) // 2
    offset_y = (size - new_h) // 2

    square_image.paste(image_resized, (offset_x, offset_y))
    square_mask.paste(mask_resized, (offset_x, offset_y))

    meta = {
        "orig_h": h,
        "orig_w": w,
        "size": size,
        "new_h": new_h,
        "new_w": new_w,
        "offset_x": offset_x,
        "offset_y": offset_y,
    }

    return np.array(square_image), np.array(square_mask), meta


def restore_square_output_to_original(output_square: np.ndarray, meta: dict) -> np.ndarray:
    """
    Undo square padding workflow and return RGB image at original size.
    """
    offset_x = meta["offset_x"]
    offset_y = meta["offset_y"]
    new_w = meta["new_w"]
    new_h = meta["new_h"]
    orig_w = meta["orig_w"]
    orig_h = meta["orig_h"]

    cropped = output_square[offset_y:offset_y + new_h, offset_x:offset_x + new_w]

    restored = Image.fromarray(cropped).resize(
        (orig_w, orig_h),
        Image.Resampling.LANCZOS
    )
    return np.array(restored)


def default_prompt_for_label(label: str) -> str:
    return (
        f"Reconstruct the occluded parts of the {label}. "
        f"Preserve the original style, perspective, composition, and structural coherence. "
        f"Do not add unrelated objects."
    )


def openai_edit(image_rgb, mask_binary, label, model="gpt-image-1", prompt=None, size="1024x1024"):
    """
    OpenAI image edit with optional custom prompt.

    Args:
        image_rgb: np.ndarray HxWx3
        mask_binary: np.ndarray HxW, truthy means editable region
        label: semantic label, used for fallback/default prompt
        model: image model name
        prompt: optional custom prompt from experimental layer pipeline
        size: image edit output size; keep square for stable API behavior
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is not set")

    if prompt is None:
        prompt = default_prompt_for_label(label)

    image_rgb = np.asarray(image_rgb)
    mask_binary = np.asarray(mask_binary) > 0

    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError(f"image_rgb must be HxWx3 RGB, got {image_rgb.shape}")

    if mask_binary.ndim != 2:
        raise ValueError(f"mask_binary must be HxW, got {mask_binary.shape}")

    if image_rgb.shape[:2] != mask_binary.shape:
        raise ValueError(
            f"Image shape {image_rgb.shape[:2]} and mask shape {mask_binary.shape} do not match"
        )

    # Prepare square inputs so we do not distort the image ourselves later.
    square_image, square_mask_rgba, meta = resize_with_padding_to_square(
        image_rgb,
        mask_binary,
        size=1024
    )

    image_path = save_temp_image(square_image)
    mask_path = save_temp_image(square_mask_rgba)

    try:
        with open(image_path, "rb") as img_f, open(mask_path, "rb") as mask_f:
            response = requests.post(
                "https://api.openai.com/v1/images/edits",
                headers={"Authorization": f"Bearer {api_key}"},
                files={
                    "image": img_f,
                    "mask": mask_f,
                },
                data={
                    "model": model,
                    "prompt": prompt,
                    "size": size,
                },
                timeout=300
            )

        response.raise_for_status()
        payload = response.json()

        if "data" not in payload or not payload["data"]:
            raise ValueError(f"Unexpected OpenAI response: {payload}")

        first = payload["data"][0]

        if "b64_json" not in first:
            raise ValueError(f"OpenAI response did not include b64_json: {payload}")

        img_bytes = base64.b64decode(first["b64_json"])
        out_square = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        out_square_np = np.array(out_square)

        restored = restore_square_output_to_original(out_square_np, meta)
        return restored

    finally:
        for path in (image_path, mask_path):
            try:
                os.remove(path)
            except OSError:
                pass