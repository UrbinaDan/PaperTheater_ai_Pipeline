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
    h, w = mask.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[..., 3] = (mask.astype(np.uint8) * 255)
    return rgba


def openai_edit(image_rgb, mask_binary, label, model="gpt-image-1"):
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is not set")

    image_path = save_temp_image(image_rgb)
    mask_path = save_temp_image(mask_to_rgba(mask_binary))

    prompt = (
        f"Reconstruct the occluded parts of the {label}. "
        f"Preserve the original style, perspective, and structural coherence. "
        f"Do not add unrelated objects."
    )

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
                "size": "1024x1024"
            },
            timeout=300
        )

    response.raise_for_status()
    payload = response.json()

    # Debugging help if API shape changes
    if "data" not in payload or not payload["data"]:
        raise ValueError(f"Unexpected OpenAI response: {payload}")

    first = payload["data"][0]

    if "b64_json" in first:
        img_bytes = base64.b64decode(first["b64_json"])
        out = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        return np.array(out)

    raise ValueError(f"OpenAI response did not include b64_json: {payload}")