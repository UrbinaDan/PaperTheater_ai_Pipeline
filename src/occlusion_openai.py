import os
import base64
import requests
from PIL import Image
import numpy as np
import tempfile

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
    api_key = os.environ["OPENAI_API_KEY"]

    image_path = save_temp_image(image_rgb)
    mask_path = save_temp_image(mask_to_rgba(mask_binary))

    prompt = (
        f"Reconstruct the occluded parts of the {label}. "
        f"Preserve the original style, perspective, silhouette logic, and surrounding scene context. "
        f"Do not add unrelated objects. Keep the completion structurally coherent."
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
    data = response.json()

    b64 = data["data"][0]["b64_json"]
    out = Image.open(tempfile.NamedTemporaryFile(suffix=".png", delete=False).name)