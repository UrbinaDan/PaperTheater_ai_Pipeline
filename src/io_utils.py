from pathlib import Path
import json
import cv2
import numpy as np
from PIL import Image

def ensure_dirs(paths_obj):
    for value in paths_obj.__dict__.values():
        if isinstance(value, Path):
            value.mkdir(parents=True, exist_ok=True)

def load_image(path, max_side=None):
    img = Image.open(path).convert("RGB")
    arr = np.array(img)
    if max_side is not None:
        h, w = arr.shape[:2]
        scale = min(max_side / max(h, w), 1.0)
        if scale < 1.0:
            arr = cv2.resize(arr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return arr

def save_image(path, arr):
    Image.fromarray(arr).save(path)

def save_mask(path, mask):
    Image.fromarray((mask.astype(np.uint8) * 255)).save(path)

def load_mask(path):
    return (np.array(Image.open(path).convert("L")) > 127).astype(np.uint8)

def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)