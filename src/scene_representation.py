import json
from pathlib import Path


def build_scene_representation(
    image_path,
    image_shape,
    caption,
    stable_objects,
):
    """
    Build a structured scene representation.

    This is the main container that will be passed to:
    - planners
    - branches
    - export
    """

    h, w = image_shape[:2]

    # IMPORTANT: remove raw masks for JSON version
    json_objects = []
    for obj in stable_objects:
        json_objects.append({
            "id": obj["id"],
            "label": obj["label"],
            "bbox": obj["bbox"],
            "score": obj["score"],
            "area": obj["area"],
            "centroid": obj["centroid"],
            "depth_mean": obj["depth_mean"],
            "depth_median": obj["depth_median"],
        })

    scene = {
        "image_path": image_path,
        "image_size": [w, h],
        "caption": caption,
        "objects": json_objects,
    }

    return scene


def save_scene_representation(scene, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(scene, f, indent=2)


def load_scene_representation(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)