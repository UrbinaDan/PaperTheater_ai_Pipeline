from typing import Dict, List, Any


def _layer_name_for_label(label: str) -> str:
    """
    Map semantic labels to readable deterministic layer names.
    """
    if label == "sky":
        return "background_sky"
    if label == "mountain":
        return "mountain_far"
    if label == "foliage":
        return "mid_foliage"
    if label == "temple":
        return "temple_main"
    return f"layer_{label}"


def plan_layers_deterministic(scene_repr: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Create a simple deterministic paper-theater layer plan from scene representation.

    Strategy:
    - read objects from scene_repr["objects"]
    - sort objects by depth_median from far to near
    - assign each object to its own layer
    - generate stable readable layer names

    Expected input:
        scene_repr = {
            "objects": [
                {
                    "id": "obj_000",
                    "label": "sky",
                    "depth_median": 0.0,
                    ...
                },
                ...
            ]
        }

    Returns:
        {
            "layers": [
                {
                    "name": "background_sky",
                    "object_ids": ["obj_002"],
                    "order": 0
                },
                ...
            ]
        }
    """
    if "objects" not in scene_repr:
        raise KeyError("scene_repr must contain an 'objects' field")

    objects = scene_repr["objects"]

    if not isinstance(objects, list):
        raise TypeError("scene_repr['objects'] must be a list")

    for obj in objects:
        if "id" not in obj:
            raise KeyError("Each object must contain an 'id'")
        if "label" not in obj:
            raise KeyError(f"Object {obj.get('id', '<unknown>')} is missing 'label'")
        if "depth_median" not in obj:
            raise KeyError(f"Object {obj.get('id', '<unknown>')} is missing 'depth_median'")

    objects_sorted = sorted(objects, key=lambda o: o["depth_median"])

    layers: List[Dict[str, Any]] = []

    for order, obj in enumerate(objects_sorted):
        layer_name = _layer_name_for_label(obj["label"])

        layers.append({
            "name": layer_name,
            "object_ids": [obj["id"]],
            "order": order,
        })

    return {"layers": layers}