import numpy as np

def parse_florence_boxes(florence_result):
    """
    Extract detection boxes from Florence post-processed output.
    This may vary slightly by task output, so inspect your first run.
    """
    boxes = []
    # Typical structure often contains bboxes + labels in a dict
    for key, value in florence_result.items():
        if isinstance(value, dict):
            bboxes = value.get("bboxes", [])
            labels = value.get("labels", [])
            for bbox, label in zip(bboxes, labels):
                boxes.append({"bbox": bbox, "label": label})
    return boxes

def object_depth_stats(depth_map, segmented_objects):
    stats = {}
    for obj in segmented_objects:
        vals = depth_map[obj["mask"].astype(bool)]
        if len(vals) == 0:
            stats[obj["label"]] = {"median": 1.0, "min": 1.0, "max": 1.0}
        else:
            stats[id(obj)] = {
                "median": float(np.median(vals)),
                "min": float(np.min(vals)),
                "max": float(np.max(vals)),
            }
    return stats

def assign_layers(segmented_objects, depth_map, target_num_layers=6):
    items = []
    for idx, obj in enumerate(segmented_objects):
        vals = depth_map[obj["mask"].astype(bool)]
        med = float(np.median(vals)) if len(vals) else 1.0
        items.append((idx, med))

    items.sort(key=lambda x: x[1])  # near to far
    assignments = {}
    n = max(len(items), 1)

    for rank, (idx, _) in enumerate(items):
        layer = min(int(rank * target_num_layers / n), target_num_layers - 1)
        assignments[idx] = layer

    return assignments