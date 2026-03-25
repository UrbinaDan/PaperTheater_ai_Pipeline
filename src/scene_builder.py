import re
import json
from pathlib import Path

import numpy as np


def canonicalize_label(label: str) -> str:
    label = label.lower().strip()

    # normalize common Florence variants
    if label in {"pagoda", "building", "temple"}:
        return "temple"
    if label in {"plants", "plant", "bush", "bushes", "foreground plants", "foreground plant", "tree"}:
        return "foliage"
    if label in {"mountain"}:
        return "mountain"
    if label in {"sky"}:
        return "sky"
    if label in {"bridge"}:
        return "bridge"

    return label


def box_iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)

    union = area_a + area_b - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union


def parse_florence_boxes(det_output, image_shape):
    """
    Parse Florence OPEN_VOCABULARY_DETECTION output and convert
    loc tokens from 0-999 space into image pixel coordinates.
    """
    if not isinstance(det_output, dict) or len(det_output) == 0:
        return []

    h, w = image_shape[:2]

    key = list(det_output.keys())[0]
    text = det_output[key]

    pattern = r'([^<>]+?)<loc_(\d+)><loc_(\d+)><loc_(\d+)><loc_(\d+)>'
    matches = re.findall(pattern, text)

    boxes = []
    for raw_label, x1, y1, x2, y2 in matches:
        label = raw_label.lower()

        for bad in ["a ", "the ", ","]:
            label = label.replace(bad, " ")

        words = [w.strip() for w in label.split() if w.strip()]
        label = words[-1] if len(words) > 0 else ""

        if not label:
            continue

        label = canonicalize_label(label)

        x1 = int(int(x1) / 999 * w)
        y1 = int(int(y1) / 999 * h)
        x2 = int(int(x2) / 999 * w)
        y2 = int(int(y2) / 999 * h)

        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(0, min(x2, w - 1))
        y2 = max(0, min(y2, h - 1))

        if x2 <= x1 or y2 <= y1:
            continue

        area = (x2 - x1) * (y2 - y1)
        if area < 0.01 * (w * h):
            continue

        if (x2 - x1) > 0.95 * w and (y2 - y1) > 0.95 * h:
            continue

        boxes.append({
            "label": label,
            "bbox": [x1, y1, x2, y2]
        })

    return deduplicate_boxes(boxes)


def deduplicate_boxes(boxes, iou_thresh=0.65):
    """
    Remove overlapping duplicate detections of the same semantic object.
    """
    kept = []

    for box in boxes:
        label = box["label"]
        bbox = box["bbox"]

        duplicate = False
        for prev in kept:
            if prev["label"] == label and box_iou(prev["bbox"], bbox) >= iou_thresh:
                duplicate = True
                break

        if not duplicate:
            kept.append(box)

    return kept


def merge_segmented_by_label(segmented):
    """
    Merge masks that share the same canonical label.
    """
    merged = {}

    for obj in segmented:
        label = canonicalize_label(obj["label"])
        mask = obj["mask"].astype(np.uint8)

        if label not in merged:
            merged[label] = {
                "label": label,
                "bbox": obj["bbox"],
                "mask": mask.copy(),
                "score": obj.get("score", 1.0)
            }
        else:
            merged[label]["mask"] = np.maximum(merged[label]["mask"], mask)

            x1, y1, x2, y2 = merged[label]["bbox"]
            bx1, by1, bx2, by2 = obj["bbox"]
            merged[label]["bbox"] = [
                min(x1, bx1),
                min(y1, by1),
                max(x2, bx2),
                max(y2, by2)
            ]
            merged[label]["score"] = max(merged[label]["score"], obj.get("score", 1.0))

    return list(merged.values())


def assign_layers(segmented_objects, depth_map, target_num_layers=6):
    """
    Assign each object to a layer based on median depth.
    Darker = farther, brighter = nearer for your current map.
    """
    items = []

    for idx, obj in enumerate(segmented_objects):
        vals = depth_map[obj["mask"].astype(bool)]
        med = float(np.median(vals)) if len(vals) else 1.0
        items.append((idx, med))

    # far to near
    items.sort(key=lambda x: x[1])

    assignments = {}
    n = max(len(items), 1)

    for rank, (idx, _) in enumerate(items):
        layer = min(int(rank * target_num_layers / n), target_num_layers - 1)
        assignments[idx] = layer

    return assignments


# ============================================================
# Increment 1 debug helpers
# These do NOT change pipeline behavior.
# They only save structured summaries so you can inspect stages.
# ============================================================

def _mask_area(mask):
    return int(mask.astype(bool).sum())


def _safe_bbox(obj):
    bbox = obj.get("bbox", None)
    if bbox is None:
        return None
    return [int(v) for v in bbox]


def summarize_boxes(boxes):
    """
    Save-friendly summary of Florence parsed boxes.
    """
    summary = []
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box["bbox"]
        summary.append({
            "index": i,
            "label": str(box["label"]),
            "bbox": [int(x1), int(y1), int(x2), int(y2)],
            "width": int(x2 - x1),
            "height": int(y2 - y1),
            "area": int((x2 - x1) * (y2 - y1))
        })
    return summary


def summarize_segmented(segmented, depth_map=None):
    """
    Save-friendly summary of segmented or merged objects.
    Does not write the raw mask into JSON.
    """
    summary = []
    for i, obj in enumerate(segmented):
        item = {
            "index": i,
            "label": str(obj.get("label", "unknown")),
            "bbox": _safe_bbox(obj),
            "score": float(obj.get("score", 1.0)),
            "mask_area": _mask_area(obj["mask"]) if "mask" in obj else None,
        }

        if depth_map is not None and "mask" in obj:
            vals = depth_map[obj["mask"].astype(bool)]
            item["depth_median"] = float(np.median(vals)) if len(vals) else None
            item["depth_mean"] = float(np.mean(vals)) if len(vals) else None

        summary.append(item)
    return summary


def summarize_layer_assignments(segmented_objects, layer_assignments, depth_map=None):
    """
    Summary of final object -> layer mapping.
    """
    rows = []
    for idx, obj in enumerate(segmented_objects):
        row = {
            "index": idx,
            "label": str(obj.get("label", "unknown")),
            "bbox": _safe_bbox(obj),
            "layer": int(layer_assignments.get(idx, -1)),
            "mask_area": _mask_area(obj["mask"]) if "mask" in obj else None,
        }

        if depth_map is not None and "mask" in obj:
            vals = depth_map[obj["mask"].astype(bool)]
            row["depth_median"] = float(np.median(vals)) if len(vals) else None
            row["depth_mean"] = float(np.mean(vals)) if len(vals) else None

        rows.append(row)

    rows = sorted(rows, key=lambda x: x["layer"])
    return rows


def save_debug_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def save_scene_debug_bundle(
    debug_dir,
    florence_raw_output=None,
    parsed_boxes=None,
    segmented=None,
    merged_segmented=None,
    layer_assignments=None,
    depth_map=None,
    extra_meta=None,
):
    """
    Save a bundle of debug JSON files for one pipeline run.

    This function is intentionally lightweight:
    - it does not change any logic
    - it only records what happened
    """
    debug_dir = Path(debug_dir)
    debug_dir.mkdir(parents=True, exist_ok=True)

    if florence_raw_output is not None:
        save_debug_json(debug_dir / "01_florence_raw_output.json", florence_raw_output)

    if parsed_boxes is not None:
        parsed_summary = summarize_boxes(parsed_boxes)
        save_debug_json(debug_dir / "02_parsed_boxes.json", parsed_summary)

    if segmented is not None:
        seg_summary = summarize_segmented(segmented, depth_map=depth_map)
        save_debug_json(debug_dir / "03_segmented_objects.json", seg_summary)

    if merged_segmented is not None:
        merged_summary = summarize_segmented(merged_segmented, depth_map=depth_map)
        save_debug_json(debug_dir / "04_merged_objects.json", merged_summary)

    if merged_segmented is not None and layer_assignments is not None:
        layer_summary = summarize_layer_assignments(
            merged_segmented,
            layer_assignments,
            depth_map=depth_map
        )
        save_debug_json(debug_dir / "05_layer_assignments.json", layer_summary)

    if extra_meta is not None:
        save_debug_json(debug_dir / "00_meta.json", extra_meta)