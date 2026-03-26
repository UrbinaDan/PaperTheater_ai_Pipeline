import re
import json
from pathlib import Path

import numpy as np


CANONICAL_MAP = {
    # temple-like labels
    "pagoda": "temple",
    "building": "temple",
    "temple": "temple",
    "bridge": "temple",

    # foliage-like labels
    "plant": "foliage",
    "plants": "foliage",
    "bush": "foliage",
    "bushes": "foliage",
    "tree": "foliage",
    "foreground plant": "foliage",
    "foreground plants": "foliage",

    # stable labels
    "mountain": "mountain",
    "sky": "sky",
}


def canonicalize_label(label: str) -> str:
    label = label.lower().strip()
    return CANONICAL_MAP.get(label, label)


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

def _box_area(bbox):
    x1, y1, x2, y2 = bbox
    return max(0, x2 - x1) * max(0, y2 - y1)


def _intersection_area(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    return inter_w * inter_h


def _overlap_fraction_of_smaller(box_a, box_b):
    """
    How much of the smaller box is covered by the intersection?
    Returns a value in [0, 1].
    """
    inter = _intersection_area(box_a, box_b)
    area_a = _box_area(box_a)
    area_b = _box_area(box_b)
    smaller = min(area_a, area_b)

    if smaller <= 0:
        return 0.0

    return inter / smaller


def deduplicate_boxes(boxes, iou_thresh=0.60, contain_thresh=0.85):
    """
    Remove overlapping duplicate detections of the same canonical label.

    A box is considered duplicate if either:
    1. IoU with a kept box is high enough, OR
    2. The intersection covers most of the smaller box
       (useful for nested Florence duplicates)

    Strategy:
    - canonicalize labels
    - group by label
    - sort larger boxes first
    - keep larger representative boxes
    """
    if not boxes:
        return []

    normalized = []
    for box in boxes:
        normalized.append({
            "label": canonicalize_label(box["label"]),
            "bbox": box["bbox"]
        })

    grouped = {}
    for box in normalized:
        grouped.setdefault(box["label"], []).append(box)

    kept_all = []

    for label, label_boxes in grouped.items():
        label_boxes = sorted(
            label_boxes,
            key=lambda b: _box_area(b["bbox"]),
            reverse=True
        )

        kept_for_label = []

        for candidate in label_boxes:
            candidate_bbox = candidate["bbox"]
            is_duplicate = False

            for kept in kept_for_label:
                kept_bbox = kept["bbox"]

                iou = box_iou(candidate_bbox, kept_bbox)
                contain = _overlap_fraction_of_smaller(candidate_bbox, kept_bbox)

                if iou >= iou_thresh or contain >= contain_thresh:
                    is_duplicate = True
                    break

            if not is_duplicate:
                kept_for_label.append(candidate)

        kept_all.extend(kept_for_label)

    kept_all = sorted(
        kept_all,
        key=lambda b: (b["label"], b["bbox"][1], b["bbox"][0])
    )

    return kept_all

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

def summarize_stable_objects(stable_objects):
    """
    Save-friendly summary of stable merged objects.
    Raw masks are omitted from JSON.
    """
    rows = []
    for obj in stable_objects:
        rows.append({
            "id": obj["id"],
            "label": obj["label"],
            "bbox": obj["bbox"],
            "score": obj["score"],
            "area": obj["area"],
            "centroid": obj["centroid"],
            "depth_mean": obj["depth_mean"],
            "depth_median": obj["depth_median"],
        })
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
    stable_objects=None,
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

    if stable_objects is not None:
        stable_summary = summarize_stable_objects(stable_objects)
        save_debug_json(debug_dir / "06_stable_objects.json", stable_summary)

def _mask_centroid(mask):
    """
    Compute centroid of a binary mask as [x, y].
    Returns None if the mask is empty.
    """
    ys, xs = np.nonzero(mask.astype(bool))
    if len(xs) == 0 or len(ys) == 0:
        return None
    return [float(xs.mean()), float(ys.mean())]


def build_stable_merged_objects(merged_segmented, depth_map):
    """
    Convert merged segmented objects into a stable object list with:
    - stable id
    - label
    - bbox
    - mask
    - score
    - area
    - centroid
    - depth stats

    This does not change masks or labels.
    It just packages them into a more reusable structure.
    """
    stable_objects = []

    for i, obj in enumerate(merged_segmented):
        mask_bool = obj["mask"].astype(bool)
        mask_area = int(mask_bool.sum())

        depth_vals = depth_map[mask_bool]
        depth_mean = float(np.mean(depth_vals)) if len(depth_vals) else None
        depth_median = float(np.median(depth_vals)) if len(depth_vals) else None

        stable_objects.append({
            "id": f"obj_{i:03d}",
            "label": str(obj["label"]),
            "bbox": [int(v) for v in obj["bbox"]],
            "mask": obj["mask"],
            "score": float(obj.get("score", 1.0)),
            "area": mask_area,
            "centroid": _mask_centroid(obj["mask"]),
            "depth_mean": depth_mean,
            "depth_median": depth_median,
        })

    return stable_objects

def get_stable_object_by_id(stable_objects, object_id):
    """
    Return one stable object by id, or None if not found.
    """
    for obj in stable_objects:
        if obj["id"] == object_id:
            return obj
    return None