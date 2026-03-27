import json
from typing import Dict, Any, List, Tuple

from src.layer_planner import plan_layers_deterministic


def build_layer_planner_prompt(scene_repr: Dict[str, Any]) -> str:
    """
    Build a strict prompt asking OpenAI to return only a valid layer plan JSON.
    """
    objects = scene_repr.get("objects", [])

    object_lines = []
    for obj in objects:
        object_lines.append(
            f'- id: {obj["id"]}, label: {obj["label"]}, '
            f'depth_median: {obj["depth_median"]:.4f}, '
            f'centroid: {obj["centroid"]}'
        )

    object_block = "\n".join(object_lines)

    prompt = f"""
You are planning layered paper-theater scene construction.

Given the scene objects below, return a JSON object only.

Scene objects:
{object_block}

Return JSON with this exact structure:
{{
  "layers": [
    {{
      "name": "background_sky",
      "object_ids": ["obj_002"],
      "order": 0
    }}
  ]
}}

Rules:
1. Return JSON only. No markdown. No explanation.
2. Every object_id must refer to a real object from the provided list.
3. Every object must appear exactly once across all layers.
4. order must start at 0 and increase by 1 with no gaps.
5. Layers must go from farthest/background to nearest/foreground.
6. Prefer sensible paper-theater grouping.
7. Use readable layer names.
8. The generated mountain must stay aligned with the input mask position and shape.
9. Do not recenter or resize the mountain composition.
"""
    return prompt.strip()


def extract_json_from_response(response_text: str) -> Dict[str, Any]:
    """
    Try to parse JSON directly. Also supports responses wrapped in ```json blocks.
    """
    response_text = response_text.strip()

    # Direct JSON parse
    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        pass

    # Try fenced code block
    if "```" in response_text:
        parts = response_text.split("```")
        for part in parts:
            candidate = part.strip()
            if candidate.startswith("json"):
                candidate = candidate[4:].strip()
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                continue

    raise ValueError("Could not parse valid JSON from OpenAI response")


def validate_layer_plan(layer_plan: Dict[str, Any], scene_repr: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Validate planner output.
    """
    if not isinstance(layer_plan, dict):
        return False, "layer_plan_not_dict"

    if "layers" not in layer_plan:
        return False, "missing_layers"

    layers = layer_plan["layers"]
    if not isinstance(layers, list):
        return False, "layers_not_list"

    scene_object_ids = {obj["id"] for obj in scene_repr.get("objects", [])}
    planned_ids: List[str] = []

    for i, layer in enumerate(layers):
        if not isinstance(layer, dict):
            return False, f"layer_{i}_not_dict"

        if "name" not in layer:
            return False, f"layer_{i}_missing_name"
        if "object_ids" not in layer:
            return False, f"layer_{i}_missing_object_ids"
        if "order" not in layer:
            return False, f"layer_{i}_missing_order"

        if not isinstance(layer["name"], str):
            return False, f"layer_{i}_name_not_str"
        if not isinstance(layer["object_ids"], list):
            return False, f"layer_{i}_object_ids_not_list"
        if not isinstance(layer["order"], int):
            return False, f"layer_{i}_order_not_int"

        if layer["order"] != i:
            return False, f"layer_{i}_order_mismatch"

        for object_id in layer["object_ids"]:
            if object_id not in scene_object_ids:
                return False, f"invalid_object_id:{object_id}"
            planned_ids.append(object_id)

    if len(planned_ids) != len(scene_object_ids):
        return False, "object_count_mismatch"

    if set(planned_ids) != scene_object_ids:
        return False, "object_set_mismatch"

    if len(planned_ids) != len(set(planned_ids)):
        return False, "duplicate_object_ids"

    return True, "ok"


def plan_layers_with_openai(
    scene_repr: Dict[str, Any],
    openai_planner_fn,
    model_name: str,
) -> Dict[str, Any]:
    """
    Ask OpenAI for a layer plan, validate it, and fall back if needed.

    openai_planner_fn should take:
        prompt, model_name
    and return text
    """
    prompt = build_layer_planner_prompt(scene_repr)

    try:
        response_text = openai_planner_fn(prompt, model_name)
    except Exception as e:
        fallback = plan_layers_deterministic(scene_repr)
        return {
            "layer_plan": fallback,
            "used_openai": False,
            "status": f"openai_exception:{type(e).__name__}",
            "raw_response": None,
            "prompt": prompt,
        }

    try:
        parsed = extract_json_from_response(response_text)
    except Exception as e:
        fallback = plan_layers_deterministic(scene_repr)
        return {
            "layer_plan": fallback,
            "used_openai": False,
            "status": f"json_parse_failed:{type(e).__name__}",
            "raw_response": response_text,
            "prompt": prompt,
        }

    is_valid, status = validate_layer_plan(parsed, scene_repr)
    if not is_valid:
        fallback = plan_layers_deterministic(scene_repr)
        return {
            "layer_plan": fallback,
            "used_openai": False,
            "status": f"validation_failed:{status}",
            "raw_response": response_text,
            "prompt": prompt,
        }

    return {
        "layer_plan": parsed,
        "used_openai": True,
        "status": "openai_layer_plan_applied",
        "raw_response": response_text,
        "prompt": prompt,
    }