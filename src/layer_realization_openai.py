from typing import Dict, Any, List


def format_labels(labels: List[str]) -> str:
    if not labels:
        return "unknown"
    return ", ".join(labels)


def build_layer_realization_prompt(layer_context: Dict[str, Any]) -> str:
    """
    Build a strong layer-specific realization prompt.
    """
    layer_name = layer_context["layer_name"]
    labels = format_labels(layer_context["labels"])
    caption = layer_context["scene_caption"]
    order = layer_context["order"]
    front_layers = layer_context["front_layer_names"]
    rear_layers = layer_context["rear_layer_names"]
    depth_summary = layer_context["depth_summary"]

    front_text = ", ".join(front_layers) if front_layers else "none"
    rear_text = ", ".join(rear_layers) if rear_layers else "none"

    prompt = f"""
You are generating a single visual layer for a paper-theater scene.

Global scene description:
{caption}

Target layer:
- layer_name: {layer_name}
- semantic labels: {labels}
- order: {order}
- rear layers behind this one: {rear_text}
- front layers in front of this one: {front_text}

Depth hints:
- average depth median: {depth_summary["depth_median_mean"]:.4f}
- min depth median: {depth_summary["depth_median_min"]:.4f}
- max depth median: {depth_summary["depth_median_max"]:.4f}

Instructions:
1. Realize only the target layer content.
2. Preserve the original scene identity, composition, mood, and style.
3. Do not add unrelated objects.
4. Treat front-layer reserved regions as valid foreground occlusion.
5. Complete the layer naturally behind the layers in front of it.
6. Keep the result visually consistent with a Japanese scenic landscape at sunset.
7. Keep the layer coherent as a standalone cut layer for paper theater.
"""
    return prompt.strip()