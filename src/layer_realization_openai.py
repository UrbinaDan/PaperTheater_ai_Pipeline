import src.layer_renderer
importlib.reload(src.layer_renderer)
import src.layer_realization_openai
importlib.reload(src.layer_realization_openai)

from src.layer_context_builder import build_layer_contexts
from src.layer_prompt_builder import build_layer_realization_prompt
from src.layer_realization_openai import build_layer_realization_prompt
from src.layer_renderer import build_object_mask_map

object_mask_map = build_object_mask_map(openai_results)
print("Object mask ids:", list(object_mask_map.keys()))

object_mask_map = build_object_mask_map(openai_results)
print("Object mask ids:", list(object_mask_map.keys()))

layer_contexts = build_layer_contexts(
    scene_repr=scene_repr,
    layer_plan=hybrid_layer_plan,
    object_mask_map=object_mask_map,
)

for ctx in layer_contexts:
    print(
        ctx["order"],
        ctx["layer_name"],
        ctx["object_ids"],
        "| labels =", ctx["labels"],
        "| ownership_bbox =", ctx["ownership_bbox"],
        "| visible_bbox =", ctx["visible_bbox"],
        "| front =", ctx["front_layer_names"],
    )

target_layer_context = None

for ctx in layer_contexts:
    if "mountain" in ctx["layer_name"] or "mountain" in ctx["labels"]:
        target_layer_context = ctx
        break

if target_layer_context is None:
    raise ValueError("Could not find mountain layer context")

print("Selected layer:", target_layer_context["layer_name"])

target_prompt = build_layer_realization_prompt(target_layer_context)

print(target_prompt)

from PIL import Image
import numpy as np


def fit_image_to_target_crop_preserve_aspect(generated: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """
    Resize while preserving aspect ratio, then center-crop to target size.
    This avoids stretching distortion.
    """
    gen_h, gen_w = generated.shape[:2]

    scale = max(target_w / gen_w, target_h / gen_h)
    new_w = int(round(gen_w * scale))
    new_h = int(round(gen_h * scale))

    generated_pil = Image.fromarray(generated)
    resized_pil = generated_pil.resize((new_w, new_h), Image.Resampling.LANCZOS)
    resized = np.array(resized_pil)

    start_x = max(0, (new_w - target_w) // 2)
    start_y = max(0, (new_h - target_h) // 2)

    cropped = resized[start_y:start_y + target_h, start_x:start_x + target_w]

    # Safety fallback
    if cropped.shape[0] != target_h or cropped.shape[1] != target_w:
        canvas = np.zeros((target_h, target_w, 3), dtype=resized.dtype)
        h = min(target_h, cropped.shape[0])
        w = min(target_w, cropped.shape[1])
        canvas[:h, :w] = cropped[:h, :w]
        cropped = canvas

    return cropped

experimental_output_dir = "/content/PaperTheater_ai_Pipeline/data/intermediate/experimental_increment_12"

from src.layer_realization_openai import realize_single_layer_experimental

realization_result = realize_single_layer_experimental(
    image=image,
    layer_context=target_layer_context,
    prompt=target_prompt,
    output_dir=experimental_output_dir,
    openai_realize_fn=experimental_layer_openai_fn,
    model_name=cfg.openai_model,
    pad=48,
)

print("Realization result:")
for k, v in realization_result.items():
    print(k, ":", v)

from PIL import Image
import numpy as np

def experimental_layer_openai_fn(
    image_crop,
    ownership_mask_crop,
    front_occlusion_mask_crop,
    layer_context,
    prompt,
    model_name,
):
    try:
        generated = openai_edit(
            image_crop,
            ownership_mask_crop,
            layer_context["labels"][0],
            model_name,
            prompt=prompt
        )
    except TypeError:
        generated = openai_edit(
            image_crop,
            ownership_mask_crop,
            layer_context["labels"][0],
            model_name
        )

    target_height, target_width = image_crop.shape[:2]

    if generated.shape[:2] != (target_height, target_width):
        generated = fit_image_to_target_crop_preserve_aspect(
            generated,
            target_h=target_height,
            target_w=target_width
        )

    return generated

experimental_layer_openai_fn()

from PIL import Image
import matplotlib.pyplot as plt

paths_to_show = [
    realization_result["input_crop_path"],
    realization_result["ownership_mask_crop_path"],
    realization_result["front_occlusion_mask_crop_path"],
    realization_result["visible_mask_crop_path"],
    realization_result["generated_crop_path"],
    realization_result["generated_visible_crop_path"],
    realization_result["original_visible_crop_path"],
]

for p in paths_to_show:
    img = Image.open(p)
    plt.figure(figsize=(8, 6))
    plt.imshow(img)
    plt.title(p.split("/")[-1])
    plt.axis("off")
    plt.show()