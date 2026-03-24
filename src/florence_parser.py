import torch
from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image


class FlorenceParser:
    def __init__(self, model_name="microsoft/Florence-2-base"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.processor = AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=True
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.float32
        ).to(self.device)

    def run_task(self, image_pil, task_prompt):
        inputs = self.processor(
            text=task_prompt,
            images=image_pil,
            return_tensors="pt"
        )

        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].float()

        generated_ids = self.model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=512,
            num_beams=3
        )

        generated_text = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=False
        )[0]

        parsed = self.processor.post_process_generation(
            generated_text,
            task=task_prompt,
            image_size=(image_pil.width, image_pil.height)
        )
        return parsed

    def get_dense_caption(self, image_rgb):
        image_pil = Image.fromarray(image_rgb)
        return self.run_task(image_pil, "<MORE_DETAILED_CAPTION>")

    def get_open_vocab_detection(self, image_rgb, query_text):
        image_pil = Image.fromarray(image_rgb)
        task = f"<OPEN_VOCABULARY_DETECTION>{query_text}"
        return self.run_task(image_pil, task)