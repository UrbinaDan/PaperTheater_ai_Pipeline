import sys
import numpy as np
import torch

sys.path.append("/content/sam2")

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

class SAM2Segmenter:
    def __init__(self, cfg_file="sam2_hiera_s.yaml", checkpoint="/content/sam2/checkpoints/sam2_hiera_small.pt"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = build_sam2(cfg_file, checkpoint, device=self.device)
        self.predictor = SAM2ImagePredictor(self.model)

    def set_image(self, image_rgb):
        self.predictor.set_image(image_rgb)

    def segment_box(self, box_xyxy):
        box = np.array(box_xyxy, dtype=np.float32)
        masks, scores, _ = self.predictor.predict(
            point_coords=None,
            point_labels=None,
            box=box[None, :],
            multimask_output=False
        )
        return masks[0].astype(np.uint8), float(scores[0])

    def segment_boxes(self, image_rgb, boxes):
        self.set_image(image_rgb)
        results = []
        for b in boxes:
            mask, score = self.segment_box(b["bbox"])
            results.append({
                "label": b["label"],
                "bbox": b["bbox"],
                "mask": mask,
                "score": score
            })
        return results