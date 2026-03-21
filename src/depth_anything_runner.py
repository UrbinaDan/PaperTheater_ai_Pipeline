import sys
import cv2
import torch
import numpy as np

sys.path.append("/content/Depth-Anything-V2")

from depth_anything_v2.dpt import DepthAnythingV2

class DepthRunner:
    def __init__(self, encoder="vits"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        model_configs = {
            "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
            "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
            "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
        }

        self.model = DepthAnythingV2(**model_configs[encoder])

        ckpt_map = {
            "vits": "/content/Depth-Anything-V2/checkpoints/depth_anything_v2_vits.pth",
            "vitb": "/content/Depth-Anything-V2/checkpoints/depth_anything_v2_vitb.pth",
            "vitl": "/content/Depth-Anything-V2/checkpoints/depth_anything_v2_vitl.pth",
        }

        self.model.load_state_dict(torch.load(ckpt_map[encoder], map_location=self.device))
        self.model = self.model.to(self.device).eval()

    def infer(self, image_rgb):
        depth = self.model.infer_image(image_rgb)
        depth = depth.astype(np.float32)
        depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
        return depth