from dataclasses import dataclass
from pathlib import Path

@dataclass
class Paths:
    root: Path = Path("/content/paper-theater-ai")
    input_dir: Path = root / "data" / "input"
    inter_dir: Path = root / "data" / "intermediate"
    output_dir: Path = root / "data" / "output"

    masks_dir: Path = inter_dir / "masks"
    depth_dir: Path = inter_dir / "depth"
    objects_dir: Path = inter_dir / "objects"
    completed_dir: Path = inter_dir / "completed"
    previews_dir: Path = output_dir / "previews"
    layers_svg_dir: Path = output_dir / "layers_svg"
    compare_dir: Path = output_dir / "compare"

@dataclass
class PipelineConfig:
    image_max_side: int = 1536
    target_num_layers: int = 6
    min_component_area: int = 500
    smooth_kernel: int = 5
    simplify_tolerance: float = 2.0
    sam2_checkpoint: str = "/content/sam2/checkpoints/sam2_hiera_small.pt"
    sam2_config: str = "sam2_hiera_s.yaml"
    florence_model: str = "microsoft/Florence-2-base"
    depth_encoder: str = "vits"   # commercial-safe smaller option is practical to start with
    openai_model: str = "gpt-image-1"