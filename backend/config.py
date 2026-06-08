import torch
from pathlib import Path
import torch 

# Device

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

# Base directory

BASE_DIR = Path(__file__).resolve().parent.parent

# Models

DEPTH_MODEL_PATH = BASE_DIR / "models" / "best_depthvision_model.pth"
YOLO_MODEL_PATH = BASE_DIR / "models" / "yolov8n.pt"

# Image Size

IMAGE_HEIGHT = 256
IMAGE_WIDTH = 512