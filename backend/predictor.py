import torch
from PIL import Image
from torchvision import transforms

from backend.model import DepthVisionNet

from backend.config import (
    DEVICE,
    DEPTH_MODEL_PATH,
    IMAGE_HEIGHT,
    IMAGE_WIDTH
)

# Load model once
depth_model = DepthVisionNet()

depth_model.load_state_dict(
    torch.load(
        DEPTH_MODEL_PATH,
        map_location=DEVICE
    )
)

depth_model.to(DEVICE)
depth_model.eval()


def preprocess_image(image_path):

    image = Image.open(
        image_path
    ).convert("RGB")

    transform = transforms.Compose([
        transforms.Resize(
            (
                IMAGE_HEIGHT,
                IMAGE_WIDTH
            )
        ),
        transforms.ToTensor()
    ])

    image_tensor = transform(image)

    image_tensor = (
        image_tensor
        .unsqueeze(0)
        .to(DEVICE)
    )

    return image_tensor


def predict_depth(image_path):

    image_tensor = preprocess_image(
        image_path
    )

    with torch.no_grad():

        depth_map = depth_model(
            image_tensor
        )

    depth_map = (
        depth_map
        .squeeze()
        .cpu()
        .numpy()
    )

    return depth_map