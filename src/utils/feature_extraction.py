import torch
from PIL import Image
from torchvision import transforms

from src.core.config import settings
from src.core.models.osnet import osnet_x1_0


class FeatureExtractor:
    def __init__(self, img_size=settings.IMG_SIZE, device=None):
        # Set device (use GPU if available)
        self.device = (
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        # Initialize model
        self.model = osnet_x1_0(
            num_classes=767,
            loss="softmax",
            pretrained=True,
        )

        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((256, 128)),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        self.model.eval()
        self.img_size = img_size

    def preprocess(self, image):
        image = Image.fromarray(image).convert("RGB")
        image_tensor = self.transform(image)
        image_tensor = image_tensor.unsqueeze(0)
        # Move input tensor to the same device as model
        return image_tensor.to(self.device)

    @torch.no_grad()  # Disable gradient calculation during inference
    def inference(self, image):
        image = self.preprocess(image)
        output = self.model(image)
        return output


# Initialize with default device
feature_extractor = FeatureExtractor()
