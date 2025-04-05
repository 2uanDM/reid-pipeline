import io

import numpy as np
import torch
from fastapi import FastAPI, UploadFile
from PIL import Image
from ray import serve
from torchvision import transforms

from src.core.config import settings
from src.core.models.osnet import osnet_x1_0

app = FastAPI()


@serve.deployment(
    num_replicas=4,
    ray_actor_options={"num_gpus": 0.25},
    max_ongoing_requests=1000,
)
@serve.ingress(app)
class FeatureExtractor:
    def __init__(self, img_size=settings.IMG_SIZE, device=None):
        self.device = (
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = osnet_x1_0(
            num_classes=767,
            loss="softmax",
            pretrained=True,
        ).to(self.device)
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

    def preprocess(self, images):
        # Handle both single image and batch
        if not isinstance(images, list):
            images = [images]

        processed_images = []
        for img in images:
            img = Image.fromarray(img).convert("RGB")
            img_tensor = self.transform(img)
            processed_images.append(img_tensor)

        batch_tensor = torch.stack(processed_images)
        return batch_tensor.to(self.device)

    @serve.batch(max_batch_size=16, batch_wait_timeout_s=0.1)
    @torch.no_grad()
    async def inference_batch(self, images_batch):
        # When batching, Ray will pass a list of the original arguments
        # We need to flatten this list before processing
        flat_images = []
        for image_item in images_batch:
            # Each item is a list containing one image from a request
            flat_images.extend(image_item)

        # Process batch of images
        batch_tensor = self.preprocess(flat_images)
        outputs = self.model(batch_tensor)
        return outputs.cpu().detach().numpy().tolist()

    @app.post("/embedding")
    async def extract(self, image: UploadFile = None):
        image_data = await image.read()
        image = Image.open(io.BytesIO(image_data))
        image = np.array(image)

        # Process single image through batched inference
        output = await self.inference_batch([image])  # Pass image as a list
        return output


# Bind the deployment
app = FeatureExtractor.bind()
