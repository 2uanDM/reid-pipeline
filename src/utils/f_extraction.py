from PIL import Image
from torchvision import transforms

from src.assets.models.LightMBN.lmbn_n import *
from src.assets.models.osnet.osnet import *
from src.core.config import settings

"""
model_factory = {
    # image classification models
    'resnet18': resnet18,
    'resnet34': resnet34,
    'resnet50': resnet50,
    'resnet101': resnet101,
    'resnet152': resnet152,
    'resnext50_32x4d': resnext50_32x4d,
    'resnext101_32x8d': resnext101_32x8d,
    'resnet50_fc512': resnet50_fc512,
    'se_resnet50': se_resnet50,
    'se_resnet50_fc512': se_resnet50_fc512,
    'se_resnet101': se_resnet101,
    'se_resnext50_32x4d': se_resnext50_32x4d,
    'se_resnext101_32x4d': se_resnext101_32x4d,
    'densenet121': densenet121,
    'densenet169': densenet169,
    'densenet201': densenet201,
    'densenet161': densenet161,
    'densenet121_fc512': densenet121_fc512,
    'inceptionresnetv2': inceptionresnetv2,
    'inceptionv4': inceptionv4,
    'xception': xception,
    'resnet50_ibn_a': resnet50_ibn_a,
    'resnet50_ibn_b': resnet50_ibn_b,
    # lightweight models
    'nasnsetmobile': nasnetamobile,
    'mobilenetv2_x1_0': mobilenetv2_x1_0,
    'mobilenetv2_x1_4': mobilenetv2_x1_4,
    'shufflenet': shufflenet,
    'squeezenet1_0': squeezenet1_0,
    'squeezenet1_0_fc512': squeezenet1_0_fc512,
    'squeezenet1_1': squeezenet1_1,
    'shufflenet_v2_x0_5': shufflenet_v2_x0_5,
    'shufflenet_v2_x1_0': shufflenet_v2_x1_0,
    'shufflenet_v2_x1_5': shufflenet_v2_x1_5,
    'shufflenet_v2_x2_0': shufflenet_v2_x2_0,
    # reid-specific models
    'mudeep': MuDeep,
    'resnet50mid': resnet50mid,
    'hacnn': HACNN,
    'pcb_p6': pcb_p6,
    'pcb_p4': pcb_p4,
    'mlfn': mlfn,
    'osnet_x1_0': osnet_x1_0,
    'osnet_x0_75': osnet_x0_75,
    'osnet_x0_5': osnet_x0_5,
    'osnet_x0_25': osnet_x0_25,
    'osnet_ibn_x1_0': osnet_ibn_x1_0,
    'osnet_ain_x1_0': osnet_ain_x1_0,
    'osnet_ain_x0_75': osnet_ain_x0_75,
    'osnet_ain_x0_5': osnet_ain_x0_5,
    'osnet_ain_x0_25': osnet_ain_x0_25
}
"""

model = osnet_x1_0(num_classes=767, loss="softmax", pretrained=True, use_gpu=False)
transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize((256, 128)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


class FeatureExtractor:
    def __init__(
        self,
        model_path=settings.MODEL_PATH,
        img_size=settings.IMG_SIZE,
        num_classes=751,
        pretrained=True,
        use_gpu=False,
    ):
        weight_path = "src/assets/models/LightMBN/lmbn_n_cuhk03_d.pth"
        self.model = model
        # state_dict = torch.load(weight_path, map_location=torch.device('cuda' if use_gpu else 'cpu'), weights_only=True)
        # self.model.load_state_dict(state_dict, strict=False)
        self.model.eval()
        self.img_size = img_size

    def preprocess(self, image):
        image = Image.fromarray(image).convert("RGB")
        image_tensor = transform(image)
        image_tensor = image_tensor.unsqueeze(0)
        return image_tensor

    def inference(self, image):
        image = self.preprocess(image)
        output = self.model(image)
        return output


feature_extractor = FeatureExtractor()
