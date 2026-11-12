# v2 - custom ResNet50-CBAM model architecture with lazy imports
import os
import io
import base64
from firebase_functions import https_fn
from firebase_admin import initialize_app

initialize_app()

# Config
IMG_SIZE = 320
MODEL_PATH = os.path.join(os.path.dirname(__file__), "best_model_weights.pth")

# 104 Class Names for FoodSeg103
CLASS_NAMES = [
    "background", "candy", "egg tart", "french fries", "chocolate", "biscuit",
    "popcorn", "pudding", "ice cream", "cheese butter", "cake", "wine",
    "milkshake", "coffee", "juice", "milk", "tea", "almond", "red beans",
    "cashew", "dried cranberries", "soy", "walnut", "peanut", "egg", "apple",
    "date", "apricot", "avocado", "banana", "strawberry", "cherry", "blueberry",
    "raspberry", "mango", "olives", "peach", "lemon", "pear", "fig", "pineapple",
    "grape", "kiwi", "melon", "orange", "watermelon", "steak", "pork",
    "chicken duck", "sausage", "fried meat", "lamb", "sauce", "crab", "fish",
    "shellfish", "shrimp", "soup", "bread", "corn", "hamburg", "pizza",
    "hanamaki baozi", "wonton dumplings", "pasta", "noodles", "rice", "pie",
    "tofu", "eggplant", "potato", "garlic", "cauliflower", "tomato", "kelp",
    "seaweed", "spring onion", "rape", "ginger", "okra", "lettuce", "pumpkin",
    "cucumber", "white radish", "carrot", "asparagus", "bamboo shoots", "broccoli",
    "celery stick", "cilantro mint", "snow peas", "cabbage", "bean sprouts",
    "onion", "pepper", "green beans", "French beans", "king oyster mushroom",
    "shiitake", "enoki mushroom", "oyster mushroom", "white button mushroom",
    "salad", "other ingredients"
]

# Generate HSL-based distinct RGBA colors
def generate_class_colors():
    colors = {}
    colors[0] = (0, 0, 0, 0)  # background transparent
    for i in range(1, 104):
        hue = (i * 137.5) % 360
        saturation = 80
        lightness = 50
        
        # HSL to RGB
        h = hue / 360.0
        s = saturation / 100.0
        l = lightness / 100.0
        
        def hue_to_rgb(p, q, t):
            if t < 0: t += 1
            if t > 1: t -= 1
            if t < 1/6: return p + (q - p) * 6 * t
            if t < 1/2: return q
            if t < 2/3: return p + (q - p) * (2/3 - t) * 6
            return p
            
        q = l * (1 + s) if l < 0.5 else l + s - l * s
        p = 2 * l - q
        r = hue_to_rgb(p, q, h + 1/3)
        g = hue_to_rgb(p, q, h)
        b = hue_to_rgb(p, q, h - 1/3)
        
        colors[i] = (int(r * 255), int(g * 255), int(b * 255), 140)
    return colors

CLASS_COLORS = generate_class_colors()

# Setup helper modules for CBAM Attention ResNet50
_model = None

def _get_model():
    global _model
    if _model is None:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from torchvision.models import resnet50

        # Define custom classes inside the function to keep global namespace clean for cold start
        class ChannelAttention(nn.Module):
            def __init__(self, in_channels, reduction=16):
                super().__init__()
                self.avg_pool = nn.AdaptiveAvgPool2d(1)
                self.max_pool = nn.AdaptiveMaxPool2d(1)
                self.fc = nn.Sequential(
                    nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
                )
                self.sigmoid = nn.Sigmoid()
            
            def forward(self, x):
                avg_out = self.fc(self.avg_pool(x))
                max_out = self.fc(self.max_pool(x))
                return self.sigmoid(avg_out + max_out)

        class SpatialAttention(nn.Module):
            def __init__(self, kernel_size=7):
                super().__init__()
                self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
                self.sigmoid = nn.Sigmoid()
            
            def forward(self, x):
                avg_out = torch.mean(x, dim=1, keepdim=True)
                max_out, _ = torch.max(x, dim=1, keepdim=True)
                out = torch.cat([avg_out, max_out], dim=1)
                return self.sigmoid(self.conv(out))

        class CBAM(nn.Module):
            def __init__(self, in_channels, reduction=16):
                super().__init__()
                self.channel_att = ChannelAttention(in_channels, reduction)
                self.spatial_att = SpatialAttention()
            
            def forward(self, x):
                x = x * self.channel_att(x)
                x = x * self.spatial_att(x)
                return x

        class SkipConnectionAttention(nn.Module):
            def __init__(self, in_channels, reduction=8):
                super().__init__()
                self.cbam = CBAM(in_channels, reduction)
                self.conv = nn.Sequential(
                    nn.Conv2d(in_channels, in_channels, 1, bias=False),
                    nn.BatchNorm2d(in_channels),
                    nn.ReLU(inplace=True)
                )
            
            def forward(self, x):
                return self.conv(self.cbam(x))

        class ASPPConv(nn.Module):
            def __init__(self, in_channels, out_channels, dilation):
                super().__init__()
                self.conv = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True)
                )
            
            def forward(self, x):
                return self.conv(x)

        class ASPPPooling(nn.Module):
            def __init__(self, in_channels, out_channels):
                super().__init__()
                self.pool = nn.Sequential(
                    nn.AdaptiveAvgPool2d(1),
                    nn.Conv2d(in_channels, out_channels, 1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True)
                )
            
            def forward(self, x):
                size = x.shape[-2:]
                x = self.pool(x)
                return F.interpolate(x, size=size, mode='bilinear', align_corners=False)

        class ASPP(nn.Module):
            def __init__(self, in_channels, out_channels=256, dilations=[6, 12, 18]):
                super().__init__()
                modules = [
                    nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, 1, bias=False),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU(inplace=True)
                    )
                ]
                for dilation in dilations:
                    modules.append(ASPPConv(in_channels, out_channels, dilation))
                modules.append(ASPPPooling(in_channels, out_channels))
                
                self.convs = nn.ModuleList(modules)
                self.project = nn.Sequential(
                    nn.Conv2d(len(self.convs) * out_channels, out_channels, 1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.5)
                )
                self.attention = CBAM(out_channels, reduction=16)
            
            def forward(self, x):
                res = [conv(x) for conv in self.convs]
                res = torch.cat(res, dim=1)
                x = self.project(res)
                return self.attention(x)

        class Decoder(nn.Module):
            def __init__(self, low_level_channels, num_classes):
                super().__init__()
                self.low_level_conv = nn.Sequential(
                    nn.Conv2d(low_level_channels, 48, 1, bias=False),
                    nn.BatchNorm2d(48),
                    nn.ReLU(inplace=True)
                )
                self.skip_attention = SkipConnectionAttention(48, reduction=4)
                self.decoder = nn.Sequential(
                    nn.Conv2d(256 + 48, 256, 3, padding=1, bias=False),
                    nn.BatchNorm2d(256),
                    nn.ReLU(inplace=True),
                    CBAM(256, reduction=16),
                    nn.Conv2d(256, 256, 3, padding=1, bias=False),
                    nn.BatchNorm2d(256),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.3),
                    nn.Conv2d(256, num_classes, 1)
                )
            
            def forward(self, x, low_level_feat):
                low_level_feat = self.skip_attention(self.low_level_conv(low_level_feat))
                x = F.interpolate(x, size=low_level_feat.shape[-2:], mode='bilinear', align_corners=False)
                x = torch.cat([x, low_level_feat], dim=1)
                return self.decoder(x)

        class FoodSegmentationModel(nn.Module):
            def __init__(self, num_classes=104, pretrained_backbone=False):
                super().__init__()
                resnet = resnet50(pretrained=False)
                
                self.conv1 = resnet.conv1
                self.bn1 = resnet.bn1
                self.relu = resnet.relu
                self.maxpool = resnet.maxpool
                
                self.layer1 = resnet.layer1
                self.attention1 = CBAM(256, reduction=16)
                
                self.layer2 = resnet.layer2
                self.attention2 = CBAM(512, reduction=16)
                
                self.layer3 = resnet.layer3
                self.attention3 = CBAM(1024, reduction=16)
                
                self.layer4 = self._make_layer4_dilated(resnet.layer4)
                self.attention4 = CBAM(2048, reduction=16)
                
                self.aspp = ASPP(2048, 256, dilations=[6, 12, 18])
                self.decoder = Decoder(low_level_channels=256, num_classes=num_classes)
            
            def _make_layer4_dilated(self, layer4):
                for n, m in layer4.named_modules():
                    if 'conv2' in n:
                        m.dilation = (2, 2)
                        m.padding = (2, 2)
                        m.stride = (1, 1)
                    elif 'downsample.0' in n:
                        m.stride = (1, 1)
                return layer4
            
            def forward(self, x):
                input_size = x.shape[-2:]
                x = self.conv1(x)
                x = self.bn1(x)
                x = self.relu(x)
                x = self.maxpool(x)
                
                x = self.attention1(self.layer1(x))
                low_level_feat = x
                
                x = self.attention2(self.layer2(x))
                x = self.attention3(self.layer3(x))
                x = self.attention4(self.layer4(x))
                
                x = self.aspp(x)
                x = self.decoder(x, low_level_feat)
                
                return F.interpolate(x, size=input_size, mode='bilinear', align_corners=False)

        print(f"Loading custom ResNet50-CBAM model from {MODEL_PATH}...")
        _model = FoodSegmentationModel(num_classes=104, pretrained_backbone=False)
        checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
        _model.load_state_dict(checkpoint)
        _model.eval()
        print("Model loaded successfully!")
    return _model

@https_fn.on_call(memory=2048, timeout_sec=540)
def segment_food(req: https_fn.CallableRequest) -> dict:
    """
    Firebase callable function for food segmentation.
    Accepts: {"image": "<base64_jpeg_or_png>"}
    Returns: {"mask": "<base64_png_overlay>", "detected_ingredients": [...]}
    """
    data = req.data
    b64_image = data.get("image")
    if not b64_image:
        raise https_fn.HttpsError(
            code=https_fn.FunctionsErrorCode.INVALID_ARGUMENT,
            message="Missing 'image' field."
        )

    try:
        import torch
        import torchvision.transforms as transforms
        import numpy as np
        from PIL import Image

        # Decode base64 image
        image_bytes = base64.b64decode(b64_image)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        orig_width, orig_height = image.size

        # Resize for inference
        img_resized = image.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)

        # Normalize
        normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        tensor_img = normalize(img_resized).unsqueeze(0)

        # Predict
        net = _get_model()
        with torch.no_grad():
            output = net(tensor_img)
            pred_mask = torch.argmax(output, dim=1).squeeze().numpy()

        # Find unique classes and calculate percentages
        unique_classes, class_counts = np.unique(pred_mask, return_counts=True)
        total_pixels = IMG_SIZE * IMG_SIZE
        
        detected_ingredients = []
        for cls_id, count in zip(unique_classes, class_counts):
            cls_id = int(cls_id)
            if cls_id == 0:
                continue
            
            percentage = float(count / total_pixels)
            color_rgba = CLASS_COLORS[cls_id]
            color_hex = f"#{color_rgba[0]:02x}{color_rgba[1]:02x}{color_rgba[2]:02x}"
            
            detected_ingredients.append({
                "class_id": cls_id,
                "name": CLASS_NAMES[cls_id].title(),
                "color": color_hex,
                "percentage": round(percentage * 100, 2)
            })

        detected_ingredients.sort(key=lambda x: x["percentage"], reverse=True)

        # Resize prediction mask back to original resolution
        mask_pil = Image.fromarray(pred_mask.astype(np.uint8))
        mask_resized = mask_pil.resize((orig_width, orig_height), Image.NEAREST)
        mask_array = np.array(mask_resized)

        # Create RGBA colored mask
        rgba_mask = np.zeros((orig_height, orig_width, 4), dtype=np.uint8)
        for cls_id in unique_classes:
            cls_id = int(cls_id)
            if cls_id == 0:
                continue
            color = CLASS_COLORS[cls_id]
            rgba_mask[mask_array == cls_id] = color

        # Save to base64 PNG
        overlay_pil = Image.fromarray(rgba_mask, mode="RGBA")
        buffer = io.BytesIO()
        overlay_pil.save(buffer, format="PNG")
        overlay_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        return {
            "detected_ingredients": detected_ingredients,
            "mask": overlay_base64
        }

    except Exception as e:
        raise https_fn.HttpsError(
            code=https_fn.FunctionsErrorCode.INTERNAL,
            message=f"Prediction error: {str(e)}"
        )
