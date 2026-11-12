import os
import io
import base64
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.models import resnet50
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

app = FastAPI(title="Food Image Segmentation Service")

# VERY IMPORTANT: This allows your browser test client to talk to the Python server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

def generate_class_colors():
    colors = {}
    colors[0] = (0, 0, 0, 0)
    for i in range(1, 104):
        hue = (i * 137.5) % 360
        saturation = 80
        lightness = 50
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

class FoodSegmentationModel(nn.Module):
    def __init__(self, num_classes=104):
        super().__init__()
        resnet = resnet50(weights=None)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        self.head = nn.Sequential(
            nn.Conv2d(2048, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, num_classes, 1)
        )
    def forward(self, x):
        size = x.shape[-2:]
        x = self.backbone(x)
        x = self.head(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "best_model_weights.pth")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = None

def get_model():
    global model
    if model is None:
        model = FoodSegmentationModel(num_classes=104)
        if os.path.exists(MODEL_PATH):
            checkpoint = torch.load(MODEL_PATH, map_location=device)
            model.load_state_dict(checkpoint, strict=False)
        model.to(device).eval()
    return model

normalize_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        orig_width, orig_height = image.size
        input_size = 320
        img_resized = image.resize((input_size, input_size), Image.BILINEAR)
        tensor_img = normalize_transform(img_resized).unsqueeze(0).to(device)

        net = get_model()
        with torch.no_grad():
            output = net(tensor_img)
            pred_mask = torch.argmax(output, dim=1).squeeze().cpu().numpy()

        unique_classes, class_counts = np.unique(pred_mask, return_counts=True)
        total_pixels = input_size * input_size
        detected_ingredients = []
        for cls_id, count in zip(unique_classes, class_counts):
            cls_id = int(cls_id)
            if cls_id == 0: continue
            percentage = float(count / total_pixels)
            color_rgba = CLASS_COLORS[cls_id]
            color_hex = f"#{color_rgba[0]:02x}{color_rgba[1]:02x}{color_rgba[2]:02x}"
            detected_ingredients.append({
                "class_id": cls_id, "name": CLASS_NAMES[cls_id].title(),
                "color": color_hex, "percentage": round(percentage * 100, 2)
            })
        detected_ingredients.sort(key=lambda x: x["percentage"], reverse=True)

        mask_pil = Image.fromarray(pred_mask.astype(np.uint8)).resize((orig_width, orig_height), Image.NEAREST)
        mask_array = np.array(mask_pil)
        rgba_mask = np.zeros((orig_height, orig_width, 4), dtype=np.uint8)
        for cls_id in unique_classes:
            if int(cls_id) == 0: continue
            rgba_mask[mask_array == cls_id] = CLASS_COLORS[int(cls_id)]

        buffer = io.BytesIO()
        Image.fromarray(rgba_mask, mode="RGBA").save(buffer, format="PNG")
        overlay_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        return {"detected_ingredients": detected_ingredients, "mask": overlay_base64}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8001)
