import torch
import segmentation_models_pytorch as smp

model_path = r"c:\Users\Rakshith Reddy\OneDrive\Desktop\DL\image_segmentation\best_deeplabv3plus_mobilenetv2.pth.zip"
checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
state_dict = checkpoint["model"]

print("Attempting to load weights into smp.DeepLabV3Plus with mobilenet_v2 backbone...")
model = smp.DeepLabV3Plus(
    encoder_name="mobilenet_v2",
    encoder_weights=None,
    in_channels=3,
    classes=104
)

# Try loading state dict
try:
    missing, unexpected = model.load_state_dict(state_dict, strict=True)
    print("Strict load successful!")
    print(f"Missing keys: {missing}")
    print(f"Unexpected keys: {unexpected}")
except Exception as e:
    print(f"Strict load failed: {e}")
    print("Trying non-strict load...")
    try:
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print("Non-strict load completed.")
        print(f"Missing keys (count: {len(missing)}): {missing[:10]}")
        print(f"Unexpected keys (count: {len(unexpected)}): {unexpected[:10]}")
    except Exception as ex:
        print(f"Non-strict load also failed: {ex}")
