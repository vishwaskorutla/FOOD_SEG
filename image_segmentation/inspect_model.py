"""
Script to inspect the DeepLab checkpoint structure.
Run: python inspect_model.py
"""
import torch
import sys

model_path = r"c:\Users\Rakshith Reddy\OneDrive\Desktop\DL\image_segmentation\best_deeplabv3plus_mobilenetv2.pth.zip"

print("Loading checkpoint...")
try:
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    print(f"Checkpoint type: {type(checkpoint)}")

    if isinstance(checkpoint, dict):
        print(f"Keys in checkpoint: {list(checkpoint.keys())}")
        for key, value in checkpoint.items():
            if isinstance(value, dict):
                sample_keys = list(value.keys())[:5]
                print(f"  '{key}': dict with {len(value)} entries. Sample keys: {sample_keys}")
            elif isinstance(value, torch.Tensor):
                print(f"  '{key}': Tensor shape {value.shape}")
            else:
                print(f"  '{key}': {type(value)} = {value}")
    elif isinstance(checkpoint, torch.nn.Module):
        print("Checkpoint is a full model object!")
        print(f"Model class: {checkpoint.__class__.__name__}")
    else:
        print(f"Unknown checkpoint type: {type(checkpoint)}")
        print(checkpoint)

except Exception as e:
    print(f"Error loading checkpoint: {e}")
    import traceback
    traceback.print_exc()
