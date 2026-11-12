"""
Extract only the model weights (model_state_dict) from the full checkpoint.
This reduces the file size from ~490MB to ~150MB by removing optimizer states
and training metadata, making it suitable for deployment to Firebase Cloud Functions.
"""
import torch
import os

src_path = r"C:\Users\Rakshith Reddy\Downloads\best_model.pth"
dest_path = r"c:\Users\Rakshith Reddy\OneDrive\Desktop\DL\image_segmentation\best_model_weights.pth"

print(f"Loading checkpoint from {src_path}...")
checkpoint = torch.load(src_path, map_location="cpu", weights_only=False)

# Get the model weights dictionary
if "model_state_dict" in checkpoint:
    model_state = checkpoint["model_state_dict"]
    print("Found 'model_state_dict' in checkpoint.")
elif "model" in checkpoint:
    model_state = checkpoint["model"]
    print("Found 'model' in checkpoint.")
else:
    model_state = checkpoint
    print("No specific dictionary wrapper found; treating checkpoint as model state dict.")

print(f"Saving optimized model weights to {dest_path}...")
torch.save(model_state, dest_path)

orig_size = os.path.getsize(src_path) / (1024 * 1024)
new_size = os.path.getsize(dest_path) / (1024 * 1024)
print(f"Success! Original size: {orig_size:.2f} MB, Optimized size: {new_size:.2f} MB")
