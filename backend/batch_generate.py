import modal
import os
import glob
import base64

# Import your existing Modal app and Segmentation class
from mammomap_api import app, SegmentationModel

@app.local_entrypoint()
def generate_demo_masks():
    # Initialize the model on the cloud
    seg_model = SegmentationModel()
    
    # Path to your frontend public folder (adjust if your folder structure differs)
    public_dir = os.path.abspath("../frontend/public")
    patients = ["patient320", "patient321", "patient322"]
    
    for pat in patients:
        pat_dir = os.path.join(public_dir, pat)
        if not os.path.exists(pat_dir):
            print(f"Directory not found: {pat_dir}")
            continue
            
        # Find all raw ROI images, skipping any existing masks
        image_files = glob.glob(os.path.join(pat_dir, "*.png"))
        raw_images = [img for img in image_files if "_mask" not in img]
        
        for img_path in raw_images:
            print(f"\nProcessing {os.path.basename(img_path)}...")
            
            with open(img_path, "rb") as f:
                img_bytes = f.read()
                
            # Send to Modal GPU
            result = seg_model.predict.remote(img_bytes)
            
            # Extract and decode the base64 mask
            mask_b64 = result.get("mask_b64")
            if not mask_b64:
                print("Error: No mask returned from model.")
                continue
                
            mask_bytes = base64.b64decode(mask_b64)
            
            # Save the new mask file
            out_path = img_path.replace(".png", "_mask.png")
            with open(out_path, "wb") as f:
                f.write(mask_bytes)
                
            print(f"Saved mask to: {out_path}")
            print(f"Metrics: {result['invasive_purity']:.1f}% Invasive")
            
    print("\nAll demo masks generated successfully!")