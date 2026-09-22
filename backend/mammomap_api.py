import modal
import io
import base64
import tempfile
import numpy as np
import math
from typing import Dict

# 1. Define the Application and Image
app = modal.App("mammomap-inference-api")

# Add the local file into the container image directly
ml_image = (
    modal.Image.from_dockerfile("Dockerfile")
    .add_local_file("inference_script.py", remote_path="/root/inference_script.py")
)

# Secrets and Volumes
hf_secret = modal.Secret.from_name("huggingface-secret")
weights_volume = modal.Volume.from_name("weights-volume", create_if_missing=True)
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)

# ─── MODEL 1: SEGMENTATION (Virchow2) ──────────────────────────────
@app.cls(
    image=ml_image, 
    gpu="T4", 
    secrets=[hf_secret],
    volumes={
        "/weights": weights_volume,
        "/root/.cache/huggingface": hf_cache
    }
)
class SegmentationModel:
    @modal.enter()
    def load(self):
        import sys
        import torch
        
        # Bypass the Dockerfile PYTHONPATH override to find our script
        if "/root" not in sys.path:
            sys.path.append("/root")
        
        import inference_script
        
        self.device = torch.device("cuda")
        
        # Override paths to point to the mounted Modal volume
        inference_script.ADAPTER_DIR = "/weights/checkpoint_epoch_18_adapter"
        inference_script.DECODER_PATH = "/weights/checkpoint_epoch_18_decoder.pth"
        
        self.model = inference_script.load_model()
        self.model_loaded = True

    @modal.method()
    def predict(self, image_bytes: bytes) -> Dict:
        import sys
        if "/root" not in sys.path:
            sys.path.append("/root")
            
        import inference_script
        from PIL import Image
        
        with tempfile.NamedTemporaryFile(suffix=".png", delete=True) as temp_img:
            temp_img.write(image_bytes)
            temp_img.flush()
            
            mask = inference_script.process_image(self.model, temp_img.name)

        total_pixels = mask.size
        purity = (np.sum(mask == 1) / total_pixels) * 100
        non_invasive = (np.sum(mask == 2) / total_pixels) * 100
        other = (np.sum(mask == 3) / total_pixels) * 100
        necrosis = (np.sum(mask == 4) / total_pixels) * 100

        # Map class indices to RGB colors
        palette = np.array([
            [0, 0, 0],       # 0: Background
            [255, 50, 50],   # 1: Invasive (Red)
            [50, 255, 50],   # 2: Non-Invasive (Green)
            [50, 150, 255],  # 3: Other (Blue)
            [255, 200, 0],   # 4: Necrosis (Yellow)
        ], dtype=np.uint8)

        color_mask = palette[mask]
        mask_img = Image.fromarray(color_mask)
        buffered = io.BytesIO()
        mask_img.save(buffered, format="PNG")
        mask_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        return {
            "invasive_purity": purity,
            "non_invasive_ratio": non_invasive,
            "other_ratio": other,
            "necrosis_ratio": necrosis,
            "mask_b64": mask_b64
        }

# ─── UNIFIED WEB ENDPOINT ──────────────────────────────────────────
@app.function(image=ml_image)
@modal.fastapi_endpoint(method="POST")
def analyze_roi(payload: dict):
    image_bytes = base64.b64decode(payload["image_b64"])
    
    # Only run the segmentation model
    seg_model = SegmentationModel()
    seg_results = seg_model.predict.remote(image_bytes)

    return {
        "status": "success",
        "segmentation_results": seg_results
    }