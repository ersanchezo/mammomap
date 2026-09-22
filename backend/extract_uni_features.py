import modal
import os
import glob

# 1. Image definition with required libraries
ml_image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install("torch", "torchvision", "timm", "h5py", "Pillow", "numpy", "huggingface_hub")
)

app = modal.App("mammomap-feature-extraction")
hf_secret = modal.Secret.from_name("huggingface-secret")
weights_volume = modal.Volume.from_name("weights-volume")

@app.cls(
    image=ml_image,
    gpu="T4",
    secrets=[hf_secret],
    volumes={"/weights": weights_volume}
)
class UniFeatureExtractor:
    @modal.enter()
    def load(self):
        import torch
        import timm
        from torchvision import transforms
        
        print("Loading UNI Foundation Model...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model = timm.create_model(
            "hf-hub:MahmoodLab/UNI", 
            pretrained=False,
            init_values=1e-5, 
            dynamic_img_size=True
        )
        self.model.load_state_dict(torch.load("/weights/uni/pytorch_model.bin", map_location="cpu"))
        self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

    @modal.method()
    def process_roi(self, img_bytes: bytes, patch_size: int = 224, stride: int = 32) -> bytes:
        import io
        import h5py
        import torch
        import numpy as np
        from PIL import Image
        
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        width, height = img.size
        
        coords = []
        patches = []
        
        # DENSE SLIDING WINDOW: Step by `stride` instead of `patch_size`
        for y in range(0, height - patch_size + 1, stride):
            for x in range(0, width - patch_size + 1, stride):
                patch = img.crop((x, y, x + patch_size, y + patch_size))
                
                # Record the CENTER of the patch for accurate UI heatmap placement
                cx = x + (patch_size // 2)
                cy = y + (patch_size // 2)
                coords.append([cx, cy])
                
                patches.append(self.transform(patch))
                
        if not patches:
            return b""

        # BATCH PROCESSING: Prevent GPU Out-Of-Memory errors
        batch_size = 64
        all_features = []
        
        with torch.no_grad():
            for i in range(0, len(patches), batch_size):
                batch = torch.stack(patches[i : i + batch_size]).to(self.device)
                features = self.model(batch).cpu().numpy()
                all_features.append(features)
                
        # Concatenate all processed batches back together
        final_features = np.vstack(all_features)
        coords = np.array(coords)
        
        bio = io.BytesIO()
        with h5py.File(bio, 'w') as f:
            f.create_dataset('coords', data=coords)
            f.create_dataset('features', data=final_features)
            
        return bio.getvalue()

@app.local_entrypoint()
def extract_features():
    extractor = UniFeatureExtractor()
    public_dir = os.path.abspath("../frontend/public")
    patients = ["patient320", "patient321", "patient322"]
    
    for pat in patients:
        pat_dir = os.path.join(public_dir, pat)
        if not os.path.exists(pat_dir):
            continue
            
        raw_images = [img for img in glob.glob(os.path.join(pat_dir, "*.png")) if "_mask" not in img]
        
        for img_path in raw_images:
            h5_out_path = img_path.replace(".png", "_features.h5")
            
            print(f"Extracting dense UNI features for {os.path.basename(img_path)}...")
            with open(img_path, "rb") as f:
                img_bytes = f.read()
            
            # Force regeneration of the files to get the dense data
            h5_bytes = extractor.process_roi.remote(img_bytes, patch_size=224, stride=32)
            
            if h5_bytes:
                with open(h5_out_path, "wb") as f:
                    f.write(h5_bytes)
                print(f"Saved dense features to {h5_out_path} ({len(h5_bytes) // 1024} KB)")