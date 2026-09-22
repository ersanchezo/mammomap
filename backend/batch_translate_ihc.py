import modal
import io
import os
import glob

# 1. Define the Modal Image
ml_image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "libgl1-mesa-glx", "libglib2.0-0")
    .pip_install("torch", "torchvision", "Pillow", "numpy", "fastapi")
    .run_commands("git clone https://github.com/lifangda01/AdaptiveSupervisedPatchNCE.git /app/ASP")
    .env({"PYTHONPATH": "/app/ASP"})
)

app = modal.App("mammomap-ihc-translation")
weights_volume = modal.Volume.from_name("weights-volume")

# Stain mapping to volume paths
STAIN_PATHS = {
    "ER": "/weights/asp/mist_er_lambda_linear/latest_net_G.pth",
    "KI67": "/weights/asp/mist_ki67_lambda_linear/latest_net_G.pth",
    "HER2": "/weights/asp/mist_her2_lambda_linear/latest_net_G.pth",
    "PR": "/weights/asp/mist_pr_lambda_linear/latest_net_G.pth"
}

@app.cls(
    image=ml_image,
    gpu="T4",
    volumes={"/weights": weights_volume}
)
class ASPTranslator:
    @modal.enter()
    def load(self):
        import torch
        from torchvision import transforms
        from models.networks import ResnetGenerator
        from types import SimpleNamespace # <-- Import this to make a dummy object
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.models = {}
        
        # ASP requires an 'opt' object to check configuration flags during initialization.
        # We create a dummy namespace to satisfy the checks.
        dummy_opt = SimpleNamespace(
            weight_norm='spectral',
            no_antialias=False,
            no_antialias_up=False
        )
        
        # Load all 4 IHC generator models into GPU memory
        for stain_key, path in STAIN_PATHS.items():
            if os.path.exists(path):
                print(f"Loading ASP Generator for {stain_key}...")
                model = ResnetGenerator(
                    input_nc=3, 
                    output_nc=3, 
                    ngf=64, 
                    norm_layer=torch.nn.InstanceNorm2d, 
                    use_dropout=False, 
                    n_blocks=6,
                    opt=dummy_opt # <-- Pass the dummy options here!
                )
                state_dict = torch.load(path, map_location="cpu")
                if list(state_dict.keys())[0].startswith('module.'):
                    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
                
                model.load_state_dict(state_dict)
                model.to(self.device)
                model.eval()
                self.models[stain_key] = model
            else:
                print(f"Warning: Could not find weight file for {stain_key} at {path}")

        # Image Transforms
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        self.inv_transform = transforms.Compose([
            transforms.Normalize((-1.0, -1.0, -1.0), (2.0, 2.0, 2.0)),
            transforms.ToPILImage()
        ])

    @modal.method()
    def translate_patch(self, img_bytes, target_stain="HER2"):
        import torch
        from PIL import Image
        
        stain_key = str(target_stain).upper()
        if stain_key not in self.models:
            stain_key = list(self.models.keys())[0] if self.models else None
            
        if not stain_key:
            raise RuntimeError("No ASP translation models are loaded!")

        model = self.models[stain_key]

        # Load H&E Image
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        original_size = img.size
        
        # ASP expects 1024x1024 input dimensions
        img_resized = img.resize((1024, 1024), Image.Resampling.LANCZOS)
        input_tensor = self.transform(img_resized).unsqueeze(0).to(self.device)

        # Run forward pass through ResNet Generator
        with torch.no_grad():
            output_tensor = model(input_tensor)
            
        output_tensor = output_tensor.squeeze(0).cpu()
        translated_img = self.inv_transform(output_tensor)
        
        # Resize back to original ROI dimensions
        translated_img = translated_img.resize(original_size, Image.Resampling.LANCZOS)
        
        img_byte_arr = io.BytesIO()
        translated_img.save(img_byte_arr, format='PNG')
        return img_byte_arr.getvalue()

# --- LOCAL BATCH EXECUTION ENTRYPOINT ---
@app.local_entrypoint()
def process_demo_images():
    translator = ASPTranslator()
    public_dir = os.path.abspath("../frontend/public")
    patients = ["patient320", "patient321", "patient322"]
    stains = ["HER2", "ER", "PR", "KI67"]
    
    print(f"Scanning {public_dir} for demo ROI images...")
    
    for pat in patients:
        pat_dir = os.path.join(public_dir, pat)
        if not os.path.exists(pat_dir):
            continue
            
        # Get raw H&E images (ignore masks and already generated stain files)
        raw_images = [
            img for img in glob.glob(os.path.join(pat_dir, "*.png"))
            if not any(suffix in img for suffix in ["_mask", "_HER2", "_ER", "_PR", "_KI67", "_ihc"])
        ]
        
        for img_path in raw_images:
            for stain in stains:
                out_path = img_path.replace(".png", f"_{stain}.png")
                
                print(f"Generating {stain} stain for {os.path.basename(img_path)}...")
                with open(img_path, "rb") as f:
                    img_bytes = f.read()
                
                # Remote execution on Modal T4 GPU
                translated_bytes = translator.translate_patch.remote(img_bytes, target_stain=stain)
                
                # Save generated PNG directly into demo folder
                with open(out_path, "wb") as f:
                    f.write(translated_bytes)
                    
                print(f" Saved: {os.path.basename(out_path)}")

    print("\nAll demo IHC translations completed and cached!")

# --- FASTAPI WEB APPS (OPTIONAL LIVE WEB HOOK) ---
@app.function(image=ml_image)
@modal.asgi_app()
def api():
    from fastapi import FastAPI, UploadFile, File, Form
    from fastapi.responses import Response

    web_app = FastAPI()

    @web_app.post("/api/translate-ihc")
    async def translate_ihc(file: UploadFile = File(...), target_stain: str = Form("HER2")):
        img_bytes = await file.read()
        translator = ASPTranslator()
        translated_bytes = translator.translate_patch.remote(img_bytes, target_stain=target_stain)
        return Response(content=translated_bytes, media_type="image/png")

    return web_app