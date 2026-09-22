import modal
import os

app = modal.App("download-sequoia")
weights_volume = modal.Volume.from_name("weights-volume")
hf_secret = modal.Secret.from_name("huggingface-secret")

# ─── FIX: Define an image that has huggingface_hub installed ───
downloader_image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install("huggingface_hub", "hf_transfer")
)

@app.function(
    image=downloader_image,          # Attach the image here!
    volumes={"/weights": weights_volume}, 
    secrets=[hf_secret], 
    timeout=3600                     # 1 hour timeout just in case
)
def download_models():
    # Enable lightning-fast rust-based downloads
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    from huggingface_hub import snapshot_download
    
    # 1. Download UNI Foundation Model (~1.2 GB)
    print("Downloading UNI Foundation Model...")
    uni_dir = "/weights/uni"
    os.makedirs(uni_dir, exist_ok=True)
    snapshot_download(
        repo_id="MahmoodLab/UNI", 
        local_dir=uni_dir,
        allow_patterns=["*.bin", "*.json", "*.safetensors"] # Added safetensors just in case
    )
    print("✅ UNI weights saved to /weights/uni")

    # 2. Download SEQUOIA BRCA Weights
    print("Downloading SEQUOIA BRCA Checkpoint...")
    sequoia_dir = "/weights/sequoia"
    os.makedirs(sequoia_dir, exist_ok=True)
    snapshot_download(
        repo_id="gevaertlab/sequoia-brca-1", 
        local_dir=sequoia_dir
    )
    print("✅ SEQUOIA BRCA weights saved to /weights/sequoia")
    
    # Commit changes to the Modal Volume so they persist
    weights_volume.commit()

@app.local_entrypoint()
def main():
    print("Starting secure download to Modal volume...")
    download_models.remote()
    print("Downloads complete! Your weights-volume is now updated.")