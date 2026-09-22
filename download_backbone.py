import modal
import os

# 1. Define an image that has huggingface_hub installed
image = modal.Image.debian_slim().pip_install("huggingface_hub")

# 2. Define the app and attach the image
app = modal.App("download-virchow", image=image)

# Reference your existing secrets and volumes
hf_secret = modal.Secret.from_name("huggingface-secret")
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)

# 3. Create the function
@app.function(secrets=[hf_secret], volumes={"/root/.cache/huggingface": hf_cache})
def download_virchow_to_volume():
    from huggingface_hub import snapshot_download, login
    
    print("Logging into Hugging Face...")
    login(token=os.environ["HF_TOKEN"])
    
    print("Downloading Virchow2 backbone (2.3 GB) to Modal Volume...")
    snapshot_download(repo_id="paige-ai/Virchow2")
    
    hf_cache.commit()
    print("Download complete and cached!")

# Entry point
@app.local_entrypoint()
def main():
    download_virchow_to_volume.remote()