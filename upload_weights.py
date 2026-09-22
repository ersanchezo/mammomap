"""
upload_weights.py — run once from your laptop to push model weights to Modal Volume

Usage:
    # 1. Mount your Google Drive locally (or copy files off first)
    # 2. Edit the paths below
    # 3. Run:  python upload_weights.py

What it uploads to the Modal Volume "pathvision-weights":
    /seg_adapter/   ← your LoRA adapter directory (checkpoint_epoch_18_adapter/)
    /seg_decoder.pth ← your ATM decoder file (checkpoint_epoch_18_decoder.pth)
"""

import sys
from pathlib import Path
import modal

# ── Edit these paths to point at your local copies of the weights ─────────────
LOCAL_ADAPTER_DIR  = Path("checkpoint_epoch_18_adapter")   # directory
LOCAL_DECODER_PATH = Path("checkpoint_epoch_18_decoder.pth")  # single file

# Optional: also upload IHC/spatial model weights here later
# LOCAL_SEQUOIA_DIR = Path("sequoia_weights/brca-fold0")

# ─────────────────────────────────────────────────────────────────────────────
def main():
    if not LOCAL_ADAPTER_DIR.exists():
        sys.exit(f"Adapter directory not found: {LOCAL_ADAPTER_DIR.resolve()}\n"
                 "Copy it from Google Drive first, then re-run this script.")
    if not LOCAL_DECODER_PATH.exists():
        sys.exit(f"Decoder file not found: {LOCAL_DECODER_PATH.resolve()}")

    print("Connecting to Modal Volume 'pathvision-weights'…")
    vol = modal.Volume.from_name("pathvision-weights", create_if_missing=True)

    print(f"Uploading adapter from {LOCAL_ADAPTER_DIR} …")
    with vol.batch_upload(force=True) as batch:
        # Upload each file in the adapter directory
        for f in sorted(LOCAL_ADAPTER_DIR.rglob("*")):
            if f.is_file():
                dest = f"/seg_adapter/{f.relative_to(LOCAL_ADAPTER_DIR)}"
                print(f"  {f} → {dest}")
                batch.put_file(str(f), dest)

        # Upload decoder
        print(f"Uploading decoder {LOCAL_DECODER_PATH} → /seg_decoder.pth")
        batch.put_file(str(LOCAL_DECODER_PATH), "/seg_decoder.pth")

    print("\nDone. Verify with:")
    print("  modal volume ls pathvision-weights /")
    print("  modal volume ls pathvision-weights /seg_adapter/")

if __name__ == "__main__":
    main()
