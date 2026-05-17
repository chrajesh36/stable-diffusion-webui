#!/usr/bin/env python3
"""
Script to download Stable Diffusion model weights
"""

import os
import sys
import requests
from pathlib import Path

def download_file(url, destination):
    """Download a file from URL to destination with progress bar"""
    print(f"Downloading {os.path.basename(destination)}...")
    
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    
    with open(destination, 'wb') as f:
        downloaded = 0
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    percent = (downloaded / total_size) * 100
                    print(f"  Progress: {percent:.1f}% ({downloaded / 1024 / 1024:.1f}MB / {total_size / 1024 / 1024:.1f}MB)")
    
    print(f"Downloaded successfully to {destination}")

def main():
    models_dir = Path("models/Stable-diffusion")
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # Note: Use a smaller model for testing. Uncomment below if you want the full model.
    # For demo purposes, we'll use the pruned version which is ~3.9GB
    
    model_url = "https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned.ckpt"
    model_path = models_dir / "v1-5-pruned.ckpt"
    
    if model_path.exists():
        print(f"Model already exists at {model_path}")
    else:
        try:
            download_file(model_url, str(model_path))
        except Exception as e:
            print(f"Error downloading model: {e}")
            print("\nYou can manually download the model from:")
            print(f"  {model_url}")
            print(f"\nAnd place it in: {models_dir}")
            sys.exit(1)

if __name__ == "__main__":
    main()
