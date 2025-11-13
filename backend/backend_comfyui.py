# ================================================================
#  ComfyUI Text-to-Video Backend
#  Combines:
#   - Environment & Model Setup (auto-download + deps)
#   - Imports & Node Initialization
#   - Utility Functions (save, memory, etc.)
# ================================================================

import os
import sys
import subprocess
import gc
import torch
import numpy as np
from PIL import Image
import imageio

# ================================================================
#  PART 1 — ENVIRONMENT & MODEL SETUP
# ================================================================

def ensure_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def run_cmd(cmd):
    """Run a shell command with real-time output."""
    print(f"▶ {cmd}")
    subprocess.run(cmd, shell=True, check=True)

def install_dependencies():
    """Install all required Python packages if not already installed."""
    print("📦 Checking & installing dependencies ...")
    pkgs = [
        "torch==2.6.0",
        "torchvision==0.21.0",
        "torchsde",
        "einops",
        "diffusers",
        "accelerate",
        "xformers==0.0.29.post2",
        "av",
        "imageio[ffmpeg]",
        "pillow",
        "numpy"
    ]
    for pkg in pkgs:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg])

def download_file(url: str, dest: str):
    """Download file using aria2c if not present."""
    if not os.path.exists(dest):
        print(f"⬇️ Downloading {os.path.basename(dest)} ...")
        subprocess.run([
            "aria2c", "--console-log-level=error", "-c", "-x", "16", "-s", "16", "-k", "1M",
            url, "-d", os.path.dirname(dest), "-o", os.path.basename(dest)
        ], check=True)
    else:
        print(f"✅ Found existing: {os.path.basename(dest)}")

def setup_models(use_q6=False):
    """Ensure all required model weights are present."""
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    model_dir = os.path.join(base_dir, "models")
    ensure_dir(model_dir)

    models_to_download = {
        "umt5_xxl_fp8_e4m3fn_scaled.safetensors":
            "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors",
        "wan_2.1_vae.safetensors":
            "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/vae/wan_2.1_vae.safetensors",
        "wan2.1-t2v-14b-Q5_0.gguf":
            "https://huggingface.co/city96/Wan2.1-T2V-14B-gguf/resolve/main/wan2.1-t2v-14b-Q5_0.gguf",
        "wan2.1-t2v-14b-Q6_K.gguf":
            "https://huggingface.co/city96/Wan2.1-T2V-14B-gguf/resolve/main/wan2.1-t2v-14b-Q6_K.gguf"
    }

    # Download core models
    download_file(models_to_download["umt5_xxl_fp8_e4m3fn_scaled.safetensors"],
                  os.path.join(model_dir, "umt5_xxl_fp8_e4m3fn_scaled.safetensors"))
    download_file(models_to_download["wan_2.1_vae.safetensors"],
                  os.path.join(model_dir, "wan_2.1_vae.safetensors"))

    # Choose UNet precision
    if use_q6:
        download_file(models_to_download["wan2.1-t2v-14b-Q6_K.gguf"],
                      os.path.join(model_dir, "wan2.1-t2v-14b-Q6_K.gguf"))
    else:
        download_file(models_to_download["wan2.1-t2v-14b-Q5_0.gguf"],
                      os.path.join(model_dir, "wan2.1-t2v-14b-Q5_0.gguf"))

    print("✅ Model setup complete!")


# ================================================================
#  PART 2 — IMPORTS & NODES INITIALIZATION
# ================================================================

def import_comfy_nodes():
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "ComfyUI")))

    from comfy import model_management
    from nodes import (
        CheckpointLoaderSimple,
        CLIPLoader,
        CLIPTextEncode,
        VAEDecode,
        VAELoader,
        KSampler,
        UNETLoader
    )
    from custom_nodes.ComfyUI_GGUF.nodes import UnetLoaderGGUF
    from comfy_extras.nodes_model_advanced import ModelSamplingSD3
    from comfy_extras.nodes_hunyuan import EmptyHunyuanLatentVideo
    from comfy_extras.nodes_images import SaveAnimatedWEBP
    from comfy_extras.nodes_video import SaveWEBM

    print("✅ ComfyUI nodes imported successfully!")

    # Initialize reusable node instances
    nodes = {
        "unet_loader": UnetLoaderGGUF(),
        "clip_loader": CLIPLoader(),
        "clip_encode_positive": CLIPTextEncode(),
        "clip_encode_negative": CLIPTextEncode(),
        "vae_loader": VAELoader(),
        "empty_latent_video": EmptyHunyuanLatentVideo(),
        "ksampler": KSampler(),
        "vae_decode": VAEDecode(),
        "save_webp": SaveAnimatedWEBP(),
        "save_webm": SaveWEBM(),
    }

    return nodes


# ================================================================
#  PART 3 — MEMORY + SAVE UTILITIES
# ================================================================

def clear_memory():
    """Free CUDA and Python memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

def save_as_mp4(images, filename_prefix, fps, output_dir="output"):
    ensure_dir(output_dir)
    output_path = os.path.join(output_dir, f"{filename_prefix}.mp4")
    frames = [(img.cpu().numpy() * 255).astype(np.uint8) for img in images]

    with imageio.get_writer(output_path, fps=fps) as writer:
        for frame in frames:
            writer.append_data(frame)

    return output_path

def save_as_webm(images, filename_prefix, fps, codec="vp9", quality=30, output_dir="output"):
    ensure_dir(output_dir)
    output_path = os.path.join(output_dir, f"{filename_prefix}.webm")
    frames = [(img.cpu().numpy() * 255).astype(np.uint8) for img in images]

    kwargs = {
        "fps": int(fps),
        "quality": int(quality),
        "codec": str(codec),
        "output_params": ["-crf", str(int(quality))]
    }

    with imageio.get_writer(output_path, format="FFMPEG", mode="I", **kwargs) as writer:
        for frame in frames:
            writer.append_data(frame)

    return output_path

def save_as_image(image, filename_prefix, output_dir="output"):
    ensure_dir(output_dir)
    output_path = os.path.join(output_dir, f"{filename_prefix}.png")
    frame = (image.cpu().numpy() * 255).astype(np.uint8)
    Image.fromarray(frame).save(output_path)
    return output_path


# ================================================================
#  ENTRY POINT (Setup Mode)
# ================================================================

if __name__ == "__main__":
    print("🚀 Setting up ComfyUI backend ...")
    install_dependencies()
    setup_models(use_q6=False)
    nodes = import_comfy_nodes()
    print("✅ Backend ready for Gradio frontend connection.")
