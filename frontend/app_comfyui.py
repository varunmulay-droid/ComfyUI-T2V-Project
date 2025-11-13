# ================================================================
#  ComfyUI Text-to-Video Frontend (Gradio)
# ================================================================

import os
import gradio as gr
import torch
import sys
from pathlib import Path

# Add backend path
base_dir = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(base_dir / "backend"))

from backend_comfyui import (
    setup_models,
    import_comfy_nodes,
    clear_memory,
    save_as_mp4,
    save_as_image
)

# Initialize backend once
print("🔧 Initializing backend ...")
setup_models(use_q6=False)
nodes = import_comfy_nodes()
print("✅ Backend ready!")

# Node shortcuts
clip_loader = nodes["clip_loader"]
clip_encode_positive = nodes["clip_encode_positive"]
clip_encode_negative = nodes["clip_encode_negative"]
vae_loader = nodes["vae_loader"]
vae_decode = nodes["vae_decode"]
ksampler = nodes["ksampler"]
empty_latent_video = nodes["empty_latent_video"]
unet_loader = nodes["unet_loader"]

# ================================================================
#  Core Generation Function
# ================================================================

def generate_t2v(
    positive_prompt,
    negative_prompt,
    width,
    height,
    seed,
    steps,
    cfg_scale,
    sampler_name,
    scheduler,
    frames,
    fps,
    output_format,
    use_q6
):
    try:
        with torch.inference_mode():
            print("🎬 Loading models...")

            # Load text encoder
            clip = clip_loader.load_clip("umt5_xxl_fp8_e4m3fn_scaled.safetensors", "wan", "default")[0]
            positive = clip_encode_positive.encode(clip, positive_prompt)[0]
            negative = clip_encode_negative.encode(clip, negative_prompt)[0]
            del clip
            torch.cuda.empty_cache()

            # Create latent video
            latent = empty_latent_video.generate(width, height, frames, 1)[0]

            # Load UNet model
            if use_q6:
                model = unet_loader.load_unet("wan2.1-t2v-14b-Q6_K.gguf")[0]
            else:
                model = unet_loader.load_unet("wan2.1-t2v-14b-Q5_0.gguf")[0]

            print("🧠 Sampling frames...")
            sampled = ksampler.sample(
                model=model,
                seed=seed,
                steps=steps,
                cfg=cfg_scale,
                sampler_name=sampler_name,
                scheduler=scheduler,
                positive=positive,
                negative=negative,
                latent_image=latent
            )[0]

            del model
            torch.cuda.empty_cache()

            # Decode with VAE
            vae = vae_loader.load_vae("wan_2.1_vae.safetensors")[0]
            decoded = vae_decode.decode(vae, sampled)[0]
            del vae
            torch.cuda.empty_cache()

            output_dir = "output"
            os.makedirs(output_dir, exist_ok=True)

            if frames == 1:
                print("🖼️ Single frame → PNG")
                output_path = save_as_image(decoded[0], "result", output_dir=output_dir)
            else:
                print("🎥 Multiple frames → Video")
                if output_format.lower() == "mp4":
                    output_path = save_as_mp4(decoded, "result", fps=fps, output_dir=output_dir)
                else:
                    from backend_comfyui import save_as_webm
                    output_path = save_as_webm(decoded, "result", fps=fps, output_dir=output_dir)

            clear_memory()
            print(f"✅ Done! Saved at {output_path}")
            return output_path

    except Exception as e:
        clear_memory()
        return f"❌ Error: {str(e)}"

# ================================================================
#  Gradio UI Layout
# ================================================================

with gr.Blocks(title="ComfyUI Text-to-Video Generator") as demo:
    gr.Markdown(
        """
        # 🎬 ComfyUI-Wan 2.1 Text-to-Video  
        Enter your text prompt to generate realistic video frames.  
        *(Runs fully local — requires GPU and models auto-downloaded.)*
        """
    )

    with gr.Row():
        with gr.Column(scale=2):
            positive_prompt = gr.Textbox(
                label="Positive Prompt",
                value="A fox moving through snowy forest at sunrise",
                lines=3,
                placeholder="Describe your video scene..."
            )
            negative_prompt = gr.Textbox(
                label="Negative Prompt",
                value="static, blurry, distorted, ugly, extra limbs, low quality",
                lines=2
            )

            with gr.Accordion("Advanced Settings", open=False):
                width = gr.Slider(256, 1024, 512, step=64, label="Width")
                height = gr.Slider(256, 1024, 512, step=64, label="Height")
                seed = gr.Number(value=12345, label="Seed", precision=0)
                steps = gr.Slider(1, 100, 20, step=1, label="Sampling Steps")
                cfg_scale = gr.Slider(1, 10, 3, step=0.5, label="CFG Scale")
                sampler_name = gr.Dropdown(
                    ["uni_pc", "euler", "dpmpp_2m", "ddim", "lms"],
                    value="uni_pc",
                    label="Sampler"
                )
                scheduler = gr.Dropdown(
                    ["simple", "normal", "karras", "exponential"],
                    value="simple",
                    label="Scheduler"
                )
                frames = gr.Slider(1, 120, 24, step=1, label="Frame Count")
                fps = gr.Slider(1, 60, 20, step=1, label="FPS")
                output_format = gr.Radio(["mp4", "webm"], value="mp4", label="Output Format")
                use_q6 = gr.Checkbox(value=False, label="Use Q6 Precision (slower but higher quality)")

            generate_btn = gr.Button("🚀 Generate Video")

        with gr.Column(scale=1):
            output_video = gr.Video(label="Generated Output", autoplay=True, loop=True)

    generate_btn.click(
        fn=generate_t2v,
        inputs=[
            positive_prompt, negative_prompt, width, height,
            seed, steps, cfg_scale, sampler_name, scheduler,
            frames, fps, output_format, use_q6
        ],
        outputs=output_video
    )

# ================================================================
#  Launch
# ================================================================

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
