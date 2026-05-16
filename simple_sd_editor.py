#!/usr/bin/env python3
import os
import time
from pathlib import Path

import gradio as gr
from PIL import Image
import numpy as np

# Try to import diffusers, fallback with instructions if not available
try:
    from diffusers import (
        DPMSolverMultistepScheduler,
        EulerAncestralDiscreteScheduler,
        StableDiffusionPipeline,
        StableDiffusionImg2ImgPipeline,
        StableDiffusionInpaintPipeline
    )
    import torch
    DIFFUSERS_AVAILABLE = True
except ImportError:
    DIFFUSERS_AVAILABLE = False
    print("⚠️  diffusers not installed. Run: pip install diffusers")


BASE_MODEL_ID = os.getenv("SD_MODEL_ID", "runwayml/stable-diffusion-v1-5")
INPAINT_MODEL_ID = os.getenv("SD_INPAINT_MODEL_ID", "runwayml/stable-diffusion-inpainting")
DEFAULT_NEGATIVE_PROMPT = (
    "cartoon, anime, painting, illustration, 3d render, cgi, plastic skin, waxy skin, "
    "over-smoothed skin, bad anatomy, deformed face, distorted eyes, extra fingers, "
    "missing fingers, blurry, low resolution, watermark, text, logo"
)
PHOTO_STYLES = {
    "Realistic Indian portrait": (
        "realistic candid photo of an Indian person, natural brown skin tones, detailed face, "
        "natural skin texture, expressive eyes, realistic hair, DSLR photo, 85mm lens, "
        "soft natural light, high detail, sharp focus"
    ),
    "Realistic photo": (
        "realistic DSLR photo, natural skin texture, accurate facial features, soft natural light, "
        "high detail, sharp focus"
    ),
    "Custom prompt only": "",
}
RESAMPLE_LANCZOS = getattr(Image, "Resampling", Image).LANCZOS


def configure_scheduler(pipe, scheduler_name):
    """Use stronger samplers than the diffusers default when available."""
    if scheduler_name == "DPM++ 2M Karras":
        try:
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                pipe.scheduler.config,
                algorithm_type="dpmsolver++",
                use_karras_sigmas=True,
            )
        except TypeError:
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    elif scheduler_name == "Euler a":
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

    return pipe


def optimize_pipeline(pipe, device):
    pipe = pipe.to(device)

    if hasattr(pipe, "enable_attention_slicing"):
        pipe.enable_attention_slicing()

    return pipe


def setup_models():
    """Initialize all Stable Diffusion pipelines"""
    if not DIFFUSERS_AVAILABLE:
        return None, None, None

    try:
        print("Loading Stable Diffusion models... this may take a minute")
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        dtype = torch.float32

        # Text-to-Image
        txt2img_pipe = StableDiffusionPipeline.from_pretrained(
            BASE_MODEL_ID,
            torch_dtype=dtype,
            safety_checker=None
        )
        txt2img_pipe = configure_scheduler(txt2img_pipe, "DPM++ 2M Karras")
        txt2img_pipe = optimize_pipeline(txt2img_pipe, device)

        # Image-to-Image
        img2img_pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            BASE_MODEL_ID,
            torch_dtype=dtype,
            safety_checker=None
        )
        img2img_pipe = configure_scheduler(img2img_pipe, "DPM++ 2M Karras")
        img2img_pipe = optimize_pipeline(img2img_pipe, device)

        # Use an inpainting-specific checkpoint; it preserves surrounding pixels better.
        inpaint_pipe = StableDiffusionInpaintPipeline.from_pretrained(
            INPAINT_MODEL_ID,
            torch_dtype=dtype,
            safety_checker=None
        )
        inpaint_pipe = configure_scheduler(inpaint_pipe, "DPM++ 2M Karras")
        inpaint_pipe = optimize_pipeline(inpaint_pipe, device)

        print(f"Models loaded on device: {device}")
        print(f"Base model: {BASE_MODEL_ID}")
        print(f"Inpaint model: {INPAINT_MODEL_ID}")
        return txt2img_pipe, img2img_pipe, inpaint_pipe
    except Exception as e:
        print(f"Error loading models: {e}")
        return None, None, None


def clean_seed(seed):
    try:
        seed = int(seed)
    except (TypeError, ValueError):
        seed = 0

    if seed <= 0:
        return int(time.time() * 1000) % 2_147_483_647

    return seed


def make_generator(seed):
    if not DIFFUSERS_AVAILABLE:
        return None

    generator_device = "mps" if torch.backends.mps.is_available() else "cpu"
    try:
        return torch.Generator(device=generator_device).manual_seed(seed)
    except Exception:
        return torch.Generator(device="cpu").manual_seed(seed)


def build_prompt(prompt, photo_style):
    style_text = PHOTO_STYLES.get(photo_style, "")
    prompt = (prompt or "").strip()

    if style_text and prompt:
        return f"{prompt}, {style_text}"

    return prompt or style_text


def normalize_image(image):
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image.astype("uint8"))

    return image.convert("RGB")


def resize_to_multiple_of_8(image, max_size):
    max_size = int(max_size)
    width, height = image.size
    scale = min(max_size / width, max_size / height, 1.0)
    new_width = max(64, int(width * scale) // 8 * 8)
    new_height = max(64, int(height * scale) // 8 * 8)

    return image.resize((new_width, new_height), RESAMPLE_LANCZOS)


def save_output(image, prefix, seed):
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    filename = output_dir / f"{prefix}_{seed}_{int(time.time())}.png"
    image.save(filename)
    return filename


def generate_image(prompt, negative_prompt, photo_style, width=512, height=768, num_steps=35, guidance_scale=6.5, seed=None, scheduler_name="DPM++ 2M Karras"):
    """Generate an image from text prompt"""
    if not DIFFUSERS_AVAILABLE:
        return None, "Error: diffusers library not installed\n\nRun: pip install diffusers torch pillow"

    if txt2img_pipe is None:
        return None, "Error: Model not loaded. Please restart and check dependencies."

    if not (prompt or photo_style != "Custom prompt only"):
        return None, "❌ Enter a prompt or choose a photo style."

    try:
        actual_seed = clean_seed(seed)
        generator = make_generator(actual_seed)
        configure_scheduler(txt2img_pipe, scheduler_name)
        final_prompt = build_prompt(prompt, photo_style)

        image = txt2img_pipe(
            prompt=final_prompt,
            negative_prompt=negative_prompt or DEFAULT_NEGATIVE_PROMPT,
            num_inference_steps=int(num_steps),
            guidance_scale=float(guidance_scale),
            height=int(height) // 8 * 8,
            width=int(width) // 8 * 8,
            generator=generator,
        ).images[0]

        filename = save_output(image, "txt2img", actual_seed)

        return image, f"✅ Image generated with seed {actual_seed} and saved to {filename}"
    except Exception as e:
        return None, f"❌ Error: {str(e)}"


def img2img_generate(input_image, prompt, negative_prompt, photo_style, strength=0.35, max_size=768, num_steps=35, guidance_scale=6.5, seed=None, scheduler_name="DPM++ 2M Karras"):
    """Generate an image based on an existing image (Image-to-Image)"""
    if not DIFFUSERS_AVAILABLE or img2img_pipe is None:
        return None, "Error: Model not available"

    if input_image is None:
        return None, "❌ Please upload an image first"

    if not (prompt or photo_style != "Custom prompt only"):
        return None, "❌ Enter a prompt or choose a photo style."

    try:
        actual_seed = clean_seed(seed)
        generator = make_generator(actual_seed)
        configure_scheduler(img2img_pipe, scheduler_name)
        input_image = resize_to_multiple_of_8(normalize_image(input_image), max_size)
        final_prompt = build_prompt(prompt, photo_style)

        image = img2img_pipe(
            prompt=final_prompt,
            negative_prompt=negative_prompt or DEFAULT_NEGATIVE_PROMPT,
            image=input_image,
            strength=float(strength),
            num_inference_steps=int(num_steps),
            guidance_scale=float(guidance_scale),
            generator=generator,
        ).images[0]

        filename = save_output(image, "img2img", actual_seed)

        return image, f"✅ Image edited with seed {actual_seed} and saved to {filename}"
    except Exception as e:
        return None, f"❌ Error: {str(e)}"


def inpaint_image(input_image, mask_image, prompt, negative_prompt, photo_style, strength=0.55, max_size=768, num_steps=35, guidance_scale=6.5, seed=None, scheduler_name="DPM++ 2M Karras"):
    """Edit part of an image using inpainting"""
    if not DIFFUSERS_AVAILABLE or inpaint_pipe is None:
        return None, "Error: Model not available"

    if input_image is None or mask_image is None:
        return None, "❌ Please upload both image and mask"

    if not (prompt or photo_style != "Custom prompt only"):
        return None, "❌ Enter a prompt or choose a photo style."

    try:
        actual_seed = clean_seed(seed)
        generator = make_generator(actual_seed)
        configure_scheduler(inpaint_pipe, scheduler_name)
        input_image = resize_to_multiple_of_8(normalize_image(input_image), max_size)
        mask_image = normalize_image(mask_image).resize(input_image.size, RESAMPLE_LANCZOS).convert("L")
        final_prompt = build_prompt(prompt, photo_style)

        image = inpaint_pipe(
            prompt=final_prompt,
            negative_prompt=negative_prompt or DEFAULT_NEGATIVE_PROMPT,
            image=input_image,
            mask_image=mask_image,
            num_inference_steps=int(num_steps),
            guidance_scale=float(guidance_scale),
            strength=float(strength),
            generator=generator,
        ).images[0]

        filename = save_output(image, "inpaint", actual_seed)

        return image, f"✅ Image inpainted with seed {actual_seed} and saved to {filename}"
    except Exception as e:
        return None, f"❌ Error: {str(e)}"

# Initialize models
print("Starting Stable Diffusion Image Editor...")
txt2img_pipe, img2img_pipe, inpaint_pipe = setup_models()

# Create Gradio interface
with gr.Blocks(title="Stable Diffusion Image Editor", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🎨 Stable Diffusion Image Editor")
    gr.Markdown(
        "Generate and edit realistic photos. Defaults are tuned for natural Indian people portraits "
        "and can be changed for other photo styles."
    )

    with gr.Tabs():
        # Text-to-Image Tab
        with gr.Tab("Text to Image"):
            gr.Markdown("### Generate realistic people photos from text")
            with gr.Row():
                with gr.Column():
                    txt_prompt = gr.Textbox(
                        label="Prompt",
                        placeholder="Example: a young Indian woman in a cotton saree, candid street portrait, warm evening light",
                        lines=4
                    )
                    txt_negative = gr.Textbox(
                        label="Negative Prompt",
                        value=DEFAULT_NEGATIVE_PROMPT,
                        lines=3
                    )
                    txt_style = gr.Radio(
                        label="Photo Style",
                        choices=list(PHOTO_STYLES.keys()),
                        value="Realistic Indian portrait"
                    )

                    with gr.Row():
                        txt_width = gr.Slider(
                            label="Width",
                            minimum=512,
                            maximum=1024,
                            value=512,
                            step=64
                        )
                        txt_height = gr.Slider(
                            label="Height",
                            minimum=512,
                            maximum=1024,
                            value=768,
                            step=64
                        )

                    with gr.Row():
                        txt_steps = gr.Slider(
                            label="Steps",
                            minimum=20,
                            maximum=60,
                            value=35,
                            step=1
                        )
                        txt_guidance = gr.Slider(
                            label="Guidance Scale",
                            minimum=1,
                            maximum=15,
                            value=6.5,
                            step=0.5
                        )

                    txt_scheduler = gr.Radio(
                        label="Sampler",
                        choices=["DPM++ 2M Karras", "Euler a"],
                        value="DPM++ 2M Karras"
                    )

                    txt_seed = gr.Number(label="Seed (Optional, 0 for random)", precision=0, value=0)
                    txt_generate_btn = gr.Button("🚀 Generate Image", variant="primary")

                with gr.Column():
                    txt_output = gr.Image(label="Generated Image")
                    txt_status = gr.Textbox(label="Status", interactive=False)

            txt_generate_btn.click(
                fn=generate_image,
                inputs=[
                    txt_prompt,
                    txt_negative,
                    txt_style,
                    txt_width,
                    txt_height,
                    txt_steps,
                    txt_guidance,
                    txt_seed,
                    txt_scheduler,
                ],
                outputs=[txt_output, txt_status]
            )

            gr.Markdown("""
            **Tips:**
            - For realistic Indian people, describe age, clothing, setting, expression, and lighting.
            - Keep CFG around 5-8 for natural skin; very high CFG can make faces harsh.
            - Portrait sizes such as 512x768 usually look better than square crops.
            - Reuse the reported seed when you want to refine the same face/composition.
            """)

        # Image to Image Tab
        with gr.Tab("Image to Image"):
            gr.Markdown("### Realistic edits to an existing photo")
            with gr.Row():
                with gr.Column():
                    img2img_input = gr.Image(label="Upload Image", type="pil")
                    img2img_strength = gr.Slider(
                        label="Strength (0=keep original, 1=full generation)",
                        minimum=0,
                        maximum=1,
                        value=0.35,
                        step=0.05
                    )
                    img2img_max_size = gr.Slider(
                        label="Max Image Size",
                        minimum=512,
                        maximum=1024,
                        value=768,
                        step=64
                    )

                with gr.Column():
                    img2img_prompt = gr.Textbox(
                        label="Edit Description",
                        placeholder="Example: make the portrait look like a natural Indian wedding photo, realistic lighting",
                        lines=4
                    )
                    img2img_negative = gr.Textbox(
                        label="Negative Prompt",
                        value=DEFAULT_NEGATIVE_PROMPT,
                        lines=3
                    )
                    img2img_style = gr.Radio(
                        label="Photo Style",
                        choices=list(PHOTO_STYLES.keys()),
                        value="Realistic Indian portrait"
                    )

                    with gr.Row():
                        img2img_steps = gr.Slider(
                            label="Steps",
                            minimum=20,
                            maximum=60,
                            value=35,
                            step=1
                        )
                        img2img_guidance = gr.Slider(
                            label="Guidance Scale",
                            minimum=1,
                            maximum=15,
                            value=6.5,
                            step=0.5
                        )

                    img2img_scheduler = gr.Radio(
                        label="Sampler",
                        choices=["DPM++ 2M Karras", "Euler a"],
                        value="DPM++ 2M Karras"
                    )

                    img2img_seed = gr.Number(label="Seed (Optional)", precision=0, value=0)
                    img2img_btn = gr.Button("🎨 Transform Image", variant="primary")

            with gr.Row():
                img2img_output = gr.Image(label="Transformed Image")
                img2img_status = gr.Textbox(label="Status", interactive=False)

            img2img_btn.click(
                fn=img2img_generate,
                inputs=[
                    img2img_input,
                    img2img_prompt,
                    img2img_negative,
                    img2img_style,
                    img2img_strength,
                    img2img_max_size,
                    img2img_steps,
                    img2img_guidance,
                    img2img_seed,
                    img2img_scheduler,
                ],
                outputs=[img2img_output, img2img_status]
            )

            gr.Markdown("""
            **Tips:**
            - Use strength 0.20-0.40 for realistic retouching while preserving identity.
            - Use strength 0.45-0.65 for stronger clothing, background, or lighting changes.
            - The app preserves aspect ratio instead of forcing everything into a square.
            """)

        # Inpainting Tab
        with gr.Tab("Inpainting (Edit Parts)"):
            gr.Markdown("### Realistic local edits with a mask")
            with gr.Row():
                with gr.Column():
                    inpaint_image_input = gr.Image(label="Upload Image", type="pil")
                    inpaint_mask_input = gr.Image(label="Mask (white = edit, black = keep)", type="pil")
                    gr.Markdown("**How to create a mask:**\n- White areas will be edited\n- Black areas will be preserved\n- Use any image editor to create the mask")

                with gr.Column():
                    inpaint_prompt = gr.Textbox(
                        label="What to put in the masked area?",
                        placeholder="Example: natural black hair, realistic Indian skin texture, soft studio light",
                        lines=4
                    )
                    inpaint_negative = gr.Textbox(
                        label="Negative Prompt",
                        value=DEFAULT_NEGATIVE_PROMPT,
                        lines=3
                    )
                    inpaint_style = gr.Radio(
                        label="Photo Style",
                        choices=list(PHOTO_STYLES.keys()),
                        value="Realistic Indian portrait"
                    )

                    with gr.Row():
                        inpaint_strength = gr.Slider(
                            label="Strength",
                            minimum=0,
                            maximum=1,
                            value=0.55,
                            step=0.05
                        )
                        inpaint_max_size = gr.Slider(
                            label="Max Image Size",
                            minimum=512,
                            maximum=1024,
                            value=768,
                            step=64
                        )

                    with gr.Row():
                        inpaint_steps = gr.Slider(
                            label="Steps",
                            minimum=20,
                            maximum=60,
                            value=35,
                            step=1
                        )
                        inpaint_guidance = gr.Slider(
                            label="Guidance Scale",
                            minimum=1,
                            maximum=15,
                            value=6.5,
                            step=0.5
                        )

                    inpaint_scheduler = gr.Radio(
                        label="Sampler",
                        choices=["DPM++ 2M Karras", "Euler a"],
                        value="DPM++ 2M Karras"
                    )

                    inpaint_seed = gr.Number(label="Seed (Optional)", precision=0, value=0)
                    inpaint_btn = gr.Button("✏️ Inpaint", variant="primary")

            with gr.Row():
                inpaint_output = gr.Image(label="Edited Image")
                inpaint_status = gr.Textbox(label="Status", interactive=False)

            inpaint_btn.click(
                fn=inpaint_image,
                inputs=[
                    inpaint_image_input,
                    inpaint_mask_input,
                    inpaint_prompt,
                    inpaint_negative,
                    inpaint_style,
                    inpaint_strength,
                    inpaint_max_size,
                    inpaint_steps,
                    inpaint_guidance,
                    inpaint_seed,
                    inpaint_scheduler,
                ],
                outputs=[inpaint_output, inpaint_status]
            )

if __name__ == "__main__":
    if DIFFUSERS_AVAILABLE and txt2img_pipe is not None:
        print("\n✅ Server is ready!")
        print("📍 Open in browser: http://localhost:7860")
        print("   or: http://127.0.0.1:7860")
        print("\n⏸️  Press Ctrl+C to stop\n")
        demo.launch(share=False, server_name="0.0.0.0", server_port=7860)
    else:
        print("\n❌ Cannot start editor - missing dependencies")
        print("Install with: pip install diffusers torch pillow")
