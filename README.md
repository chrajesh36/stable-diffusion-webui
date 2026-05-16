# Stable Diffusion Image Editor

This workspace contains two related Stable Diffusion apps:

- A small custom Gradio editor in `simple_sd_editor.py`.
- A larger AUTOMATIC1111-style WebUI in `webui.py` and `modules/`.

Use the simplified editor when you want a focused photo editor for realistic people generation, image-to-image edits, and inpainting. Use the full WebUI when you want the largest set of advanced controls, extensions, samplers, model switching, LoRA support, upscalers, and richer inpainting.

For an identity-preserving outfit-change recipe (same person, different outfit) on the full WebUI — launch with `./START_WEBUI.sh` and follow [`IDENTITY_OUTFIT_WORKFLOW.md`](IDENTITY_OUTFIT_WORKFLOW.md).

## Quick Start: Simplified Editor

Run:

```bash
./START_UI.sh
```

Or directly:

```bash
python3 simple_sd_editor.py
```

Then open `http://localhost:7860`.

On first launch, the Diffusers models download automatically and are cached for later runs. The default base model is `runwayml/stable-diffusion-v1-5`; inpainting uses `runwayml/stable-diffusion-inpainting`.

You can override the startup models without editing code:

```bash
SD_MODEL_ID="your/photo-model" SD_INPAINT_MODEL_ID="your/inpaint-model" ./START_UI.sh
```

## Simplified Editor Flow

`START_UI.sh` is a thin launcher. It changes into the repository directory and runs `simple_sd_editor.py` with `/usr/local/bin/python3`.

`simple_sd_editor.py` does these main things:

- Imports Gradio, Pillow, NumPy, Diffusers, and PyTorch.
- Calls `setup_models()` at startup.
- Builds a Gradio `Blocks` UI with three tabs.
- Wires each tab's button to a Python callback.
- Adds realistic-photo defaults for Indian people portraits, negative prompts, samplers, seed handling, and aspect-aware resizing.

The model setup loads three separate Diffusers pipelines:

- `StableDiffusionPipeline` for text-to-image from `SD_MODEL_ID`.
- `StableDiffusionImg2ImgPipeline` for image-to-image from `SD_MODEL_ID`.
- `StableDiffusionInpaintPipeline` for inpainting from `SD_INPAINT_MODEL_ID`.

The simplified callback path is:

```text
START_UI.sh
  -> simple_sd_editor.py
    -> setup_models()
    -> Gradio tabs
      -> generate_image()
      -> img2img_generate()
      -> inpaint_image()
    -> outputs/
```

`generate_image()` creates a photo from a prompt, negative prompt, selected photo style, dimensions, sampler, seed, steps, and guidance. `img2img_generate()` preserves the uploaded image aspect ratio, resizes to a model-friendly multiple of 8, and transforms it with a prompt and strength value. `inpaint_image()` uses an inpainting-specific model, resizes the mask to match the source image, converts the mask to grayscale, and fills the white masked area from the prompt.

## Realistic Indian Photo Settings

The simplified editor now defaults to `Realistic Indian portrait`, which appends photo-realistic guidance such as natural brown skin tones, natural skin texture, realistic hair, DSLR photo, and soft natural light.

Recommended starting points:

- Text-to-image portraits: `512x768`, `35` steps, CFG `6.5`, sampler `DPM++ 2M Karras`.
- Subtle image edits: img2img strength `0.20-0.40`.
- Stronger clothing, background, or lighting edits: img2img strength `0.45-0.65`.
- Inpainting: strength around `0.55`, then lower it if the face identity changes too much.
- Negative prompt: keep the default unless you see a specific recurring flaw.

For best realism, describe the person and setting plainly: age, expression, clothing, location, lighting, lens feel, and mood. Example: `middle-aged Indian man in a linen kurta, candid portrait at home, warm window light, natural skin texture, realistic DSLR photo`.

## Full WebUI Flow

The full WebUI is the inherited AUTOMATIC1111-style application. Its entry points are `launch.py`, `webui.py`, and `webui.sh`.

The main startup path is:

```text
launch.py or webui.sh
  -> webui.py
    -> initialize.imports()
    -> initialize.check_versions()
    -> initialize.initialize()
    -> modules.ui.create_ui()
    -> Gradio launch
```

Important full WebUI modules:

- `modules/initialize.py` sets up paths, versions, models, samplers, extensions, upscalers, VAE, textual inversion, and extra networks.
- `modules/ui.py` builds the full Gradio interface.
- `modules/txt2img.py` converts UI text-to-image requests into a `StableDiffusionProcessingTxt2Img` object.
- `modules/img2img.py` converts image-to-image and inpainting UI requests into a `StableDiffusionProcessingImg2Img` object.
- `modules/processing.py` contains the shared generation loop, including seed handling, prompt conditioning, sampling, VAE decode, postprocessing, saving, and metadata.
- `modules/sd_models.py` discovers and loads checkpoints from the model folders.
- `modules/scripts.py` loads extension and script hooks.

The generation path is:

```text
modules.ui.create_ui()
  -> modules.txt2img.txt2img() or modules.img2img.img2img()
  -> modules.scripts hooks
  -> modules.processing.process_images()
  -> sampler
  -> decoded image
  -> saved output and metadata
```

## Which App Should You Use?

Use `simple_sd_editor.py` for quick experiments or when you want a compact code path that is easy to edit.

Use the full WebUI for the best practical image generation and editing experience. It exposes advanced features that the simplified editor does not, including sampler selection, highres fix, batch controls, richer masks, checkpoint switching, extensions, LoRAs, face restoration, and upscalers.

## Current Cleanup Notes

The README used to mix custom simplified-editor instructions with the upstream AUTOMATIC1111 feature list without clearly separating them. This file now treats the custom editor and the full WebUI as separate paths.

The simplified editor has a few intentional tradeoffs:

- It defaults to `runwayml/stable-diffusion-v1-5`, but `SD_MODEL_ID` can point to a stronger photo model.
- It loads three separate pipelines at startup, which is simple but memory-heavy.
- It does not expose WebUI features such as highres fix, LoRAs, VAE selection, or advanced inpainting controls.

## System Requirements

- RAM: 8GB minimum, 16GB recommended.
- Disk: 10GB+ available space for models and outputs.
- GPU: optional but recommended for faster generation.
- Apple Silicon: supported through MPS-compatible PyTorch paths.

## Prompting Tips

- Be specific and detailed.
- Include art style, such as "oil painting", "photorealistic", or "digital art".
- Specify lighting, mood, composition, and subject details.
- Reuse seeds when you want reproducible results.

## Upstream WebUI Feature Reference

[Detailed feature showcase with images](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Features):
- Original txt2img and img2img modes
- One click install and run script (but you still must install python and git)
- Outpainting
- Inpainting
- Color Sketch
- Prompt Matrix
- Stable Diffusion Upscale
- Attention, specify parts of text that the model should pay more attention to
    - a man in a `((tuxedo))` - will pay more attention to tuxedo
    - a man in a `(tuxedo:1.21)` - alternative syntax
    - select text and press `Ctrl+Up` or `Ctrl+Down` (or `Command+Up` or `Command+Down` if you're on a MacOS) to automatically adjust attention to selected text (code contributed by anonymous user)
- Loopback, run img2img processing multiple times
- X/Y/Z plot, a way to draw a 3 dimensional plot of images with different parameters
- Textual Inversion
    - have as many embeddings as you want and use any names you like for them
    - use multiple embeddings with different numbers of vectors per token
    - works with half precision floating point numbers
    - train embeddings on 8GB (also reports of 6GB working)
- Extras tab with:
    - GFPGAN, neural network that fixes faces
    - CodeFormer, face restoration tool as an alternative to GFPGAN
    - RealESRGAN, neural network upscaler
    - ESRGAN, neural network upscaler with a lot of third party models
    - SwinIR and Swin2SR ([see here](https://github.com/AUTOMATIC1111/stable-diffusion-webui/pull/2092)), neural network upscalers
    - LDSR, Latent diffusion super resolution upscaling
- Resizing aspect ratio options
- Sampling method selection
    - Adjust sampler eta values (noise multiplier)
    - More advanced noise setting options
- Interrupt processing at any time
- 4GB video card support (also reports of 2GB working)
- Correct seeds for batches
- Live prompt token length validation
- Generation parameters
     - parameters you used to generate images are saved with that image
     - in PNG chunks for PNG, in EXIF for JPEG
     - can drag the image to PNG info tab to restore generation parameters and automatically copy them into UI
     - can be disabled in settings
     - drag and drop an image/text-parameters to promptbox
- Read Generation Parameters Button, loads parameters in promptbox to UI
- Settings page
- Running arbitrary python code from UI (must run with `--allow-code` to enable)
- Mouseover hints for most UI elements
- Possible to change defaults/mix/max/step values for UI elements via text config
- Tiling support, a checkbox to create images that can be tiled like textures
- Progress bar and live image generation preview
    - Can use a separate neural network to produce previews with almost none VRAM or compute requirement
- Negative prompt, an extra text field that allows you to list what you don't want to see in generated image
- Styles, a way to save part of prompt and easily apply them via dropdown later
- Variations, a way to generate same image but with tiny differences
- Seed resizing, a way to generate same image but at slightly different resolution
- CLIP interrogator, a button that tries to guess prompt from an image
- Prompt Editing, a way to change prompt mid-generation, say to start making a watermelon and switch to anime girl midway
- Batch Processing, process a group of files using img2img
- Img2img Alternative, reverse Euler method of cross attention control
- Highres Fix, a convenience option to produce high resolution pictures in one click without usual distortions
- Reloading checkpoints on the fly
- Checkpoint Merger, a tab that allows you to merge up to 3 checkpoints into one
- [Custom scripts](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Custom-Scripts) with many extensions from community
- [Composable-Diffusion](https://energy-based-model.github.io/Compositional-Visual-Generation-with-Composable-Diffusion-Models/), a way to use multiple prompts at once
     - separate prompts using uppercase `AND`
     - also supports weights for prompts: `a cat :1.2 AND a dog AND a penguin :2.2`
- No token limit for prompts (original stable diffusion lets you use up to 75 tokens)
- DeepDanbooru integration, creates danbooru style tags for anime prompts
- [xformers](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Xformers), major speed increase for select cards: (add `--xformers` to commandline args)
- via extension: [History tab](https://github.com/yfszzx/stable-diffusion-webui-images-browser): view, direct and delete images conveniently within the UI
- Generate forever option
- Training tab
     - hypernetworks and embeddings options
     - Preprocessing images: cropping, mirroring, autotagging using BLIP or deepdanbooru (for anime)
- Clip skip
- Hypernetworks
- Loras (same as Hypernetworks but more pretty)
- A separate UI where you can choose, with preview, which embeddings, hypernetworks or Loras to add to your prompt
- Can select to load a different VAE from settings screen
- Estimated completion time in progress bar
- API
- Support for dedicated [inpainting model](https://github.com/runwayml/stable-diffusion#inpainting-with-stable-diffusion) by RunwayML
- via extension: [Aesthetic Gradients](https://github.com/AUTOMATIC1111/stable-diffusion-webui-aesthetic-gradients), a way to generate images with a specific aesthetic by using clip images embeds (implementation of [https://github.com/vicgalle/stable-diffusion-aesthetic-gradients](https://github.com/vicgalle/stable-diffusion-aesthetic-gradients))
- [Stable Diffusion 2.0](https://github.com/Stability-AI/stablediffusion) support - see [wiki](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Features#stable-diffusion-20) for instructions
- [Alt-Diffusion](https://arxiv.org/abs/2211.06679) support - see [wiki](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Features#alt-diffusion) for instructions
- Now without any bad letters!
- Load checkpoints in safetensors format
- Eased resolution restriction: generated image's dimensions must be a multiple of 8 rather than 64
- Now with a license!
- Reorder elements in the UI from settings screen
- [Segmind Stable Diffusion](https://huggingface.co/segmind/SSD-1B) support

## Installation and Running
Make sure the required [dependencies](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Dependencies) are met and follow the instructions available for:
- [NVidia](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Install-and-Run-on-NVidia-GPUs) (recommended)
- [AMD](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Install-and-Run-on-AMD-GPUs) GPUs.
- [Intel CPUs, Intel GPUs (both integrated and discrete)](https://github.com/openvinotoolkit/stable-diffusion-webui/wiki/Installation-on-Intel-Silicon) (external wiki page)
- [Ascend NPUs](https://github.com/wangshuai09/stable-diffusion-webui/wiki/Install-and-run-on-Ascend-NPUs) (external wiki page)

Alternatively, use online services (like Google Colab):

- [List of Online Services](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Online-Services)

### Installation on Windows 10/11 with NVidia-GPUs using release package
1. Download `sd.webui.zip` from [v1.0.0-pre](https://github.com/AUTOMATIC1111/stable-diffusion-webui/releases/tag/v1.0.0-pre) and extract its contents.
2. Run `update.bat`.
3. Run `run.bat`.
> For more details see [Install-and-Run-on-NVidia-GPUs](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Install-and-Run-on-NVidia-GPUs)

### Automatic Installation on Windows
1. Install [Python 3.10.6](https://www.python.org/downloads/release/python-3106/) (Newer version of Python does not support torch), checking "Add Python to PATH".
2. Install [git](https://git-scm.com/download/win).
3. Download the stable-diffusion-webui repository, for example by running `git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui.git`.
4. Run `webui-user.bat` from Windows Explorer as normal, non-administrator, user.

### Automatic Installation on Linux
1. Install the dependencies:
```bash
# Debian-based:
sudo apt install wget git python3 python3-venv libgl1 libglib2.0-0
# Red Hat-based:
sudo dnf install wget git python3 gperftools-libs libglvnd-glx
# openSUSE-based:
sudo zypper install wget git python3 libtcmalloc4 libglvnd
# Arch-based:
sudo pacman -S wget git python3
```
If your system is very new, you need to install python3.11 or python3.10:
```bash
# Ubuntu 24.04
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install python3.11

# Manjaro/Arch
sudo pacman -S yay
yay -S python311 # do not confuse with python3.11 package

# Only for 3.11
# Then set up env variable in launch script
export python_cmd="python3.11"
# or in webui-user.sh
python_cmd="python3.11"
```
2. Navigate to the directory you would like the webui to be installed and execute the following command:
```bash
wget -q https://raw.githubusercontent.com/AUTOMATIC1111/stable-diffusion-webui/master/webui.sh
```
Or just clone the repo wherever you want:
```bash
git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui
```

3. Run `webui.sh`.
4. Check `webui-user.sh` for options.
### Installation on Apple Silicon

Find the instructions [here](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Installation-on-Apple-Silicon).

## Contributing
Here's how to add code to this repo: [Contributing](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Contributing)

## Documentation

The documentation was moved from this README over to the project's [wiki](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki).

For the purposes of getting Google and other search engines to crawl the wiki, here's a link to the (not for humans) [crawlable wiki](https://github-wiki-see.page/m/AUTOMATIC1111/stable-diffusion-webui/wiki).

## Credits
Licenses for borrowed code can be found in `Settings -> Licenses` screen, and also in `html/licenses.html` file.

- Stable Diffusion - https://github.com/Stability-AI/stablediffusion, https://github.com/CompVis/taming-transformers, https://github.com/mcmonkey4eva/sd3-ref
- k-diffusion - https://github.com/crowsonkb/k-diffusion.git
- Spandrel - https://github.com/chaiNNer-org/spandrel implementing
  - GFPGAN - https://github.com/TencentARC/GFPGAN.git
  - CodeFormer - https://github.com/sczhou/CodeFormer
  - ESRGAN - https://github.com/xinntao/ESRGAN
  - SwinIR - https://github.com/JingyunLiang/SwinIR
  - Swin2SR - https://github.com/mv-lab/swin2sr
- LDSR - https://github.com/Hafiidz/latent-diffusion
- MiDaS - https://github.com/isl-org/MiDaS
- Ideas for optimizations - https://github.com/basujindal/stable-diffusion
- Cross Attention layer optimization - Doggettx - https://github.com/Doggettx/stable-diffusion, original idea for prompt editing.
- Cross Attention layer optimization - InvokeAI, lstein - https://github.com/invoke-ai/InvokeAI (originally http://github.com/lstein/stable-diffusion)
- Sub-quadratic Cross Attention layer optimization - Alex Birch (https://github.com/Birch-san/diffusers/pull/1), Amin Rezaei (https://github.com/AminRezaei0x443/memory-efficient-attention)
- Textual Inversion - Rinon Gal - https://github.com/rinongal/textual_inversion (we're not using his code, but we are using his ideas).
- Idea for SD upscale - https://github.com/jquesnelle/txt2imghd
- Noise generation for outpainting mk2 - https://github.com/parlance-zz/g-diffuser-bot
- CLIP interrogator idea and borrowing some code - https://github.com/pharmapsychotic/clip-interrogator
- Idea for Composable Diffusion - https://github.com/energy-based-model/Compositional-Visual-Generation-with-Composable-Diffusion-Models-PyTorch
- xformers - https://github.com/facebookresearch/xformers
- DeepDanbooru - interrogator for anime diffusers https://github.com/KichangKim/DeepDanbooru
- Sampling in float32 precision from a float16 UNet - marunine for the idea, Birch-san for the example Diffusers implementation (https://github.com/Birch-san/diffusers-play/tree/92feee6)
- Instruct pix2pix - Tim Brooks (star), Aleksander Holynski (star), Alexei A. Efros (no star) - https://github.com/timothybrooks/instruct-pix2pix
- Security advice - RyotaK
- UniPC sampler - Wenliang Zhao - https://github.com/wl-zhao/UniPC
- TAESD - Ollin Boer Bohan - https://github.com/madebyollin/taesd
- LyCORIS - KohakuBlueleaf
- Restart sampling - lambertae - https://github.com/Newbeeer/diffusion_restart_sampling
- Hypertile - tfernd - https://github.com/tfernd/HyperTile
- Initial Gradio script - posted on 4chan by an Anonymous user. Thank you Anonymous user.
- (You)
