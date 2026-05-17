#!/usr/bin/env python3
"""
Identity-preservation validation for simple_sd_editor outfit/attire swaps.

Runs entirely on a synthetic txt2img base portrait (NO real photos), then
img2img-swaps the wardrobe at multiple strengths and reports CLIP-based
image-image cosine similarity to the base. Outputs go under tests_identity/.

This script is INTENDED to be local-only validation. It does not modify any
tracked file in the repo and is safe to delete (rm -rf tests_identity/).
"""
from __future__ import annotations

import csv
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from diffusers import (
    DPMSolverMultistepScheduler,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionPipeline,
)


HERE = Path(__file__).resolve().parent
BASE_MODEL_ID = os.getenv("SD_MODEL_ID", "runwayml/stable-diffusion-v1-5")

DEFAULT_NEGATIVE_PROMPT = (
    "cartoon, anime, painting, illustration, 3d render, cgi, plastic skin, waxy skin, "
    "over-smoothed skin, bad anatomy, deformed face, distorted eyes, extra fingers, "
    "missing fingers, blurry, low resolution, watermark, text, logo, "
    "nudity, nude, nsfw, swimsuit, bikini, underwear, shirtless"
)

BASE_PROMPT = (
    "realistic candid photo of a young Indian man with short black hair, "
    "plain background, neutral expression, DSLR photo, 85mm lens, "
    "soft natural light, high detail, sharp focus"
)

OUTFIT_PROMPTS = {
    "blazer": (
        "same young Indian man wearing a formal navy blue blazer and white shirt, "
        "realistic, natural skin texture, soft natural light, DSLR photo, high detail"
    ),
    "kurta": (
        "same young Indian man wearing a traditional white kurta with subtle embroidery, "
        "realistic, natural skin texture, DSLR photo, high detail"
    ),
}

STRENGTHS = [0.25, 0.35, 0.45, 0.55]
SEED = 12345
WIDTH, HEIGHT = 512, 768
NUM_STEPS = 35
GUIDANCE = 6.5


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def configure_scheduler(pipe):
    try:
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            pipe.scheduler.config,
            algorithm_type="dpmsolver++",
            use_karras_sigmas=True,
        )
    except TypeError:
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    return pipe


def make_generator(device: str, seed: int) -> torch.Generator:
    # MPS generator has restrictions; use CPU generator to feed the pipeline.
    gen_device = "cpu" if device == "mps" else device
    return torch.Generator(device=gen_device).manual_seed(seed)


def generate_base(device: str) -> Image.Image:
    log("Loading StableDiffusionPipeline (txt2img)...")
    pipe = StableDiffusionPipeline.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch.float32,
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipe = configure_scheduler(pipe)
    pipe = pipe.to(device)
    if hasattr(pipe, "enable_attention_slicing"):
        pipe.enable_attention_slicing()

    log(f"Generating base portrait on {device} ({WIDTH}x{HEIGHT}, {NUM_STEPS} steps, seed={SEED})...")
    t0 = time.time()
    out = pipe(
        prompt=BASE_PROMPT,
        negative_prompt=DEFAULT_NEGATIVE_PROMPT,
        width=WIDTH,
        height=HEIGHT,
        num_inference_steps=NUM_STEPS,
        guidance_scale=GUIDANCE,
        generator=make_generator(device, SEED),
    )
    log(f"Base portrait generated in {time.time() - t0:.1f}s.")
    base_img = out.images[0]
    base_path = HERE / "base.png"
    base_img.save(base_path)
    log(f"Saved {base_path}")

    # free memory
    del pipe
    if device == "mps":
        try:
            torch.mps.empty_cache()
        except Exception:
            pass
    return base_img


def run_img2img_swaps(base_img: Image.Image, device: str) -> list[dict]:
    log("Loading StableDiffusionImg2ImgPipeline...")
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch.float32,
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipe = configure_scheduler(pipe)
    pipe = pipe.to(device)
    if hasattr(pipe, "enable_attention_slicing"):
        pipe.enable_attention_slicing()

    runs = []
    for outfit_name, prompt in OUTFIT_PROMPTS.items():
        for strength in STRENGTHS:
            tag = f"{outfit_name}_s{strength:.2f}"
            out_path = HERE / f"{tag}.png"
            log(f"img2img -> {tag}: strength={strength}")
            t0 = time.time()
            try:
                out = pipe(
                    prompt=prompt,
                    negative_prompt=DEFAULT_NEGATIVE_PROMPT,
                    image=base_img,
                    strength=strength,
                    num_inference_steps=NUM_STEPS,
                    guidance_scale=GUIDANCE,
                    generator=make_generator(device, SEED),
                )
                img = out.images[0]
                img.save(out_path)
                dt = time.time() - t0
                log(f"  saved {out_path.name} in {dt:.1f}s")
                runs.append(
                    dict(
                        image=out_path.name,
                        path=str(out_path),
                        outfit=outfit_name,
                        strength=strength,
                        seconds=round(dt, 1),
                    )
                )
            except Exception as e:
                log(f"  FAILED {tag}: {e!r}")
                runs.append(
                    dict(
                        image=out_path.name,
                        path=str(out_path),
                        outfit=outfit_name,
                        strength=strength,
                        seconds=None,
                        error=repr(e),
                    )
                )
    del pipe
    if device == "mps":
        try:
            torch.mps.empty_cache()
        except Exception:
            pass
    return runs


# ------------------------------ similarity ------------------------------------


def similarity_clip(images: list[tuple[str, Image.Image]]) -> dict[str, float] | None:
    """Return cosine sim of each image vs the first (base) using CLIP image embeddings."""
    try:
        from transformers import CLIPModel, CLIPProcessor
    except Exception as e:
        log(f"CLIP imports failed: {e!r}")
        return None
    try:
        model_id = "openai/clip-vit-base-patch32"
        log(f"Loading CLIP model {model_id} for similarity...")
        model = CLIPModel.from_pretrained(model_id)
        processor = CLIPProcessor.from_pretrained(model_id)
        model.eval()
        device = "cpu"  # similarity is fast on cpu, avoid mps quirks
        model.to(device)
        feats = []
        with torch.no_grad():
            for name, img in images:
                inputs = processor(images=img, return_tensors="pt").to(device)
                emb = model.get_image_features(**inputs)
                emb = emb / emb.norm(dim=-1, keepdim=True)
                feats.append((name, emb))
        base_name, base_emb = feats[0]
        sims = {}
        for name, emb in feats:
            cos = float((base_emb * emb).sum().item())
            sims[name] = cos
        return sims
    except Exception as e:
        log(f"CLIP similarity failed: {e!r}")
        traceback.print_exc()
        return None


def similarity_phash(images: list[tuple[str, Image.Image]]) -> dict[str, float]:
    """Tiny pure-numpy perceptual hash similarity fallback (1 - normalized hamming distance)."""
    def phash(img: Image.Image, hash_size: int = 16) -> np.ndarray:
        g = img.convert("L").resize((hash_size * 4, hash_size * 4), Image.LANCZOS)
        a = np.asarray(g, dtype=np.float32)
        # 2D DCT-II via FFT trick approximation: use a simple DCT
        from numpy.fft import fft2
        f = np.real(fft2(a))[:hash_size, :hash_size]
        med = np.median(f)
        return (f > med).flatten()

    base_name, base_img = images[0]
    base_h = phash(base_img)
    sims = {}
    for name, img in images:
        h = phash(img)
        ham = int(np.count_nonzero(base_h != h))
        sims[name] = 1.0 - ham / base_h.size
    return sims


def similarity_ssim_center(images: list[tuple[str, Image.Image]]) -> dict[str, float]:
    """Last-resort: center-crop grayscale correlation (NOT a true SSIM)."""
    def feat(img: Image.Image) -> np.ndarray:
        w, h = img.size
        s = min(w, h)
        left = (w - s) // 2
        top = (h - s) // 2
        c = img.crop((left, top, left + s, top + s)).convert("L").resize((128, 128), Image.LANCZOS)
        a = np.asarray(c, dtype=np.float32)
        a -= a.mean()
        n = np.linalg.norm(a) + 1e-9
        return a / n

    base_f = feat(images[0][1])
    sims = {}
    for name, img in images:
        f = feat(img)
        sims[name] = float((base_f * f).sum())
    return sims


def write_scores_csv(rows: list[dict], path: Path) -> None:
    cols = [
        "image",
        "outfit",
        "strength",
        "similarity_metric",
        "similarity_score",
        "seconds",
    ]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in cols})
    log(f"Wrote {path}")


def main() -> int:
    HERE.mkdir(parents=True, exist_ok=True)
    device = pick_device()
    log(f"Device: {device}")
    log(f"torch={torch.__version__}")
    try:
        import diffusers, transformers  # noqa: F401
        log(f"diffusers={diffusers.__version__} transformers={transformers.__version__}")
    except Exception:
        pass

    t_start = time.time()
    try:
        base_img = generate_base(device)
    except Exception as e:
        log(f"Base generation failed on {device}: {e!r}")
        if device != "cpu":
            log("Retrying base generation on CPU...")
            device = "cpu"
            base_img = generate_base(device)
        else:
            raise

    try:
        runs = run_img2img_swaps(base_img, device)
    except Exception as e:
        log(f"img2img loop failed on {device}: {e!r}")
        if device != "cpu":
            log("Retrying img2img on CPU...")
            device = "cpu"
            runs = run_img2img_swaps(base_img, device)
        else:
            raise

    # Similarity pass
    images = [("base.png", base_img)]
    for r in runs:
        p = Path(r["path"])
        if p.exists():
            images.append((r["image"], Image.open(p).convert("RGB")))

    metric = "clip"
    sims = similarity_clip(images)
    if sims is None:
        log("Falling back to perceptual hash similarity")
        metric = "phash"
        sims = similarity_phash(images)
    if not sims:
        log("Falling back to center-crop grayscale correlation")
        metric = "ssim_center"
        sims = similarity_ssim_center(images)

    rows = []
    base_score = sims.get("base.png", 1.0)
    rows.append(
        dict(
            image="base.png",
            outfit="base",
            strength=0.0,
            similarity_metric=metric,
            similarity_score=round(base_score, 4),
            seconds="",
        )
    )
    for r in runs:
        s = sims.get(r["image"])
        rows.append(
            dict(
                image=r["image"],
                outfit=r["outfit"],
                strength=r["strength"],
                similarity_metric=metric,
                similarity_score=(round(s, 4) if s is not None else "ERR"),
                seconds=r.get("seconds", ""),
            )
        )

    write_scores_csv(rows, HERE / "scores.csv")

    log("=== Summary ===")
    for r in rows:
        log(
            f"  {r['image']:32s} outfit={r['outfit']:>6s} "
            f"strength={r['strength']:.2f} "
            f"{r['similarity_metric']}_sim={r['similarity_score']}"
        )
    log(f"Total elapsed: {(time.time() - t_start)/60:.1f} min")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        log(f"FATAL: {e!r}")
        traceback.print_exc()
        sys.exit(1)
