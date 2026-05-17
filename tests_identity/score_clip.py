#!/usr/bin/env python3
"""Standalone CLIP similarity pass (run after run_identity_test.py).

Reads the existing PNGs in tests_identity/, computes cosine similarity of
each image's CLIP image embedding against base.png, and rewrites scores.csv
with both CLIP and perceptual-hash similarity columns.
"""
from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

HERE = Path(__file__).resolve().parent

ORDER = [
    ("base.png", "base", 0.00),
    ("blazer_s0.25.png", "blazer", 0.25),
    ("blazer_s0.35.png", "blazer", 0.35),
    ("blazer_s0.45.png", "blazer", 0.45),
    ("blazer_s0.55.png", "blazer", 0.55),
    ("kurta_s0.25.png", "kurta", 0.25),
    ("kurta_s0.35.png", "kurta", 0.35),
    ("kurta_s0.45.png", "kurta", 0.45),
    ("kurta_s0.55.png", "kurta", 0.55),
]


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def phash_sim(images: list[tuple[str, Image.Image]]) -> dict[str, float]:
    def phash(img: Image.Image, size: int = 16) -> np.ndarray:
        g = img.convert("L").resize((size * 4, size * 4), Image.LANCZOS)
        a = np.asarray(g, dtype=np.float32)
        from numpy.fft import fft2
        f = np.real(fft2(a))[:size, :size]
        return (f > np.median(f)).flatten()

    base_h = phash(images[0][1])
    return {
        n: 1.0 - int(np.count_nonzero(base_h != phash(im))) / base_h.size
        for n, im in images
    }


def clip_sim(images: list[tuple[str, Image.Image]]) -> dict[str, float] | None:
    try:
        from transformers import CLIPModel, CLIPProcessor
    except Exception as e:
        log(f"CLIP imports failed: {e!r}")
        return None
    try:
        mid = "openai/clip-vit-base-patch32"
        log(f"Loading {mid}...")
        model = CLIPModel.from_pretrained(mid)
        proc = CLIPProcessor.from_pretrained(mid)
        model.eval()
        feats: list[tuple[str, torch.Tensor]] = []
        with torch.no_grad():
            for n, im in images:
                inp = proc(images=im, return_tensors="pt")
                e = model.get_image_features(**inp)
                e = e / e.norm(dim=-1, keepdim=True)
                feats.append((n, e))
        base_e = feats[0][1]
        return {n: float((base_e * e).sum().item()) for n, e in feats}
    except Exception as e:
        log(f"CLIP failed: {e!r}")
        return None


def vae_sim(images: list[tuple[str, Image.Image]]) -> dict[str, float] | None:
    """Embed via the locally-cached SD VAE encoder and cosine-compare flattened means.

    This is sensitive to face/body/composition changes, more so than phash, while
    still being usable fully offline (uses the SD checkpoint already in HF cache).
    """
    try:
        from diffusers import AutoencoderKL
    except Exception as e:
        log(f"diffusers import failed: {e!r}")
        return None
    try:
        log("Loading SD VAE from local cache...")
        vae = AutoencoderKL.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            subfolder="vae",
            torch_dtype=torch.float32,
            local_files_only=True,
        )
        vae.eval()
        feats: list[tuple[str, torch.Tensor]] = []
        with torch.no_grad():
            for n, im in images:
                im = im.convert("RGB")
                a = np.asarray(im, dtype=np.float32) / 127.5 - 1.0
                t = torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0)  # 1x3xHxW
                lat = vae.encode(t).latent_dist.mean  # 1x4xH/8xW/8
                v = lat.flatten(1)
                v = v / v.norm(dim=-1, keepdim=True)
                feats.append((n, v))
        base_v = feats[0][1]
        return {n: float((base_v * v).sum().item()) for n, v in feats}
    except Exception as e:
        log(f"VAE similarity failed: {e!r}")
        return None


def main() -> int:
    images = []
    for fname, _outfit, _s in ORDER:
        p = HERE / fname
        if not p.exists():
            log(f"missing {p}")
            return 1
        images.append((fname, Image.open(p).convert("RGB")))

    clip = clip_sim(images)
    vae = vae_sim(images)
    ph = phash_sim(images)

    rows = []
    for fname, outfit, strength in ORDER:
        rows.append(
            dict(
                image=fname,
                outfit=outfit,
                strength=strength,
                clip_sim=(round(clip[fname], 4) if clip else "ERR"),
                vae_sim=(round(vae[fname], 4) if vae else "ERR"),
                phash_sim=round(ph[fname], 4),
            )
        )

    out = HERE / "scores.csv"
    with out.open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "image",
                "outfit",
                "strength",
                "clip_sim",
                "vae_sim",
                "phash_sim",
            ],
        )
        w.writeheader()
        for r in rows:
            w.writerow(r)
    log(f"Wrote {out}")
    log("=== Similarity table ===")
    print(
        f"{'image':<22s} {'outfit':>6s} {'str':>5s} "
        f"{'clip':>7s} {'vae':>7s} {'phash':>7s}"
    )
    for r in rows:
        print(
            f"{r['image']:<22s} {r['outfit']:>6s} "
            f"{r['strength']:>5.2f} "
            f"{str(r['clip_sim']):>7s} "
            f"{str(r['vae_sim']):>7s} "
            f"{r['phash_sim']:>7.4f}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
