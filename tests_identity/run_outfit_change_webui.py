#!/usr/bin/env python3
"""
Outfit-change quality test against the local AUTOMATIC1111 WebUI API.

Drives a single img2img inpainting request with:
  - ControlNet Unit 0: IP-Adapter FaceID Plus v2 (SDXL) — identity lock
  - ControlNet Unit 1: OpenPose (SDXL) — pose lock
  - ADetailer face refinement
  - Inpaint Only Masked + high mask blur + Inpaint masked mode

It validates the four-criteria definition of "good" clothing change:

    1. same face
    2. same posture
    3. same body
    4. only clothing change

The script does NOT spin up the WebUI. Start it first with:

    ./START_WEBUI.sh

then in another terminal, run:

    python tests_identity/run_outfit_change_webui.py \
        --image  tests_identity/base.png \
        --prompt "wearing a tailored navy blue blazer over a crisp white dress shirt, photorealistic, sharp fabric texture"

Outputs land in tests_identity/webui_runs/<timestamp>/.

Usage notes
-----------
* If you do not pass --mask, a coarse central-torso rectangle mask is auto-built
  so you can smoke-test end-to-end. Pass a real brushed mask (white = edit) for
  production-quality results.
* If a ControlNet or ADetailer model is missing in the WebUI, the script will
  print a clear "missing model" message instead of guessing.
* No real-person photos are bundled with this script. The default --image
  points at tests_identity/base.png which is a synthetic txt2img portrait.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

try:
    import requests
except ImportError:
    sys.stderr.write(
        "Missing 'requests'. Install with:\n"
        "  python -m pip install requests pillow\n"
    )
    sys.exit(2)

try:
    from PIL import Image
except ImportError:
    sys.stderr.write(
        "Missing 'pillow'. Install with:\n"
        "  python -m pip install requests pillow\n"
    )
    sys.exit(2)


HERE = Path(__file__).resolve().parent

DEFAULT_NEGATIVE = (
    "cartoon, anime, 3d render, plastic skin, deformed face, distorted eyes, "
    "extra fingers, missing fingers, mangled hands, blurry, low resolution, "
    "watermark, text, logo"
)

DEFAULT_PROMPT = (
    "wearing a tailored navy blue blazer over a crisp white dress shirt, "
    "professional studio lighting, sharp fabric texture, photorealistic"
)


def img_to_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def b64_to_img(b64: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")


def auto_torso_mask(image: Image.Image) -> Image.Image:
    """Build a coarse central torso rectangle. White = edit, black = preserve.

    This is intentionally crude — it covers the middle vertical band of the
    image, which is where the clothing usually is. For real results, brush a
    proper mask in the WebUI Inpaint tab and pass --mask.
    """
    w, h = image.size
    mask = Image.new("L", (w, h), 0)
    # Vertical band: top ~38% (just below shoulders) to bottom ~92%
    top = int(h * 0.38)
    bottom = int(h * 0.92)
    # Horizontal band: middle 70%
    left = int(w * 0.15)
    right = int(w * 0.85)
    px = mask.load()
    for y in range(top, bottom):
        for x in range(left, right):
            px[x, y] = 255
    return mask


def wait_for_webui(base_url: str, timeout_s: int) -> dict:
    """Return /sdapi/v1/options once reachable, else raise SystemExit."""
    deadline = time.time() + timeout_s
    last_err = None
    print(f"Waiting for WebUI at {base_url} (timeout {timeout_s}s) ...")
    while time.time() < deadline:
        try:
            r = requests.get(f"{base_url}/sdapi/v1/options", timeout=4)
            if r.status_code == 200:
                return r.json()
            last_err = f"HTTP {r.status_code}: {r.text[:200]}"
        except Exception as e:  # connection refused, etc.
            last_err = repr(e)
        time.sleep(2)
    sys.exit(
        f"WebUI did not come up within {timeout_s}s. "
        f"Last error: {last_err}\n"
        "Start it with `./START_WEBUI.sh` and wait for 'Running on local URL'."
    )


def find_model(available: list, *needles: str) -> str | None:
    """Pick first available model whose name contains any of the needles (case-insensitive)."""
    names = available
    for needle in needles:
        n = needle.lower()
        for cand in names:
            if n in str(cand).lower():
                return cand
    return None


def list_controlnet_models(base_url: str) -> list[str]:
    try:
        r = requests.get(f"{base_url}/controlnet/model_list", timeout=10)
        if r.status_code == 200:
            return r.json().get("model_list", [])
    except Exception:
        pass
    return []


def list_adetailer_models(base_url: str) -> list[str]:
    try:
        r = requests.get(f"{base_url}/adetailer/v1/ad_model", timeout=10)
        if r.status_code == 200:
            return r.json().get("ad_model", [])
    except Exception:
        pass
    return []


def list_sd_models(base_url: str) -> list[dict]:
    r = requests.get(f"{base_url}/sdapi/v1/sd-models", timeout=10)
    r.raise_for_status()
    return r.json()


def build_payload(
    image_b64: str,
    mask_b64: str,
    width: int,
    height: int,
    prompt: str,
    negative: str,
    steps: int,
    cfg: float,
    denoising: float,
    sampler: str,
    seed: int,
    cn_ip_model: str | None,
    cn_pose_model: str | None,
    adetailer_model: str | None,
) -> dict:
    alwayson: dict = {}

    cn_units = []
    if cn_ip_model:
        cn_units.append(
            {
                "enabled": True,
                "module": "ip-adapter_face_id_plus",
                "model": cn_ip_model,
                "weight": 0.9,
                "image": image_b64,
                "resize_mode": "Crop and Resize",
                "lowvram": False,
                "processor_res": 512,
                "guidance_start": 0.0,
                "guidance_end": 1.0,
                "pixel_perfect": True,
                "control_mode": "Balanced",
            }
        )
    if cn_pose_model:
        cn_units.append(
            {
                "enabled": True,
                "module": "openpose_full",
                "model": cn_pose_model,
                "weight": 0.7,
                "image": image_b64,
                "resize_mode": "Crop and Resize",
                "lowvram": False,
                "processor_res": 512,
                "guidance_start": 0.0,
                "guidance_end": 1.0,
                "pixel_perfect": True,
                "control_mode": "Balanced",
            }
        )
    if cn_units:
        alwayson["ControlNet"] = {"args": cn_units}

    if adetailer_model:
        alwayson["ADetailer"] = {
            "args": [
                True,
                False,
                {
                    "ad_model": adetailer_model,
                    "ad_prompt": "same person, sharp eyes, natural skin texture, photorealistic",
                    "ad_negative_prompt": "blurry, distorted, plastic skin",
                    "ad_confidence": 0.3,
                    "ad_mask_blur": 4,
                    "ad_denoising_strength": 0.3,
                    "ad_inpaint_only_masked": True,
                    "ad_inpaint_only_masked_padding": 32,
                },
            ]
        }

    return {
        "init_images": [image_b64],
        "mask": mask_b64,
        "prompt": prompt,
        "negative_prompt": negative,
        "steps": steps,
        "cfg_scale": cfg,
        "denoising_strength": denoising,
        "sampler_name": sampler,
        "scheduler": "Karras",
        "width": width,
        "height": height,
        "seed": seed,
        "inpainting_fill": 1,            # 0=fill 1=original 2=latent_noise 3=latent_nothing
        "inpaint_full_res": True,        # "Only masked"
        "inpaint_full_res_padding": 32,
        "mask_blur": 12,
        "inpainting_mask_invert": 0,
        "resize_mode": 0,
        "alwayson_scripts": alwayson,
    }


def composite_side_by_side(original: Image.Image, result: Image.Image, mask: Image.Image, out_path: Path) -> None:
    """Save a triptych: original | mask-overlay | result, for quick visual scoring."""
    h = max(original.height, result.height)
    w_each = max(original.width, result.width)

    def fit(img: Image.Image) -> Image.Image:
        return img.resize((w_each, h), Image.Resampling.LANCZOS)

    orig = fit(original.convert("RGB"))
    res = fit(result.convert("RGB"))
    m = fit(mask.convert("L"))

    red = Image.new("RGB", orig.size, (255, 0, 0))
    red_on_orig = Image.composite(red, orig, m)
    overlay = Image.blend(orig, red_on_orig, alpha=0.5)

    canvas = Image.new("RGB", (w_each * 3, h), (0, 0, 0))
    canvas.paste(orig, (0, 0))
    canvas.paste(overlay, (w_each, 0))
    canvas.paste(res, (w_each * 2, 0))
    canvas.save(out_path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Outfit-change quality test via WebUI API.")
    parser.add_argument("--image", default=str(HERE / "base.png"), help="Source photo (PNG/JPG).")
    parser.add_argument("--mask", default=None, help="Mask PNG (white = clothing, black = preserve). Auto if omitted.")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="Outfit prompt (do NOT describe the face).")
    parser.add_argument("--negative", default=DEFAULT_NEGATIVE)
    parser.add_argument("--denoising", type=float, default=0.65, help="0.55–0.75 sweet spot.")
    parser.add_argument("--cfg", type=float, default=5.5)
    parser.add_argument("--steps", type=int, default=35)
    parser.add_argument("--sampler", default="DPM++ 2M Karras")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--wait", type=int, default=120, help="Seconds to wait for WebUI before failing.")
    parser.add_argument("--no-controlnet", action="store_true", help="Disable ControlNet (faster, no identity lock).")
    parser.add_argument("--no-adetailer", action="store_true", help="Disable ADetailer face refinement.")
    args = parser.parse_args()

    base_url = f"http://{args.host}:{args.port}"

    image_path = Path(args.image).expanduser().resolve()
    if not image_path.exists():
        print(f"ERROR: image not found: {image_path}", file=sys.stderr)
        return 2
    image = Image.open(image_path).convert("RGB")
    w, h = image.size
    # Round to multiple of 8 for SD-friendly sizes.
    w8, h8 = (w // 8) * 8, (h // 8) * 8
    if (w, h) != (w8, h8):
        image = image.resize((w8, h8), Image.Resampling.LANCZOS)

    if args.mask:
        mask = Image.open(args.mask).convert("L").resize(image.size, Image.Resampling.NEAREST)
    else:
        print("No --mask given; auto-generating a coarse central-torso mask.")
        mask = auto_torso_mask(image)

    out_dir = HERE / "webui_runs" / datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    image.save(out_dir / "input.png")
    mask.save(out_dir / "mask.png")

    wait_for_webui(base_url, args.wait)

    sd_models = list_sd_models(base_url)
    model_names = [m.get("model_name") or m.get("title") for m in sd_models]
    print(f"Available SD checkpoints ({len(model_names)}): {model_names[:5]} ...")

    cn_models = list_controlnet_models(base_url) if not args.no_controlnet else []
    ad_models = list_adetailer_models(base_url) if not args.no_adetailer else []
    print(f"ControlNet models found: {len(cn_models)}; ADetailer models found: {len(ad_models)}")

    cn_ip = find_model(cn_models, "ip-adapter-faceid-plusv2_sdxl", "faceid_plusv2_sdxl", "faceid")
    cn_pose = find_model(cn_models, "thibaud_xl_openpose", "openpose_sdxl", "openpose")
    ad_face = find_model(ad_models, "face_yolov8n", "face_yolov8s")

    if not args.no_controlnet:
        if not cn_ip:
            print("WARN: IP-Adapter FaceID SDXL model not found; identity lock disabled.")
        if not cn_pose:
            print("WARN: OpenPose SDXL ControlNet not found; pose lock disabled.")
    if not args.no_adetailer and not ad_face:
        print("WARN: ADetailer face model not found; face refinement disabled.")

    payload = build_payload(
        image_b64=img_to_b64(image),
        mask_b64=img_to_b64(mask),
        width=image.width,
        height=image.height,
        prompt=args.prompt,
        negative=args.negative,
        steps=args.steps,
        cfg=args.cfg,
        denoising=args.denoising,
        sampler=args.sampler,
        seed=args.seed,
        cn_ip_model=None if args.no_controlnet else cn_ip,
        cn_pose_model=None if args.no_controlnet else cn_pose,
        adetailer_model=None if args.no_adetailer else ad_face,
    )

    with (out_dir / "payload.json").open("w") as f:
        # Strip the heavy image data from the saved payload for readability.
        light = {**payload, "init_images": ["<png-b64-stripped>"], "mask": "<png-b64-stripped>"}
        for k, v in light.get("alwayson_scripts", {}).items():
            if k == "ControlNet":
                for u in v["args"]:
                    if "image" in u:
                        u["image"] = "<png-b64-stripped>"
        json.dump(light, f, indent=2)

    print("Submitting /sdapi/v1/img2img ...")
    t0 = time.time()
    r = requests.post(f"{base_url}/sdapi/v1/img2img", json=payload, timeout=15 * 60)
    elapsed = time.time() - t0
    if r.status_code != 200:
        print(f"ERROR: HTTP {r.status_code}\n{r.text[:800]}", file=sys.stderr)
        return 1
    data = r.json()
    images_b64 = data.get("images", [])
    if not images_b64:
        print("ERROR: no images returned in response.", file=sys.stderr)
        return 1

    primary = b64_to_img(images_b64[0])
    primary.save(out_dir / "result.png")

    composite_side_by_side(image, primary, mask, out_dir / "compare.png")

    info = {}
    try:
        info = json.loads(data.get("info", "{}"))
    except Exception:
        pass

    summary = {
        "elapsed_seconds": round(elapsed, 1),
        "seed_used": info.get("seed", args.seed),
        "subseed": info.get("subseed"),
        "sd_model": info.get("sd_model_name") or info.get("sd_model_hash"),
        "sampler": info.get("sampler_name"),
        "steps": info.get("steps"),
        "cfg_scale": info.get("cfg_scale"),
        "denoising_strength": info.get("denoising_strength"),
        "controlnet_ip_model": cn_ip if not args.no_controlnet else None,
        "controlnet_pose_model": cn_pose if not args.no_controlnet else None,
        "adetailer_model": ad_face if not args.no_adetailer else None,
        "prompt": args.prompt,
        "negative_prompt": args.negative,
        "image_size": f"{image.width}x{image.height}",
        "output_dir": str(out_dir),
    }
    with (out_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    print()
    print("=" * 64)
    print("Done.")
    print(f"  Time:         {summary['elapsed_seconds']}s")
    print(f"  Seed:         {summary['seed_used']}")
    print(f"  Checkpoint:   {summary['sd_model']}")
    print(f"  IP-Adapter:   {summary['controlnet_ip_model']}")
    print(f"  OpenPose:     {summary['controlnet_pose_model']}")
    print(f"  ADetailer:    {summary['adetailer_model']}")
    print(f"  Output:       {out_dir}/result.png")
    print(f"  Compare:      {out_dir}/compare.png  (input | mask-overlay | result)")
    print("=" * 64)
    print()
    print("Now score against the four criteria visually:")
    print("  1. same face?      open compare.png, third panel")
    print("  2. same posture?   compare arm/shoulder positions in panel 1 vs 3")
    print("  3. same body?      check skin tone, hair, hands")
    print("  4. only clothing?  background and unmasked regions should match")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
