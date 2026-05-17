#!/usr/bin/env python3
"""
Build a single side-by-side strip showing the same input portrait next to the
result of each denoising strength in the sweep. Output:

    tests_identity/webui_runs/sweep_strip.png

Layout (left -> right): input | d=0.55 | d=0.65 | d=0.75
Each panel is labeled with denoising and elapsed seconds.
"""
from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent / "webui_runs"


def load_runs() -> list[tuple[Path, dict]]:
    runs = []
    for d in sorted(ROOT.iterdir()):
        if not d.is_dir():
            continue
        sjson = d / "summary.json"
        res = d / "result.png"
        if not (sjson.exists() and res.exists()):
            continue
        with sjson.open() as f:
            runs.append((d, json.load(f)))
    return runs


def text_bar(panel: Image.Image, label: str) -> Image.Image:
    w = panel.width
    bar_h = max(24, panel.height // 24)
    bar = Image.new("RGB", (w, bar_h), (0, 0, 0))
    draw = ImageDraw.Draw(bar)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", max(14, bar_h - 10))
    except Exception:
        font = ImageFont.load_default()
    draw.text((8, 4), label, fill=(255, 255, 255), font=font)
    out = Image.new("RGB", (w, panel.height + bar_h), (0, 0, 0))
    out.paste(bar, (0, 0))
    out.paste(panel, (0, bar_h))
    return out


def main() -> int:
    runs = load_runs()
    if not runs:
        print("No completed runs to strip.")
        return 1

    first_dir, first_summary = runs[0]
    input_img = Image.open(first_dir / "input.png").convert("RGB")
    w, h = input_img.size

    panels = [text_bar(input_img, "INPUT")]
    for d, s in runs:
        res = Image.open(d / "result.png").convert("RGB").resize((w, h), Image.Resampling.LANCZOS)
        label = f"d={s.get('denoising_strength')}  ({int(round(s.get('elapsed_seconds', 0)))}s)"
        panels.append(text_bar(res, label))

    total_w = sum(p.width for p in panels)
    total_h = max(p.height for p in panels)
    canvas = Image.new("RGB", (total_w, total_h), (0, 0, 0))
    x = 0
    for p in panels:
        canvas.paste(p, (x, 0))
        x += p.width

    out_path = ROOT / "sweep_strip.png"
    canvas.save(out_path)
    print(f"Wrote {out_path}  ({canvas.width}x{canvas.height})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
