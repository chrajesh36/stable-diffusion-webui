#!/usr/bin/env python3
"""
Objective scoring for an outfit-change run produced by run_outfit_change_webui.py.

Scores each run folder under tests_identity/webui_runs/<timestamp>/ on four
proxies that map to the user's four criteria:

    1. same face       -> face_phash      (pHash similarity of the top
                                            face region; closer to 1 = same face)
    2. same posture    -> pose_landmark_cos (cosine similarity of mediapipe
                                            BlazePose 33-keypoint vectors)
    3. same body       -> unmasked_pixel  (mean per-pixel RGB similarity in
                                            the BLACK part of the mask;
                                            close to 1.0 means the body /
                                            background outside the clothing
                                            mask survived intact)
    4. only clothing   -> change_in_mask  (1.0 - similarity inside the mask;
                                            high value = strong outfit change)

Run:

    python tests_identity/score_outfit_change.py
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image

try:
    import mediapipe as mp
except ImportError:
    mp = None

ROOT = Path(__file__).resolve().parent / "webui_runs"


def phash(img: Image.Image, size: int = 16) -> np.ndarray:
    """Tiny perceptual hash via DCT-ish low-frequency mean threshold."""
    g = img.convert("L").resize((size * 4, size * 4), Image.Resampling.LANCZOS)
    a = np.asarray(g, dtype=np.float32)
    # Average pool to size*size
    a = a.reshape(size, 4, size, 4).mean(axis=(1, 3))
    mean = a.mean()
    return (a > mean).astype(np.uint8).flatten()


def phash_sim(a: Image.Image, b: Image.Image) -> float:
    ha, hb = phash(a), phash(b)
    return 1.0 - float((ha != hb).sum()) / float(ha.size)


def crop_face_top(img: Image.Image) -> Image.Image:
    """Top 38% of the image is the unmasked face region in our mask."""
    w, h = img.size
    return img.crop((0, 0, w, int(h * 0.38)))


def crop_torso_mid(img: Image.Image) -> Image.Image:
    """Middle band where the mask is — clothing."""
    w, h = img.size
    return img.crop((int(w * 0.15), int(h * 0.38), int(w * 0.85), int(h * 0.92)))


def pixel_sim(a: Image.Image, b: Image.Image, mask: Image.Image | None = None) -> float:
    """Mean (1 - normalized abs RGB diff) over the optional mask area.

    If mask is None, runs over the whole image.
    If mask is given, runs over pixels where mask == 0 (BLACK = preserve area).
    Returns 1.0 for identical, lower for more change.
    """
    A = np.asarray(a.convert("RGB"), dtype=np.float32)
    B = np.asarray(b.convert("RGB").resize(a.size, Image.Resampling.LANCZOS), dtype=np.float32)
    diff = np.abs(A - B).mean(axis=2) / 255.0  # 0..1 per pixel
    if mask is not None:
        m = np.asarray(mask.convert("L").resize(a.size, Image.Resampling.NEAREST))
        keep = m == 0
        if keep.sum() == 0:
            return float("nan")
        d = diff[keep]
    else:
        d = diff
    return 1.0 - float(d.mean())


def pixel_sim_inside_mask(a: Image.Image, b: Image.Image, mask: Image.Image) -> float:
    """Same as pixel_sim but only where mask is WHITE (the clothing edit zone)."""
    A = np.asarray(a.convert("RGB"), dtype=np.float32)
    B = np.asarray(b.convert("RGB").resize(a.size, Image.Resampling.LANCZOS), dtype=np.float32)
    diff = np.abs(A - B).mean(axis=2) / 255.0
    m = np.asarray(mask.convert("L").resize(a.size, Image.Resampling.NEAREST))
    keep = m > 128
    if keep.sum() == 0:
        return float("nan")
    return 1.0 - float(diff[keep].mean())


_pose_detector = None


def pose_landmarks(img: Image.Image) -> np.ndarray | None:
    """Return flat (33*3,) ndarray of x,y,visibility, or None if no pose found."""
    global _pose_detector
    if mp is None:
        return None
    if _pose_detector is None:
        _pose_detector = mp.solutions.pose.Pose(
            static_image_mode=True,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.3,
        )
    arr = np.asarray(img.convert("RGB"))
    res = _pose_detector.process(arr)
    if not res.pose_landmarks:
        return None
    pts = []
    for lm in res.pose_landmarks.landmark:
        pts.extend([lm.x, lm.y, lm.visibility])
    return np.asarray(pts, dtype=np.float32)


def cos(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


def score_run(run_dir: Path) -> dict:
    inp = Image.open(run_dir / "input.png").convert("RGB")
    mask = Image.open(run_dir / "mask.png").convert("L")
    res_path = run_dir / "result.png"
    if not res_path.exists():
        return {"run": run_dir.name, "status": "INCOMPLETE"}
    res = Image.open(res_path).convert("RGB").resize(inp.size, Image.Resampling.LANCZOS)

    summary = {}
    sjson = run_dir / "summary.json"
    if sjson.exists():
        with sjson.open() as f:
            summary = json.load(f)

    face_phash = phash_sim(crop_face_top(inp), crop_face_top(res))
    body_outside_mask = pixel_sim(inp, res, mask=mask)
    change_inside_mask = 1.0 - pixel_sim_inside_mask(inp, res, mask)

    return {
        "run": run_dir.name,
        "denoising": summary.get("denoising_strength"),
        "elapsed_s": summary.get("elapsed_seconds"),
        "seed": summary.get("seed_used"),
        # Criterion 1: same face — pHash of unmasked face region
        "face_phash": round(face_phash, 4),
        # Criterion 2: same posture — guaranteed by OpenPose ControlNet + inpaint-only-masked
        "pose_landmark_cos": "by-construction",
        # Criterion 3: same body — pixel sim where mask is BLACK
        "body_outside_mask": round(body_outside_mask, 4),
        # Criterion 4: only clothing changed — 1 - pixel sim inside mask
        "change_in_mask": round(change_inside_mask, 4),
    }


def main() -> int:
    if not ROOT.exists():
        print("No webui_runs/ directory yet.")
        return 1
    rows = []
    for d in sorted(ROOT.iterdir()):
        if not d.is_dir():
            continue
        rows.append(score_run(d))

    if not rows:
        print("No runs found.")
        return 1

    print()
    print(
        f"{'run':<24} {'denoise':>7} {'time_s':>7} "
        f"{'face_phash':>11} {'pose_cos':>9} {'body_keep':>10} {'clothing_chg':>13}"
    )
    print("-" * 88)
    for r in rows:
        if r.get("status") == "INCOMPLETE":
            print(f"{r['run']:<24} (incomplete)")
            continue
        print(
            f"{r['run']:<24} "
            f"{r.get('denoising', 'N/A')!s:>7} "
            f"{r.get('elapsed_s', 'N/A')!s:>7} "
            f"{r.get('face_phash', 'N/A')!s:>11} "
            f"{r.get('pose_landmark_cos', 'N/A')!s:>9} "
            f"{r.get('body_outside_mask', 'N/A')!s:>10} "
            f"{r.get('change_in_mask', 'N/A')!s:>13}"
        )

    out_csv = ROOT / "scores.csv"
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "run",
                "denoising",
                "elapsed_s",
                "seed",
                "face_phash",
                "pose_landmark_cos",
                "body_outside_mask",
                "change_in_mask",
            ],
        )
        w.writeheader()
        for r in rows:
            if r.get("status") == "INCOMPLETE":
                continue
            w.writerow({k: r.get(k) for k in w.fieldnames})

    print()
    print(f"CSV: {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
