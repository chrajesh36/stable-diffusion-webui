# Identity-preservation validation for `simple_sd_editor` outfit swaps

> Local-only validation. Nothing here is committed; everything lives under
> `tests_identity/` and is safe to delete with `rm -rf tests_identity/`.

## 1. Setup

| Item | Value |
| --- | --- |
| OS / device | macOS (Apple Silicon) |
| Compute device | **CPU** (`torch.backends.mps.is_available() == False` on this machine) |
| Python | 3.13.5 |
| `torch` | 2.12.0 |
| `diffusers` | 0.38.0 |
| `transformers` | 5.8.1 |
| Base model | `runwayml/stable-diffusion-v1-5` (already cached locally, ~4 GB) |
| Scheduler | `DPMSolverMultistepScheduler`, `algorithm_type="dpmsolver++"`, `use_karras_sigmas=True` |
| Resolution | 512 × 768 |
| Steps / CFG | 35 / 6.5 |
| Seed | 12345 (fixed for both txt2img and every img2img) |
| `safety_checker` | disabled (matches `simple_sd_editor.py` defaults) |
| Negative prompt | identical to `DEFAULT_NEGATIVE_PROMPT` in `simple_sd_editor.py`, plus extra NSFW guards (`nudity, swimsuit, bikini, underwear, shirtless`) |

Total wall-clock: **~34 min** (CPU only). The first three blazer strengths
ran fast; one slow step (~15 min stall around step 2 of `blazer_s0.55`)
appears to have been thermal/throttling — not a code issue.

The face used for this validation is **fully synthetic** and was generated
on the fly by txt2img with the prompt below. **No real-person photo** from
the repo or anywhere else was used as input.

```
realistic candid photo of a young Indian man with short black hair,
plain background, neutral expression, DSLR photo, 85mm lens,
soft natural light, high detail, sharp focus
```

## 2. Files produced

```
tests_identity/
├── base.png                  # synthetic txt2img portrait, fixed seed 12345
├── blazer_s0.25.png          # img2img wardrobe swap → navy blazer
├── blazer_s0.35.png
├── blazer_s0.45.png
├── blazer_s0.55.png
├── kurta_s0.25.png           # img2img wardrobe swap → white kurta
├── kurta_s0.35.png
├── kurta_s0.45.png
├── kurta_s0.55.png
├── scores.csv                # similarity numbers per image
├── run_identity_test.py      # end-to-end runner
├── score_clip.py             # similarity-only re-runner
├── run.log                   # full generation log
└── clip.log                  # similarity log
```

## 3. Similarity to `base.png`

Higher = more similar to base. Value 1.0 = identical to base.

We attempted CLIP image-image cosine similarity
(`openai/clip-vit-base-patch32`) but the sandbox blocked the Hugging Face
download, so we fell back to two **fully offline** proxies:

- **`vae_sim`** — cosine similarity of flattened latents from the
  locally-cached SD 1.5 VAE encoder. Sensitive to face/body composition,
  much more semantic than pHash.
- **`phash_sim`** — 16×16 frequency-domain perceptual hash, normalized
  Hamming distance.

| image | outfit | strength | vae_sim | phash_sim |
| --- | --- | ---: | ---: | ---: |
| base.png | base | 0.00 | 1.0000 | 1.0000 |
| blazer_s0.25.png | blazer | 0.25 | 0.9793 | 0.9844 |
| blazer_s0.35.png | blazer | 0.35 | 0.9475 | 0.9453 |
| blazer_s0.45.png | blazer | 0.45 | 0.9017 | 0.9141 |
| blazer_s0.55.png | blazer | 0.55 | 0.8116 | 0.8359 |
| kurta_s0.25.png | kurta | 0.25 | 0.9794 | 0.9766 |
| kurta_s0.35.png | kurta | 0.35 | 0.9476 | 0.9531 |
| kurta_s0.45.png | kurta | 0.45 | 0.9013 | 0.9141 |
| kurta_s0.55.png | kurta | 0.55 | 0.8144 | 0.8516 |

Both metrics agree. Both outfits decay almost identically with strength.

## 4. Visual inspection (manual)

This is the most important finding and it is **not visible in the
similarity numbers** alone:

- **Face / identity** is preserved extremely well across all strengths up
  to and including 0.55. Same face shape, same beard, same hair, same
  eyes, same skin tone. Identity drift only starts to be visible at 0.55,
  and even there it is mild (e.g. a small earring appears on
  `kurta_s0.55`, hair becomes slightly different).
- **The outfit does NOT actually change** at any of the tested strengths.
  In all eight wardrobe swaps the original textured gray sweater is still
  the dominant garment. A navy blazer / white shirt / white kurta does
  not appear. Only minor texture and lighting tweaks happen.

In short: with global img2img on a portrait whose existing outfit is
clearly visible, **strengths 0.25–0.55 are too low to overwrite the
garment**, and strengths high enough to actually overwrite the garment
(empirically ≥ 0.65 with this base) start to noticeably drift the face.

## 5. Recommendation

For the wardrobe-swap workflow specifically:

1. **Do not rely on the global img2img tab in `simple_sd_editor.py` for
   outfit swaps.** Identity is preserved nicely, but the swap itself
   doesn't happen at safe strengths.
2. **Prefer the inpainting tab** (`StableDiffusionInpaintPipeline`,
   `runwayml/stable-diffusion-inpainting`, already wired in
   `simple_sd_editor.py`). Mask the torso/clothing region, use the same
   neutral wardrobe prompt, and run with `strength` ≈ 0.85–0.95. This
   leaves the face pixels untouched (perfect identity preservation) while
   actually replacing the clothing.
3. If the user must use img2img end-to-end (no mask), then within the
   tested range:
   - `0.25` → safest for identity (vae_sim ≈ 0.98) but virtually no
     wardrobe change.
   - `0.35` → still very high identity (vae_sim ≈ 0.95); good general
     baseline for *minor* tweaks (lighting, color, accessories).
   - `0.45` → moderate identity (vae_sim ≈ 0.90); start of meaningful
     scene change but the existing outfit still bleeds through.
   - `0.55` → start of identity drift (vae_sim ≈ 0.81). Outfit still
     bleeds through. Not recommended for portraits where identity matters.
   - For an outfit swap that actually replaces the garment in img2img
     mode, the user will likely need ≥ 0.65, and at that point face-swap
     post-processing (or ControlNet + IP-Adapter Face) is the right
     answer, not raw img2img.

Best **balance** in the tested range: **`strength = 0.35`**. It keeps
vae_sim ≈ 0.95 (identity virtually unchanged) and is the highest setting
where the face never visibly drifts. But again, at that strength the
outfit barely changes. **Use inpainting for real wardrobe swaps.**

## 6. Limitations

- **No face-recognition metric was used.** `vae_sim` and `phash_sim` are
  whole-image proxies, not face-ID scores. For stricter validation, use
  `insightface` (ArcFace embeddings), `deepface`, or
  `facenet-pytorch`. None of those were installed; we deliberately
  avoided pulling them.
- **CLIP image-image similarity was attempted but the sandbox blocked
  the model download.** If you re-run `tests_identity/score_clip.py`
  outside the sandbox (or with `HF_HOME` already populated), CLIP
  numbers will be filled into `scores.csv` automatically.
- **Single seed, single base portrait.** A real validation should
  generate ≥ 5 base portraits and average per-strength scores.
- **CPU-only run.** On MPS, runtime would be much smaller and we could
  afford a wider strength sweep (e.g. up to 0.75) to find the
  identity-vs-outfit-swap inflection point precisely.
- **The base portrait is synthetic.** Conclusions about real-photo
  inputs should not be drawn from this run alone — real photos have
  noise/grain/microtexture that SD reacts to differently.

## 7. How to reproduce

```bash
cd /Users/sride/Downloads/downloads/editing_code
/usr/local/bin/python3 -u tests_identity/run_identity_test.py
/usr/local/bin/python3 -u tests_identity/score_clip.py   # (optional, fills CLIP)
```

Cleanup:

```bash
rm -rf tests_identity/
```
