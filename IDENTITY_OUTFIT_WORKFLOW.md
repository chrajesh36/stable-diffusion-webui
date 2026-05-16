# Identity-Preserving Outfit Change Workflow

Best-in-market recipe for **same person, different outfit** edits on realistic
photos using this AUTOMATIC1111 WebUI fork on Apple Silicon (MPS).

This workflow combines four techniques:

1. **SDXL inpainting** with a photoreal checkpoint (RealVisXL V4.0).
2. **IP-Adapter FaceID Plus v2 (SDXL)** via ControlNet to lock the subject's
   face identity.
3. **OpenPose (SDXL)** via ControlNet to lock the body pose.
4. **ADetailer + ReActor** post-passes to refine and re-impose the original
   face on the final composition.

The result: clothing changes; face, hair, skin tone and pose stay faithful to
the source photo.

---

## 0. Prerequisites

- macOS on Apple Silicon (M-series).
- Python 3.10 available (the webui sets up its own venv on first run).
- ~10 GB free disk space for models.
- This repo's `cursor/improve-realistic-photo-editor` branch checked out.

The setup script in this repo (`./START_WEBUI.sh`) configures MPS env vars
and Apple-Silicon-friendly launch flags for you.

---

## 1. Launch the WebUI

From the repo root:

```bash
./START_WEBUI.sh
```

Wait for the log line `Running on local URL: http://0.0.0.0:7860` (first
launch can take 5–15 minutes while it builds the venv and downloads core
deps), then open:

```
http://localhost:7860
```

If port 7860 is already taken, the launcher will print the actual port.

---

## 2. Select the photoreal SDXL checkpoint

Top-left of the WebUI, in the **Stable Diffusion checkpoint** dropdown,
pick:

```
realvisxlV40_v40Bakedvae.safetensors
```

(Click the small refresh arrow next to the dropdown if you don't see it.)

If that file is missing because the download was blocked or interrupted,
re-run the download manually — see **Manual fallback URLs** at the bottom
of this document.

---

## 3. Open Inpaint upload

1. Go to the **img2img** tab.
2. Inside img2img, switch to the **Inpaint upload** sub-tab.
3. **Image**: drop in the source photo of the person.
4. **Mask**: upload a mask image where **white = clothing region to
   replace** and **black = keep as-is**. Keep the face, hair and hands
   masked black so identity is preserved.

You can build the mask in any image editor, or use the **Inpaint** tab with
the brush and then export the mask.

---

## 4. Prompt and negative prompt

Describe **only the new outfit** (clearly, in concrete terms). Do not
describe the face — let IP-Adapter handle that.

Three example prompts:

- **Formal**:
  `wearing a tailored navy blue blazer over a crisp white dress shirt,
  silver tie clip, professional studio lighting, sharp fabric texture,
  photorealistic`
- **Traditional**:
  `wearing a traditional white kurta with subtle gold embroidery,
  natural daylight, soft shadows, photorealistic, high detail fabric`
- **Casual**:
  `wearing a vintage blue denim jacket over a plain grey t-shirt,
  outdoor natural light, photorealistic, sharp focus`

**Negative prompt** (paste verbatim):

```
cartoon, anime, 3d render, plastic skin, deformed face, distorted eyes, extra fingers, blurry, low resolution, watermark, text, logo
```

---

## 5. Sampler and inpaint settings

In the img2img / Inpaint upload panel:

| Setting              | Value                                  |
| -------------------- | -------------------------------------- |
| Sampling method      | `DPM++ 2M Karras`                      |
| Sampling steps       | 30–40 (start at 35)                    |
| CFG scale            | 4.5–6.5 (start at 5.5)                 |
| Denoising strength   | 0.55–0.75 (start at 0.65)              |
| Inpaint area         | **Only masked**                        |
| Mask blur            | 12                                     |
| Mask mode            | Inpaint masked                         |
| Resize mode          | Just resize (or Crop and resize)       |
| Width × Height       | match source aspect, e.g. 896×1152     |
| Batch count / size   | 1 / 1 to start                         |
| Seed                 | -1 (random); fix seed once you like it |

Higher denoising = more fabric variation but more risk of identity drift;
counter that with stronger ControlNet weights below.

---

## 6. ControlNet Unit 0 — IP-Adapter FaceID (identity lock)

Expand the **ControlNet** panel below the prompt box, click **Unit 0**.

| Field             | Value                                |
| ----------------- | ------------------------------------ |
| Enable            | ✅                                   |
| Pixel Perfect     | ✅                                   |
| Single Image      | drop the **original photo** here     |
| Control Type      | IP-Adapter                           |
| Preprocessor      | `ip-adapter_face_id_plus`            |
| Model             | `ip-adapter-faceid-plusv2_sdxl`      |
| Control Weight    | `0.9`                                |
| Starting Control Step | `0.0`                            |
| Ending Control Step   | `1.0`                            |
| Control Mode      | Balanced                             |
| Resize Mode       | Crop and Resize                      |

This is the single most important setting for identity preservation. If
the model dropdown is empty, see **Manual fallback URLs**.

---

## 7. ControlNet Unit 1 — OpenPose (pose lock, optional but recommended)

Click **Unit 1**.

| Field             | Value                                |
| ----------------- | ------------------------------------ |
| Enable            | ✅                                   |
| Pixel Perfect     | ✅                                   |
| Single Image      | drop the **original photo** here     |
| Control Type      | OpenPose                             |
| Preprocessor      | `openpose_full`                      |
| Model             | `thibaud_xl_openpose`                |
| Control Weight    | `0.7`                                |
| Starting Control Step | `0.0`                            |
| Ending Control Step   | `1.0`                            |

Pose lock matters most when the new outfit has very different silhouette
(e.g. blazer with shoulder pads vs. t-shirt).

---

## 8. ADetailer — refine the face

Expand the **ADetailer** panel.

| Field         | Value                              |
| ------------- | ---------------------------------- |
| Enable ADetailer | ✅                              |
| Model         | `face_yolov8n.pt`                  |
| Detection conf | 0.3                               |
| Mask blur     | 4                                  |
| Denoising strength | `0.3`                         |
| Inpaint only masked | ✅                            |
| Use separate prompt | optional: `same person, sharp eyes, natural skin texture, photorealistic` |

ADetailer auto-detects the face after the main pass and re-inpaints it at
low denoise so eyes/skin recover sharpness.

---

## 9. ReActor — re-impose the source face

Expand the **ReActor** panel inside the **img2img** tab (not txt2img).

| Field             | Value                                |
| ----------------- | ------------------------------------ |
| Enable            | ✅                                   |
| Source Image      | the **original photo**               |
| Face index (source/target) | `0 / 0`                     |
| Restore Face      | `GFPGAN`                             |
| Restore visibility | 1.0                                 |
| Codeformer weight | 0.5 (only if you switch restore to CodeFormer) |
| Upscaler          | None (let img2img handle resolution) |

ReActor runs last; it does a final face swap from the source photo onto
the generated image, which catches anything IP-Adapter / ADetailer missed.

> **Repo note:** the original `Gourieff/sd-webui-reactor` GitHub repo has
> been disabled by GitHub staff. This setup script installs the SFW fork
> at `https://github.com/Gourieff/sd-webui-reactor-sfw.git` instead. The
> UI controls and the ReActor panel inside the img2img tab are identical;
> the SFW version only blocks NSFW face swaps (which we want anyway —
> this workflow is intended for clothing swaps, not nudity).
>
> **macOS arm64 note:** ReActor depends on `onnxruntime` and `insightface`.
> If pip fails to install them on Apple Silicon, see the fork's README:
> https://github.com/Gourieff/sd-webui-reactor-sfw#installation
> The usual fix is `pip install onnxruntime` (CPU build) and ensuring
> `insightface==0.7.3` is installed in the WebUI's venv. If `insightface`
> still fails, you can disable just the ReActor pass (uncheck it in the
> img2img panel) — the IP-Adapter + ADetailer combo alone gets ~90% of
> the identity preservation benefit.

---

## 10. Generate

Click **Generate**. On Apple Silicon expect roughly 60–180 seconds per
1024-ish-pixel SDXL inpaint with two ControlNet units, ADetailer and
ReActor enabled.

After generation, verify all three:

- **Face identity preserved** — same person, not a lookalike.
- **Only clothing changed** — hair, jewelry, background, hands intact.
- **Pose consistent** — limbs in the same positions, no extra arms.

If any of those drift, see Tuning below.

---

## 11. Tuning if results drift

| Symptom                           | Fix                                                                |
| --------------------------------- | ------------------------------------------------------------------ |
| Face looks like a different person | Raise IP-Adapter Control Weight from 0.9 → 1.0–1.1; lower Denoising to 0.5 |
| Outfit barely changed             | Raise Denoising to 0.7–0.8                                         |
| Outfit changed but bled into face | Re-check mask: face/hair must be black; raise Mask blur to 16      |
| Hands mangled                     | Mask hands black; or add `(perfect hands:1.2)` to prompt           |
| Pose subtly wrong                 | Raise OpenPose weight 0.7 → 0.9                                    |
| Skin looks plastic                | Lower CFG to 4.5; add `natural skin texture, fine pores` to prompt |
| Face soft / blurry                | Raise ADetailer denoising 0.3 → 0.4; ensure ReActor enabled        |

The prompt **must** describe the outfit in concrete terms (fabric, color,
silhouette, lighting). Vague prompts ("nice shirt") will produce vague
results.

---

## 12. Privacy and consent

- Only run this on photos **you own** or have explicit permission to edit.
- **Do not** run this on identifiable third parties without their consent.
- **Do not** use this workflow to generate undressing, swimwear, or nudity
  edits of real people. The repo's intent is benign outfit changes (formal
  vs. casual vs. traditional clothing) on photos of yourself or consenting
  subjects.
- Keep generated images private if the subject hasn't approved publication.

If you publish edited images, label them clearly as edited.

---

## Manual fallback URLs

If any of these files is missing from disk after `./START_WEBUI.sh`, the
automated download was blocked (HuggingFace gating, 401/403, network).
Download manually from a browser logged into HuggingFace, then drop into
the listed folder.

| File                                              | Folder                       | URL                                                                                                          |
| ------------------------------------------------- | ---------------------------- | ------------------------------------------------------------------------------------------------------------ |
| `realvisxlV40_v40Bakedvae.safetensors` (~6.5 GB)  | `models/Stable-diffusion/`   | https://huggingface.co/SG161222/RealVisXL_V4.0/resolve/main/RealVisXL_V4.0.safetensors                       |
| Alt (Lightning, faster, slightly different look)  | `models/Stable-diffusion/`   | https://huggingface.co/SG161222/RealVisXL_V4.0_Lightning/resolve/main/RealVisXL_V4.0_Lightning.safetensors  |
| `ip-adapter-faceid-plusv2_sdxl.bin`               | `models/ControlNet/`         | https://huggingface.co/h94/IP-Adapter-FaceID/resolve/main/ip-adapter-faceid-plusv2_sdxl.bin                  |
| `ip-adapter-faceid-plusv2_sdxl_lora.safetensors`  | `models/ControlNet/`         | https://huggingface.co/h94/IP-Adapter-FaceID/resolve/main/ip-adapter-faceid-plusv2_sdxl_lora.safetensors     |
| `thibaud_xl_openpose.safetensors` (rename from `OpenPoseXL2.safetensors`) | `models/ControlNet/` | https://huggingface.co/thibaud/controlnet-openpose-sdxl-1.0/resolve/main/OpenPoseXL2.safetensors |

The IP-Adapter FaceID files require accepting the license on the
HuggingFace model page first.

After dropping files in, click the small refresh arrow next to the
checkpoint dropdown and the ControlNet model dropdown to pick them up
without restarting.
