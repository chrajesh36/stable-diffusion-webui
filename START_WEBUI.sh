#!/usr/bin/env bash
# START_WEBUI.sh — Launch the full AUTOMATIC1111 stable-diffusion-webui
# tuned for Apple Silicon (MPS) and identity-preserving outfit edits.
#
# Use this for the full-featured WebUI (img2img inpainting + ControlNet +
# IP-Adapter FaceID + ReActor + ADetailer). For the simpler Gradio editor,
# use START_UI.sh instead.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Apple Silicon / MPS friendly env: fall back to CPU when an op isn't
# implemented on MPS instead of crashing.
export PYTORCH_ENABLE_MPS_FALLBACK=1

# Hint URL up front so the user knows where to go once the server is ready.
echo ""
echo "========================================================"
echo " Stable Diffusion WebUI (Apple Silicon / MPS)"
echo " Once the server prints 'Running on local URL', open:"
echo "   http://localhost:7860"
echo "========================================================"
echo ""

# Pass Apple Silicon friendly args to webui.sh. Anything after `--` is
# forwarded through to launch.py / webui.py as COMMANDLINE_ARGS.
export COMMANDLINE_ARGS="${COMMANDLINE_ARGS:-} \
  --skip-torch-cuda-test \
  --upcast-sampling \
  --no-half-vae \
  --opt-split-attention \
  --use-cpu interrogate"

exec ./webui.sh "$@"
