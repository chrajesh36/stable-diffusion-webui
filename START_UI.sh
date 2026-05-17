#!/bin/bash

# Start Simplified Stable Diffusion Image Editor
# Uses the diffusers library for better compatibility on macOS

cd "$(dirname "$0")"

echo "Starting Stable Diffusion Image Editor..."
echo ""
echo "ℹ️  First time startup will download the Stable Diffusion model (~4GB)"
echo "📍 Model will be cached locally for future runs"
echo ""
LAN_IP="$(ipconfig getifaddr en0 2>/dev/null || ipconfig getifaddr en1 2>/dev/null || true)"
echo "🌐 This Mac:    http://localhost:7860"
if [ -n "${LAN_IP}" ]; then
  echo "🌐 Same Wi‑Fi:  http://${LAN_IP}:7860  (phone / tablet; Gradio binds 0.0.0.0)"
else
  echo "🌐 Same Wi‑Fi:  http://<LAN-IP>:7860  (find IP in System Settings → Network)"
fi
echo ""
echo "⏸️  Press Ctrl+C to stop the server"
echo ""

/usr/local/bin/python3 simple_sd_editor.py
