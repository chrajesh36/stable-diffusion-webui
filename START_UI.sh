#!/bin/bash

# Start Simplified Stable Diffusion Image Editor
# Uses the diffusers library for better compatibility on macOS

cd "$(dirname "$0")"

echo "Starting Stable Diffusion Image Editor..."
echo ""
echo "ℹ️  First time startup will download the Stable Diffusion model (~4GB)"
echo "📍 Model will be cached locally for future runs"
echo ""
echo "🌐 Open your browser to: http://localhost:7860"
echo ""
echo "⏸️  Press Ctrl+C to stop the server"
echo ""

/usr/local/bin/python3 simple_sd_editor.py
