#!/bin/bash
# Runs ONCE on the RunPod pod after you SSH in.
# Sets up the environment and downloads BTC data.
set -e

REPO_DIR="/workspace/Kronos"
DATA_DIR="$REPO_DIR/finetune_csv/data"

echo "=== Installing system deps ==="
apt-get update -q && apt-get install -y -q rsync

echo "=== Installing Python deps ==="
pip install --quiet \
    torch>=2.0.0 \
    pandas==2.2.2 \
    numpy \
    einops==0.8.1 \
    huggingface_hub==0.33.1 \
    matplotlib==3.9.3 \
    tqdm==4.67.1 \
    safetensors==0.6.2 \
    requests \
    hf_transfer

unset HF_HUB_ENABLE_HF_TRANSFER

echo "=== Checking BTC 15m data ==="
mkdir -p "$DATA_DIR"
if [ -f "$DATA_DIR/btc_15m.csv" ]; then
    echo "Data already present at $DATA_DIR/btc_15m.csv — skipping download."
else
    echo "Downloading from Binance..."
    cd "$REPO_DIR"
    python finetune_csv/download_btc_15m.py --start 2020-01-01 --out "$DATA_DIR/btc_15m.csv"
fi

echo "=== Setup complete ==="
