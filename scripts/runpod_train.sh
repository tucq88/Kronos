#!/bin/bash
# Runs fine-tuning on the RunPod pod.
# Run after runpod_setup.sh completes.
set -e

REPO_DIR="/workspace/Kronos"
CONFIG="configs/config_btc_15m.yaml"
GPUS=${GPUS:-1}

cd "$REPO_DIR/finetune_csv"

# Fix config paths to absolute (needed when running from finetune_csv/)
sed -i "s|data_path: \"finetune_csv/|data_path: \"$REPO_DIR/finetune_csv/|g" "$CONFIG"
sed -i "s|base_path: \"finetune_csv/|base_path: \"$REPO_DIR/finetune_csv/|g" "$CONFIG"

echo "=== Starting fine-tuning on $GPUS GPU(s) ==="

if [ "$GPUS" -gt 1 ]; then
    DIST_BACKEND=nccl torchrun --standalone --nproc_per_node=$GPUS \
        train_sequential.py --config "$CONFIG"
else
    python train_sequential.py --config "$CONFIG"
fi

echo "=== Training complete ==="
echo "Model saved to: $REPO_DIR/finetune_csv/finetuned/btc_15m/"
