#!/bin/bash
# Run this LOCALLY to pull the fine-tuned model back from RunPod.
#
# Usage:
#   ./scripts/pull_from_runpod.sh "root@123.45.67.89 -p 12345"

set -e

if [ -z "$1" ]; then
    echo "Usage: $0 <ssh-host>"
    exit 1
fi

SSH_HOST="$1"
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
REMOTE_DIR="/workspace/Kronos/finetune_csv/finetuned/btc_15m"
LOCAL_DIR="$REPO_DIR/finetune_csv/finetuned/btc_15m"

SSH_IP=$(echo "$SSH_HOST" | awk '{print $1}')
SSH_PORT=$(echo "$SSH_HOST" | grep -oE '\-p [0-9]+' | awk '{print $2}'); SSH_PORT="${SSH_PORT:-22}"
SSH_KEY=$(echo "$SSH_HOST" | grep -oE '\-i [^ ]+' | awk '{print $2}')

SSH_CMD="ssh -p $SSH_PORT"
[ -n "$SSH_KEY" ] && SSH_CMD="$SSH_CMD -i $SSH_KEY"

mkdir -p "$LOCAL_DIR"

echo "=== Pulling fine-tuned model from pod ==="
rsync -avz --progress -e "$SSH_CMD" \
    "$SSH_IP:$REMOTE_DIR/" "$LOCAL_DIR/"

echo "=== Done. Model saved to: $LOCAL_DIR ==="
echo "serve.py will automatically pick it up on next start."
