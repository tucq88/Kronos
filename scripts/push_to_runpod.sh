#!/bin/bash
# Run this LOCALLY to push code (and optionally data) to your RunPod pod.
#
# Usage:
#   ./scripts/push_to_runpod.sh <pod-ssh-host> [--with-data]
#
# Get <pod-ssh-host> from RunPod UI → pod → Connect → SSH over exposed TCP
# It looks like:  root@<ip> -p <port>
# Pass it quoted: ./push_to_runpod.sh "root@123.45.67.89 -p 12345"
#
# Example:
#   ./scripts/push_to_runpod.sh "root@123.45.67.89 -p 12345"
#   ./scripts/push_to_runpod.sh "root@123.45.67.89 -p 12345" --with-data

set -e

if [ -z "$1" ]; then
    echo "Usage: $0 <ssh-host> [--with-data]"
    echo "Example: $0 \"root@123.45.67.89 -p 12345\""
    exit 1
fi

SSH_HOST="$1"
WITH_DATA="${2:-}"
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
REMOTE_DIR="/workspace/Kronos"

# Parse host and optional port / identity file from SSH_HOST string
SSH_IP=$(echo "$SSH_HOST" | awk '{print $1}')
SSH_PORT=$(echo "$SSH_HOST" | grep -oE '\-p [0-9]+' | awk '{print $2}'); SSH_PORT="${SSH_PORT:-22}"
SSH_KEY=$(echo "$SSH_HOST" | grep -oE '\-i [^ ]+' | awk '{print $2}')

SSH_CMD="ssh -p $SSH_PORT"
[ -n "$SSH_KEY" ] && SSH_CMD="$SSH_CMD -i $SSH_KEY"

echo "=== Pushing code to $SSH_IP:$REMOTE_DIR ==="
rsync -avz --no-o --no-g --progress \
    -e "$SSH_CMD" \
    --exclude='.git' --exclude='__pycache__' --exclude='*.pyc' --exclude='.env' \
    --exclude='.venv/' --exclude='venv/' --exclude='env/' --exclude='lib/' \
    --exclude='finetune_csv/data/' \
    --exclude='finetune_csv/finetuned/' \
    "$REPO_DIR/" "$SSH_IP:$REMOTE_DIR/"

if [ "$WITH_DATA" = "--with-data" ]; then
    echo "=== Pushing data file ==="
    $SSH_CMD "$SSH_IP" "mkdir -p $REMOTE_DIR/finetune_csv/data"
    rsync -avz --no-o --no-g --progress \
        -e "$SSH_CMD" \
        "$REPO_DIR/finetune_csv/data/btc_15m.csv" \
        "$SSH_IP:$REMOTE_DIR/finetune_csv/data/btc_15m.csv"
fi

echo "=== Done. Now SSH in and run: ==="
echo "    ssh $SSH_HOST"
echo "    bash /workspace/Kronos/scripts/runpod_setup.sh"
echo "    bash /workspace/Kronos/scripts/runpod_train.sh"
