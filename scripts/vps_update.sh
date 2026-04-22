#!/bin/bash
# Pull latest code and restart the service.
# Run as root on the VPS.
set -e

REPO_DIR="/opt/kronos"

echo "=== Pulling latest code ==="
git -C "$REPO_DIR" pull

echo "=== Restarting service ==="
systemctl restart kronos
systemctl status kronos --no-pager
