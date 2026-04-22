#!/bin/bash
# Usage: ./scripts/vps_update.sh <vps-ip>
# Or run directly on the VPS: bash /opt/kronos/scripts/vps_update.sh
set -e

REPO_DIR="/opt/kronos"

if [ -n "$1" ]; then
    ssh "root@$1" "git -C $REPO_DIR pull && systemctl restart kronos && systemctl status kronos --no-pager"
else
    git -C "$REPO_DIR" pull
    systemctl restart kronos
    systemctl status kronos --no-pager
fi
