#!/bin/bash
# Usage: ./scripts/vps_update.sh <user@vps-ip> [-i /path/to/key]
# Or run directly on the VPS: bash /opt/kronos/scripts/vps_update.sh
set -e

REPO_DIR="/opt/kronos"
VPS_IP="$1"; shift

SSH_OPTS=""
while getopts "i:" opt "$@"; do
    case $opt in
        i) SSH_OPTS="-i $OPTARG" ;;
    esac
done

if [ -n "$VPS_IP" ]; then
    ssh $SSH_OPTS "$VPS_IP" "git -C $REPO_DIR pull && systemctl restart kronos && systemctl status kronos --no-pager"
else
    git -C "$REPO_DIR" pull
    systemctl restart kronos
    systemctl status kronos --no-pager
fi
