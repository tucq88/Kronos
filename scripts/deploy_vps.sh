#!/bin/bash
# One-shot VPS setup. Run as root on a fresh Ubuntu 22.04/24.04 VPS.
# Usage: bash deploy_vps.sh <api_secret>
set -e

API_SECRET="${1:?Usage: $0 <api_secret>}"
REPO_DIR="/opt/kronos"
SERVICE_USER="kronos"
PYTHON="python3"

echo "=== System packages ==="
apt-get update -q
apt-get install -y -q python3 python3-pip python3-venv git

echo "=== Create service user ==="
id -u "$SERVICE_USER" &>/dev/null || useradd -r -s /bin/false -d "$REPO_DIR" "$SERVICE_USER"

echo "=== Clone / update repo ==="
if [ -d "$REPO_DIR/.git" ]; then
    git -C "$REPO_DIR" pull
else
    git clone https://github.com/tucq88/Kronos.git "$REPO_DIR"
fi
chown -R "$SERVICE_USER:$SERVICE_USER" "$REPO_DIR"

echo "=== Python venv + deps ==="
sudo -u "$SERVICE_USER" $PYTHON -m venv "$REPO_DIR/.venv"
sudo -u "$SERVICE_USER" "$REPO_DIR/.venv/bin/pip" install --quiet \
    gunicorn \
    flask \
    torch \
    pandas==2.2.2 \
    numpy \
    einops==0.8.1 \
    "huggingface_hub==0.33.1" \
    matplotlib==3.9.3 \
    tqdm==4.67.1 \
    safetensors==0.6.2 \
    requests \
    hf_transfer

echo "=== Systemd service ==="
cat > /etc/systemd/system/kronos.service <<EOF
[Unit]
Description=Kronos BTC prediction API
After=network.target

[Service]
Type=simple
User=$SERVICE_USER
WorkingDirectory=$REPO_DIR
Environment="API_SECRET=$API_SECRET"
Environment="PORT=8000"
ExecStart=$REPO_DIR/.venv/bin/gunicorn serve:app \
    --workers 1 \
    --threads 2 \
    --bind 0.0.0.0:8000 \
    --timeout 120 \
    --access-logfile /var/log/kronos/access.log \
    --error-logfile /var/log/kronos/error.log
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

mkdir -p /var/log/kronos
chown "$SERVICE_USER:$SERVICE_USER" /var/log/kronos

systemctl daemon-reload
systemctl enable kronos
systemctl restart kronos

echo ""
echo "=== Done ==="
echo "Service status:"
systemctl status kronos --no-pager
echo ""
echo "API running at: http://$(curl -s ifconfig.me):8000"
echo "Test: curl http://$(curl -s ifconfig.me):8000/health"
