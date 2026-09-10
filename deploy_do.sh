#!/usr/bin/env bash
# deploy_do.sh — Push and restart F-Intel on a DigitalOcean Droplet
# Usage: bash deploy_do.sh
set -e

# Load config from .env if present
if [ -f .env ]; then
  export $(grep -v '^#' .env | grep -E 'DO_DROPLET_IP|DO_SSH_USER' | xargs)
fi

REMOTE_USER="${DO_SSH_USER:-root}"
REMOTE_IP="${DO_DROPLET_IP}"
REMOTE_DIR="/opt/fintel"

if [ -z "$REMOTE_IP" ]; then
  echo "ERROR: Set DO_DROPLET_IP in .env or as an environment variable."
  exit 1
fi

echo "==> Syncing project to ${REMOTE_USER}@${REMOTE_IP}:${REMOTE_DIR}/"
rsync -avz --exclude='.git' --exclude='__pycache__' --exclude='*.pyc' \
      --exclude='.env' --exclude='data/*.db' \
      ./ "${REMOTE_USER}@${REMOTE_IP}:${REMOTE_DIR}/"

echo "==> Uploading .env"
scp .env "${REMOTE_USER}@${REMOTE_IP}:${REMOTE_DIR}/.env"

echo "==> Restarting containers on remote"
ssh "${REMOTE_USER}@${REMOTE_IP}" "
  cd ${REMOTE_DIR}
  docker compose pull 2>/dev/null || true
  docker compose build --no-cache fintel
  docker compose up -d
  docker compose logs --tail=30 fintel
"

echo "==> Waiting for readiness probe..."
sleep 10
STATUS=$(curl -s -o /dev/null -w '%{http_code}' "http://${REMOTE_IP}:8082/ready" 2>/dev/null || echo '000')
if [ "$STATUS" = "200" ]; then
  echo "==> F-Intel is READY at http://${REMOTE_IP}:8082"
else
  echo "==> /ready returned HTTP $STATUS — server may still be warming up"
  echo "    Watch logs: ssh ${REMOTE_USER}@${REMOTE_IP} 'docker compose -f ${REMOTE_DIR}/docker-compose.yml logs -f fintel'"
fi
