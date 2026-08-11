#!/bin/bash
# Removes AlpaSim-generated Docker leftovers that accumulate from repeated
# experiment runs: exited containers named like a generation run
# (t2[0-9]..._gen/smoke/...) and dangling (untagged, unreferenced) images.
#
# Scoped precisely to avoid touching other users' containers/images on this
# shared Docker daemon: only removes exited containers whose name starts
# with t2<digit> (matching AlpaSim's docker-compose project naming), never
# running containers, never other users' tagged images, never build cache,
# never Docker's data-root itself.
#
# Safe to run periodically — e.g. after each batch of experiments finishes.
#
# Usage:
#   bash scripts/cleanup_alpasim_docker.sh --dry-run   # show what would be removed
#   bash scripts/cleanup_alpasim_docker.sh              # actually remove it
set -uo pipefail
DRY=0
[ "${1:-}" = "--dry-run" ] && DRY=1

TS=$(date -u +%Y%m%dT%H%M%SZ)
LOGDIR=/home/qiren/alpasafe_migration_runs/docker_cleanup_logs
mkdir -p "$LOGDIR"
LOG="$LOGDIR/cleanup_$TS.log"

echo "=== AlpaSim Docker cleanup $TS ===" | tee "$LOG"

CIDS=$(docker ps -a --filter status=exited --format '{{.ID}} {{.Names}}' | awk '$2 ~ /^t2[0-9]/{print $1}')
N=$(echo "$CIDS" | grep -c . || true)
echo "matched exited AlpaSim containers: $N" | tee -a "$LOG"

DANGLING=$(docker images -f dangling=true --format '{{.ID}} {{.Repository}} {{.Size}}')
echo "dangling images:" | tee -a "$LOG"
echo "$DANGLING" | tee -a "$LOG"

OTHER=$(docker ps -a --filter status=exited --format '{{.Names}}' | grep -vE '^t2[0-9]' || true)
[ -n "$OTHER" ] && echo "NOT touching (name doesn't match t2[0-9] pattern): $OTHER" | tee -a "$LOG"

if [ "$DRY" = 1 ]; then
  echo "--dry-run: nothing removed." | tee -a "$LOG"
  exit 0
fi

[ "$N" -eq 0 ] && [ -z "$DANGLING" ] && { echo "Nothing to clean up." | tee -a "$LOG"; exit 0; }

BEFORE=$(df -BG --output=avail / | tail -1 | tr -dc 0-9)

if [ "$N" -gt 0 ]; then
  echo "$CIDS" | xargs -n50 docker rm >> "$LOG" 2>&1
  echo "container removal exit=$?" | tee -a "$LOG"
fi

if [ -n "$DANGLING" ]; then
  echo "$DANGLING" | awk '{print $1}' | xargs -r docker rmi >> "$LOG" 2>&1
  echo "image removal exit=$?" | tee -a "$LOG"
fi

AFTER=$(df -BG --output=avail / | tail -1 | tr -dc 0-9)
echo "SSD avail before=${BEFORE}G after=${AFTER}G freed=$((AFTER-BEFORE))G" | tee -a "$LOG"
docker system df | tee -a "$LOG"
echo "Log saved: $LOG"
