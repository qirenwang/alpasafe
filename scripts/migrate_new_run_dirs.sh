#!/bin/bash
# Incrementally migrate NEW top-level generated run/artifact directories to
# /storage, using the exact method validated in migration run
# 20260713T053439Z. Whole-directory granularity only: a directory is either
# entirely a real directory on the system SSD (not yet processed) or
# entirely a symlink to /storage (migrated). Never splits files within one
# directory. Size is NOT used as a filter — every matching directory that
# isn't already a symlink is migrated, regardless of how small it is.
#
# Scope (same two namespaces as the original migration):
#   external/alpasim/t2[0-9]*        (AlpaSim generation/dump/smoke run dirs)
#   safeworld-alpamayo/artifacts/*   (frozen experiment-family dirs)
#
# Usage:
#   bash scripts/migrate_new_run_dirs.sh --dry-run   # list what would move, touch nothing
#   bash scripts/migrate_new_run_dirs.sh              # actually migrate
#
# Run this only after a batch of experiments has finished (no active
# AlpaSim/Docker process should be using the directories being migrated).
set -uo pipefail
DRY=0
[ "${1:-}" = "--dry-run" ] && DRY=1

ALP=/home/qiren/alpasafe/external/alpasim
ART=/home/qiren/alpasafe/safeworld-alpamayo/artifacts
DALP=/storage/alpasafe/external/alpasim
DART=/storage/alpasafe/safeworld-alpamayo/artifacts
TS=$(date -u +%Y%m%dT%H%M%SZ)
RUN=/home/qiren/alpasafe_migration_runs/incremental_$TS
mkdir -p "$RUN"/{logs,hashes,rollback}

bash /home/qiren/alpasafe/scripts/check_external_storage.sh > "$RUN/logs/preflight.txt" 2>&1
grep -q '^PREFLIGHT PASS$' "$RUN/logs/preflight.txt" || { echo "preflight failed, see $RUN/logs/preflight.txt"; exit 1; }

docker ps -q | grep -q . && { echo "containers running — finish/stop them before migrating"; exit 1; }

# --- discover: real directories not yet symlinked, in each namespace ---
cd "$ALP"
NEW_ALP=$(find . -maxdepth 1 -type d -name 't2[0-9]*' -printf '%f\n' | sort)
cd "$ART"
NEW_ART=$(find . -maxdepth 1 -type d ! -name '.' -printf '%f\n' | sort)

echo "=== new alpasim run dirs (real dir, not yet symlink) ==="
echo "$NEW_ALP" | tee "$RUN/logs/new_alpasim_dirs.txt"
echo "=== new artifacts children (real dir, not yet symlink) ==="
echo "$NEW_ART" | tee "$RUN/logs/new_artifacts_dirs.txt"

[ "$DRY" = 1 ] && { echo "--dry-run: nothing moved."; exit 0; }
[ -z "$NEW_ALP" ] && [ -z "$NEW_ART" ] && { echo "Nothing new to migrate."; exit 0; }

migrate_one() {  # $1=src_root $2=dst_root $3=name
  local sroot=$1 droot=$2 name=$3
  local src="$sroot/$name" dst="$droot/$name"
  [ -z "$name" ] && return
  [ -L "$src" ] && return  # already migrated
  fuser -s "$src" 2>/dev/null && { echo "SKIP (in use): $src"; return; }
  mkdir -p "$(dirname "$dst")"
  rsync -aHAXS --numeric-ids "$src/" "$dst/" || { echo "RSYNC FAIL: $src"; return; }
  rsync -rlptDHAXS --no-o --no-g -n --itemize-changes "$src/" "$dst/" \
    | grep -v '^\.d.*\./$' > "$RUN/logs/${name}_secondpass.txt"
  if [ -s "$RUN/logs/${name}_secondpass.txt" ]; then
    echo "SECOND PASS DIFF for $name — not cutting over, inspect $RUN/logs/${name}_secondpass.txt"
    return
  fi
  find "$src" -type f -print0 | xargs -0 -P4 -n32 sha256sum > "$RUN/hashes/${name}.sha256" 2>/dev/null
  (cd "$droot" && sha256sum -c --quiet "$RUN/hashes/${name}.sha256" > "$RUN/logs/${name}_destcheck.txt" 2>&1)
  if [ -s "$RUN/logs/${name}_destcheck.txt" ]; then
    echo "HASH MISMATCH for $name — not cutting over, inspect $RUN/logs/${name}_destcheck.txt"
    return
  fi
  mv "$src" "$src.pre_storage_migration_$TS"
  ln -s "$dst" "$src"
  echo "rm '$src' && mv '$src.pre_storage_migration_$TS' '$src'" >> "$RUN/rollback/rollback_commands.sh"
  echo "MIGRATED: $src -> $dst"
}

for name in $NEW_ALP; do migrate_one "$ALP" "$DALP" "$name"; done
for name in $NEW_ART; do migrate_one "$ART" "$DART" "$name"; done

echo "Done. Backups kept as *.pre_storage_migration_$TS — delete manually after you're satisfied."
echo "Rollback commands: $RUN/rollback/rollback_commands.sh"
