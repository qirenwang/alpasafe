#!/bin/bash
# AlpaSafe external-storage preflight guard (fail-closed).
# Run before any Alpamayo/AlpaSim data generation or SafeWorld experiment.
# Exits nonzero unless the external disk and all migrated paths are healthy.
# Created by storage migration run 20260713T053439Z.
set -u
EXPECTED_UUID=40267e65-788c-436f-934f-79af0d9e21ad
MARKER=/storage/.alpasafe_external_storage_marker
ALP=/home/qiren/alpasafe/external/alpasim
ART=/home/qiren/alpasafe/safeworld-alpamayo/artifacts
DST=/storage/alpasafe
MIN_STORAGE_FREE_GIB=200
MIN_SSD_FREE_GIB=50

rc=0
ok()   { echo "OK   $1"; }
warn() { echo "WARN $1"; }
bad()  { echo "FAIL $1"; rc=1; }

# 1. /storage must be the mounted external filesystem
mountpoint -q /storage || bad "/storage is not a mountpoint"
SRC=$(findmnt -no SOURCE /storage 2>/dev/null)
FST=$(findmnt -no FSTYPE /storage 2>/dev/null)
UUID=$(lsblk -no UUID "$SRC" 2>/dev/null)
[ "$FST" = "ext4" ] || bad "/storage fstype=$FST (expected ext4)"
[ "$UUID" = "$EXPECTED_UUID" ] || bad "/storage UUID=$UUID (expected $EXPECTED_UUID)"
[ -f "$MARKER" ] && grep -q "uuid=$EXPECTED_UUID" "$MARKER" \
  && ok "identity marker matches" || bad "identity marker missing/mismatched"

# 2. Destination directories writable
for d in "$DST/external/alpasim" "$DST/safeworld-alpamayo/artifacts"; do
  [ -d "$d" ] && [ -w "$d" ] && ok "writable: $d" || bad "not writable: $d"
done

# 3. Migrated symlinks must resolve (t2* run dirs, artifacts children)
B1=$(find "$ALP" -maxdepth 1 -xtype l 2>/dev/null | wc -l)
B2=$(find "$ART" -maxdepth 1 -xtype l 2>/dev/null | wc -l)
[ "$B1" -eq 0 ] && ok "alpasim run-dir symlinks resolve" || bad "$B1 broken symlinks under $ALP"
[ "$B2" -eq 0 ] && ok "artifacts symlinks resolve" || bad "$B2 broken symlinks under $ART"

# 4. nre-artifacts: healthy if bind-mounted from /storage OR still native on SSD (staged)
NRE=$ALP/data/nre-artifacts
if [ -d "$NRE" ]; then
  N=$(ls "$NRE/all-usdzs" 2>/dev/null | wc -l)
  if findmnt -no SOURCE --target "$NRE" 2>/dev/null | grep -q sda1 && mountpoint -q "$NRE"; then
    [ "$N" -ge 24 ] && ok "nre-artifacts bind mount active ($N usdz)" || bad "nre-artifacts mount empty"
  elif [ "$N" -ge 24 ]; then
    warn "nre-artifacts still on system SSD (bind-mount cutover pending sudo)"
  else
    bad "nre-artifacts unreadable/empty ($N usdz) and no bind mount active"
  fi
else
  bad "nre-artifacts path missing"
fi

# 5. Free space
SF=$(df -BG --output=avail /storage | tail -1 | tr -dc 0-9)
HF=$(df -BG --output=avail /home | tail -1 | tr -dc 0-9)
[ "$SF" -ge "$MIN_STORAGE_FREE_GIB" ] && ok "/storage free ${SF}G" || bad "/storage free ${SF}G < ${MIN_STORAGE_FREE_GIB}G"
[ "$HF" -ge "$MIN_SSD_FREE_GIB" ] && ok "system SSD free ${HF}G" || bad "system SSD free ${HF}G < ${MIN_SSD_FREE_GIB}G"

if [ $rc -eq 0 ]; then echo "PREFLIGHT PASS"; else echo "PREFLIGHT FAIL — do not start data generation"; fi
exit $rc
