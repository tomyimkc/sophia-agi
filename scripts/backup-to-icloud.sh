#!/usr/bin/env bash
#
# Safe one-way backup of this repo into an iCloud Drive folder.
#
# WHY one-way + exclude .git: a live git working tree inside iCloud Drive corrupts
# (placeholder eviction, partial syncs, "file 2" conflict copies). Git already
# syncs your code via the GitHub remote. This script ONLY mirrors a copy of the
# non-git files into iCloud for backup/cross-device viewing — it never moves the
# repo into iCloud, and it refuses to run if the repo already lives there.
#
# Secrets (.env) and build artifacts are excluded.
#
# Usage:
#   scripts/backup-to-icloud.sh                 # back up to the default iCloud folder
#   scripts/backup-to-icloud.sh -n              # dry run (show what would copy)
#   scripts/backup-to-icloud.sh --mirror        # also delete extras in the backup (exact mirror)
#   ICLOUD_BACKUP_DIR=~/path scripts/backup-to-icloud.sh   # custom destination
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${ICLOUD_BACKUP_DIR:-$HOME/Library/Mobile Documents/com~apple~CloudDocs/SophiaBackup}"
DRY=()
MIRROR=()

while [ $# -gt 0 ]; do
  case "$1" in
    -n|--dry-run) DRY=(--dry-run) ;;
    --mirror)     MIRROR=(--delete) ;;
    --dest)       shift; DEST="$1" ;;
    -h|--help)    sed -n '2,20p' "$0"; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
  shift
done

# Footgun guard: never back up FROM inside iCloud (that means the repo itself is in
# iCloud — the thing we're telling you not to do).
case "$REPO_ROOT" in
  *"Mobile Documents"*|*"com~apple~CloudDocs"*)
    echo "ERROR: the repo is inside iCloud Drive ($REPO_ROOT)." >&2
    echo "Move it out (e.g. ~/dev/sophia-agi) and git-clone there; iCloud must not hold .git." >&2
    exit 1 ;;
esac

EXCLUDES=(
  --exclude '.git/'                         # git is synced via GitHub, never iCloud
  --exclude '.env' --exclude '.env.*'       # secrets must never leave the machine
  --exclude '__pycache__/' --exclude '*.pyc'
  --exclude '.venv/' --exclude 'venv/' --exclude 'node_modules/'
  --exclude 'training/lora/checkpoints/'    # model weights (large, regenerable)
  --exclude 'agent/memory/'                 # runtime caches
  --exclude '.DS_Store'
)

mkdir -p "$DEST"
echo "Backing up:"
echo "  from: $REPO_ROOT"
echo "  to:   $DEST"
[ ${#DRY[@]} -gt 0 ] && echo "  (dry run — nothing will be written)"
[ ${#MIRROR[@]} -gt 0 ] && echo "  (mirror — extras in the backup will be deleted)"

rsync -ah --info=stats1 "${DRY[@]}" "${MIRROR[@]}" "${EXCLUDES[@]}" "$REPO_ROOT"/ "$DEST"/

echo "Done. Reminder: this is a BACKUP copy — keep editing the repo in its git folder, not here."
