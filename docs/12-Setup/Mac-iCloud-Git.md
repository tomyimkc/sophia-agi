# Mac + iCloud + git — the safe setup

You want your repo on your Mac alongside your iCloud media. Here's how to do that
**without breaking git** — plus a safe backup script and a git-LFS option.

## TL;DR

- **Do NOT put the git repo inside iCloud Drive.** Keep code in git; keep media in
  iCloud; they're different sync systems and they fight.
- Clone the repo to a normal folder **outside** iCloud and let **git** sync it via
  GitHub (you already have the remote).
- If you want repo files *visible/backed-up* in iCloud, use the one-way backup
  script (`scripts/backup-to-icloud.sh`) — it excludes `.git` and secrets.
- If some media must live *inside* the repo, track it with **git-LFS**, not iCloud.

## Why not put the repo in iCloud Drive

A live git working tree inside iCloud Drive corrupts in practice:

- **"Optimize Mac Storage" evicts files** to cloud placeholders. If it evicts
  objects under `.git/`, git reports missing objects and builds fail until iCloud
  re-downloads them.
- **iCloud syncs file-by-file, asynchronously** — it races git's atomic writes,
  corrupts `.git` on partial syncs, and creates conflict copies (`index 2`,
  `config 2`).
- **Two Macs on one iCloud working tree = merge conflicts** iCloud can't resolve.
  Git is the tool built for that.

## The recommended setup

```bash
# 1) clone OUTSIDE iCloud (a plain folder), and sync via git — not iCloud
mkdir -p ~/dev && cd ~/dev
git clone https://github.com/tomyimkc/sophia-agi.git
cd sophia-agi

# 2) day-to-day sync across machines / the cloud is just git:
git pull        # get the latest
git push        # publish your work
```

| What | Where it lives | Synced by |
|---|---|---|
| Code repo (incl. `.git`) | `~/dev/sophia-agi` (outside iCloud) | **git → GitHub** |
| Personal media / large assets | iCloud Drive (as today) | iCloud |
| Media that must live in the repo | the repo, via **git-LFS** | git → GitHub |

## Option A — back up repo files into iCloud (one-way, safe)

If you want a *copy* of the repo in iCloud (cross-device viewing / extra backup)
without putting the live tree there:

```bash
scripts/backup-to-icloud.sh            # mirror non-git files to the default iCloud folder
scripts/backup-to-icloud.sh -n         # dry run (preview)
scripts/backup-to-icloud.sh --mirror   # exact mirror (also deletes extras in the backup)
ICLOUD_BACKUP_DIR="$HOME/Library/Mobile Documents/com~apple~CloudDocs/Sophia" \
  scripts/backup-to-icloud.sh          # custom destination
```

It **excludes `.git/`, `.env` (your API keys!), caches, and model weights**, and it
**refuses to run if the repo is itself inside iCloud**. Keep editing in the git
folder — the iCloud copy is a backup, not your workspace.

## Option B — media *inside* the repo via git-LFS

If your goal is that large media/datasets live in the repo and sync through GitHub
(not iCloud):

```bash
brew install git-lfs && git lfs install
# track the big binary types you keep in-repo:
git lfs track "*.mp4" "*.mov" "*.png" "*.wav" "*.safetensors" "*.gguf"
git add .gitattributes
git add path/to/media && git commit -m "media via LFS" && git push
```
LFS stores big files as pointers in git and the blobs on the LFS server, so clones
stay fast and the media syncs with `git pull`. (Mind GitHub LFS storage/bandwidth
quotas for very large media — for personal video libraries, iCloud-separate is
often better.)

## Which to choose

- **Just want to work on the Mac + keep media in iCloud?** → recommended setup, done.
- **Want a backup copy of the repo in iCloud too?** → add Option A.
- **Want specific media versioned in the repo?** → Option B (git-LFS).
- **Never** → the live repo (or `.git`) inside iCloud Drive.
