# Keeping private files in this public repo (git-crypt)

This repo is **public**, but selected files are stored **encrypted** with
[`git-crypt`](https://github.com/AGWA/git-crypt). They upload and download
through git like any other file (your "OneDrive" workflow), but anyone who
views or clones the repo on GitHub sees only encrypted bytes — not the
contents.

## What gets encrypted

Encryption is driven by `.gitattributes`. Anything matching these patterns is
encrypted automatically:

| Pattern | Use for |
|---|---|
| `secret/**` | anything you drop in the `secret/` folder |
| `*.secret.md` | private markdown notes anywhere in the repo |
| `*.secret.json` | private settings/config |
| `*.private.md` | alternative private-markdown name |

Everything else (README.md, code, etc.) stays normal/public.

## One-time setup on a new machine

1. Install git-crypt: `brew install git-crypt` (mac) / `sudo apt install git-crypt` (linux) / `choco install git-crypt` (windows).
2. Clone the repo as usual.
3. Unlock with your key (you must copy `sophia-git-crypt.key` to the machine
   **outside** of git — e.g. a USB stick or password manager, never commit it):

   ```bash
   git-crypt unlock /path/to/sophia-git-crypt.key
   ```

After unlocking, the encrypted files appear as normal text and stay decrypted
on that machine. Edit, `git add`, `git commit`, `git push` as usual — git-crypt
re-encrypts automatically on the way out.

## Verify it's working

```bash
git-crypt status            # shows which files are encrypted vs not
git-crypt status -e         # list only the encrypted files
```

## ⚠️ Important limits (read this)

- **Only file *contents* are hidden.** Filenames, folder names, file sizes, and
  commit messages are still public. Don't put secrets in a *filename*.
- **The key is everything.** Lose `sophia-git-crypt.key` and the encrypted
  files are unrecoverable. Leak it and the protection is gone. Keep backups in
  a password manager.
- **Never commit the key.** It is not tracked by git; keep it that way.
- **History is forever.** If a file was ever pushed *unencrypted*, encrypting it
  now does not remove the old plaintext from git history.
