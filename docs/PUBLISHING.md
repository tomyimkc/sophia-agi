# Publishing Sophia AGI to GitHub (one-time)

The local repository is ready at:

`C:\Users\tomyim\Documents\GitHub\sophia-agi`

## Create the public repository

1. Open https://github.com/new
2. **Repository name:** `sophia-agi`
3. **Description:** `Wisdom before intelligence — open corpus for provenance-aware philosophy and AGI-shaped reasoning`
4. **Visibility:** Public
5. **Do not** initialize with README, .gitignore, or license (already in local repo)
6. Click **Create repository**

## Push

```powershell
cd C:\Users\tomyim\Documents\GitHub\sophia-agi
git remote set-url origin https://github.com/tomyimkc/sophia-agi.git
git push -u origin main
```

If `origin` is not set yet:

```powershell
git remote add origin https://github.com/tomyimkc/sophia-agi.git
git push -u origin main
```

## After publish

- Add topics: `agi`, `philosophy`, `nlp`, `training-data`, `provenance`, `sophia`, `open-source`
- Enable **Issues** and **Discussions** (optional)
- Pin the repo on your profile if desired

## Verify

```powershell
python tools/validate_attribution.py
python tools/export_training_jsonl.py
```