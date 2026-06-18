# Publishing to GitHub (one-time)

The local repository is ready at:

`C:\Users\tomyim\Documents\GitHub\source-discipline`

## Create the public repository

1. Open https://github.com/new
2. **Repository name:** `source-discipline`
3. **Visibility:** Public
4. **Do not** initialize with README, .gitignore, or license (already in local repo)
5. Click **Create repository**

## Push

```powershell
cd C:\Users\tomyim\Documents\GitHub\source-discipline
git remote add origin https://github.com/tomyimkc/source-discipline.git
git push -u origin main
```

If `origin` already exists:

```powershell
git push -u origin main
```

## After publish

- Enable **Issues** and **Discussions** (optional) under repo Settings
- Add topics: `philosophy`, `nlp`, `training-data`, `provenance`, `open-source`
- Pin `README.md` on your GitHub profile if desired

## Verify

```powershell
python tools/validate_attribution.py
python tools/export_training_jsonl.py
```