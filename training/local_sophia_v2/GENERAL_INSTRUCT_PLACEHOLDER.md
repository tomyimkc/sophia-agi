# Missing required input: general instruction-retention slice

The recommended training mix reserves **~10%** for *general instruction-following* data so the
local Sophia model stays a useful assistant instead of collapsing into a narrow
"No, X did not write Y" refusal machine. **The repo does not ship this** — you must bring a
**license-clean** external slice and drop it here as `general_instruct.jsonl`.

Suggested, permissively-licensed sources (verify the license yourself before use):
- a small sample of **Tulu-3 SFT**, **OpenOrca**, **Dolly-15k**, or **OASST1** (chat format).

Format: same chat schema as the other packs —
`{"messages":[{"role":"user",...},{"role":"assistant",...}]}` — assistant-only loss
(MLX `--mask-prompt`).

After adding it, re-run `python tools/build_local_sophia_dataset.py` — the contamination
guard will check it against the held-out eval sets just like every other pack.

> Do **not** synthesize this from the Sophia corpus — that defeats the purpose (retention
> comes from *out-of-domain* generality, not more provenance examples).
