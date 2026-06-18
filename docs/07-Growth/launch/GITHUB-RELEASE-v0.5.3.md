# GitHub Release v0.5.3 — publish manually

**Tag exists:** `v0.5.3`  
**Create release:** https://github.com/tomyimkc/sophia-agi/releases/new?tag=v0.5.3

If API returns 403, add **Contents → Read and write** to your fine-grained PAT, then:

```bash
python tools/create_github_release.py --tag v0.5.3
```

---

## Release title

`Sophia AGI v0.5.3`

## Release notes (paste below)

# Sophia AGI v0.5.3

### Added

- Portable user skill: `skills/portable/sophia-source-discipline/` (`/sophia-source-discipline`)
- `tools/install_skills.py` — install to `~/.grok/skills/` (+ optional `~/.cursor/skills/`)
- MCP expanded: attribution lookup, domain records, disputes, export corpus (10 tools total)
- `sophia_mcp/` package, `tests/test_mcp_tools.py`
- [Skills-Install.md](https://github.com/tomyimkc/sophia-agi/blob/main/docs/09-Agent/Skills-Install.md)

### Also in this milestone (v0.5.0–v0.5.2)

- **500** training examples (Claude teacher loop)
- Runtime epistemic gate + Claude Sonnet **100%** on all four domain benchmarks
- LoRA experiment harness with benchmark holdout

**Links**

- Thesis: https://tomyimkc.github.io/sophia-agi/
- HF dataset (500 examples): https://huggingface.co/datasets/tomyimkc/sophia-agi-corpus