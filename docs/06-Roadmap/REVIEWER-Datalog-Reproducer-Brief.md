# Reviewer brief — the Datalog provenance-faithfulness reproducer

> **Who this is for:** an independent reviewer with no prior involvement in this
> repo. **Why it exists:** every result in Sophia-AGI so far has been produced or
> labeled by the project itself. This one command lets *you*, on *your* host,
> re-derive the headline gate-faithfulness result from scratch and confirm it —
> trusting no file we committed. **Time required:** ~2 minutes, one command.
>
> Running this is the single highest-leverage thing an external party can do for
> the project: it is what converts a *candidate-grade* claim into an
> *externally-reproduced* one. No amount of additional self-runs by us is a
> substitute.

---

## What you are checking (the exact, bounded claim)

> The Datalog port of the production provenance gate (`provenance_faithful`)
> returns **byte-identical** `{passed, violations}` to the original Python gate on
> **every** committed provenance case × 3 answer variants. **PASS = 0
> divergences** across all 957 comparisons (319 cases × 3 variants).

This is a claim about **logic**, not about a model. That is why it is fully
reproducible from the data alone — no GPU, no network, no LLM judge, no API key,
no randomness. If the two implementations ever disagree on any case, the tool
prints `FAIL` and lists the divergences.

**What this does NOT claim (please hold us to this):**
- It does **not** validate the model-side +12.5pt hallucination delta.
- It is **not** a capability claim and is **not** a claim of AGI. The repo's
  `canClaimAGI` flag stays `false` regardless of this run's outcome.
- A PASS means *"the logic substrate is faithful to the production gate"* — that
  and nothing more.

---

## How to run it (one command)

```bash
git clone https://github.com/tomyimkc/sophia-agi
cd sophia-agi
python3 tools/run_datalog_reproducer.py --print
```

Requirements: **Python 3.11+**. The reproducer path is pure standard library —
no `pip install` should be needed. (If an import fails on your host, run
`pip install -e .` or `pip install -r requirements.txt` from the repo root and
re-run.)

The tool exits `0` on PASS, `1` on FAIL.

---

## What a PASS looks like

```
Datalog provenance-faithfulness REPRODUCER: PASS
  pre-registration SHA-256: 2215ea912acfc52ea56d89355c049c01d5b4cfdb0cfa53c012eb0219f6df6276
  live audit: cases=319 comparisons=957 match=957 diverge=0
  matches pre-registration: True
```

`match=957 diverge=0` and `matches pre-registration: True` is the result we
claim. Anything else — any nonzero `diverge`, any `FAIL`, any hash mismatch — is
a **real finding**, and we want to hear about it.

---

## Why you can trust the run without trusting us

The tool is deliberately built to **trust no committed artifact**:

1. **It hash-pins the input data.** Before doing anything, it recomputes the
   SHA-256 of the two provenance data files on disk and compares them to a
   pre-registration that was frozen *before* the tool was first published. If we
   had quietly swapped the data to make the audit pass, the hashes would mismatch
   and the tool prints `DATA TAMPER / MISMATCH` and `FAIL`. The pinned hashes are:
   - `misattributions.json` → `b2b101d0553b550049ad365865f129897873d00a74629410f3f8a2774f2b320e`
   - `wikidata_snapshot.json` → `977e786d17e1e2ffad3587e328683bc36c46d3e1372f3a0cb37774fc45ed7660`
2. **It re-derives the audit live.** It does not read a committed results file. It
   rebuilds all 319 cases in-process and re-runs *both* the Python gate and the
   Datalog engine on each case × variant, counting divergences itself.
3. **It prints the pre-registration's own SHA-256** (`2215ea9...6276` above) so
   you can confirm we did not silently swap the pre-registration either. This
   hash is published in this brief and in the commit history; if the value the
   tool prints differs from the one here, that is itself a finding.

Independent integrity check you can run yourself:

```bash
sha256sum provenance_bench/data/misattributions.json provenance_bench/data/wikidata_snapshot.json
sha256sum agi-proof/datalog-provenance-audit/reproducer.preregistration.json
```

These should match the three hashes above. The logic check (the live audit) and
the integrity check (the hashes) are independent: a tampered data pack triggers
`FAIL` *even if the live audit still happened to pass*.

### Optional: confirm the negative control works

To convince yourself the tool can actually fail, edit one byte of
`provenance_bench/data/misattributions.json` and re-run — you should get a
`DATA TAMPER / MISMATCH` and exit `1`. Then `git checkout` the file to restore it.

---

## Where the claim is recorded (for cross-checking)

- Pre-registration: `agi-proof/datalog-provenance-audit/reproducer.preregistration.json`
- The reproducer source (readable, ~230 lines): `tools/run_datalog_reproducer.py`
- The full audit harness it mirrors: `tools/run_datalog_provenance_audit.py`
- The Datalog engine + the abstention rule as one Horn clause:
  `agent/datalog_engine.py`, `agent/datalog_provenance.py`
- The pre-registered result entry:
  `agi-proof/failure-ledger.md` → `datalog-provenance-faithful-port-preregistered-2026-06-27`

## How to report back

A copy-paste of the tool's terminal output (the `PASS`/`FAIL` block) is enough.
If it failed, the `--print` flag already lists the first divergences; including
those, plus your `python3 --version` and OS, lets us reproduce your environment.
