# Adopting the entity-disjoint held-out split — human-review checklist

> **Why this exists.** `tools/carve_entity_disjoint_split.py` staged a clean,
> entity-disjoint held-out **candidate** at
> `agi-proof/data-health/seib_entity_disjoint_candidate/` (75 cases, machine-proof
> `sharedWithTrain=[]`). Adopting it is a **human decision** (Leiden value 2: humans own
> results) — the agent proposes, you dispose. This is the exact, copy-pasteable runbook
> to review it, promote it to a sealed eval surface, and flip the entity-contamination
> gate to fail-closed on it. It closes failure-ledger
> `entity-decontam-candidate-staged-not-gated-2026-06-29` and advances
> `seib-generalization-split-not-validated-2026-06-23`.
>
> Until every box below is checked, the split stays a candidate and the gate stays a
> diagnostic. Do **not** skip the content review — entity-disjointness is proven
> mechanically, but *correctness* of each attribution trap is not.

---

## 0. What is already proven (so you don't re-check it)

- **Entity-disjointness:** every candidate prompt's recognised entities are disjoint
  from all training-prompt entities. Re-verify any time:
  ```bash
  python tools/assert_entity_decontam.py \
    --eval-file agi-proof/data-health/seib_entity_disjoint_candidate/candidate.jsonl \
    --fail-covered 0
  # expect: shared=0  evalFullyCovered=0  → exit 0
  ```
- **Freshness:** `tests/test_carve_entity_disjoint_split.py::test_staged_candidate_is_fresh`
  fails if the staged file drifts from a re-carve, so what you review is what the tool produces.

## 1. Pre-requisites

- [ ] Clean working tree on a feature branch (run the **git-discipline** skill first —
      `main` moves under you).
- [ ] Baseline green: `python -m pytest -q tests/test_assert_entity_decontam.py
      tests/test_carve_entity_disjoint_split.py` passes.

## 2. Step 1 — Content review (the part only a human can do)

The set is small (75 cases) — review **all** of them:
```bash
python - <<'PY'
import json
p="agi-proof/data-health/seib_entity_disjoint_candidate/candidate.jsonl"
for i,l in enumerate(open(p),1):
    d=json.loads(l); print(f"{i:>3} {d['entities']}  {d['prompt']}")
PY
```
For each case confirm:
- [ ] **Well-formed probe** — it is a genuine attribution/authorship trap, not filler.
- [ ] **Gold answer is derivable** from committed sources (`data/attributions.json`,
      `provenance_bench/data/wikidata_snapshot.json`, the OKF wiki) — so it is gradable.
- [ ] **No teaching-to-the-test** — the *answer* is not present verbatim in training
      (entity-disjointness already prevents the entity from being a training subject; you
      are double-checking the phrasing isn't a memorised train row).
- [ ] **No PII / privacy issue** — these are public figures/works; confirm nothing slipped in.
- [ ] **Balanced** — note the trap-type spread (e.g. "did X write Y" yes/no, cross-tradition
      merge). Top up via the carver if a type is thin (it draws from the existing eval pool).

If any case fails review, fix the upstream source (`data/attributions.json` etc.) or the
eval pool, re-run the carver `--out` to refresh the candidate, and re-review.

## 3. Step 2 — Promote to a sealed eval surface

Convention: every sealed held-out lives under `data/<name>/` with `heldout_v1.jsonl` +
a sealed `manifest.json` (see `data/team_agents_benchmark/`). The staged manifest already
carries the right `contentHash` and `trainingDisjoint:true`, so reuse it:

```bash
mkdir -p data/seib_entity_disjoint
cp agi-proof/data-health/seib_entity_disjoint_candidate/candidate.jsonl \
   data/seib_entity_disjoint/heldout_v1.jsonl
# adopt the manifest, flipping candidateOnly -> false now that a human approved it:
python - <<'PY'
import json, pathlib
src=json.load(open("agi-proof/data-health/seib_entity_disjoint_candidate/manifest.json"))
src.update({"candidateOnly": False, "sealed": True, "humanReviewed": True,
            "files": {"heldout": "heldout_v1.jsonl"}})
pathlib.Path("data/seib_entity_disjoint/manifest.json").write_text(
    json.dumps(src, indent=2, ensure_ascii=False, sort_keys=True)+"\n")
print("wrote data/seib_entity_disjoint/manifest.json")
PY
```
- [ ] `data/seib_entity_disjoint/heldout_v1.jsonl` and `manifest.json` exist; manifest shows
      `sealed:true`, `trainingDisjoint:true`, `candidateOnly:false`.

## 4. Step 3 — Register it as an eval surface + re-check lexical decontam

Add it to the eval set so the existing guards cover it. Edit
`provenance_bench/dataset_guard.py`:
```python
EVAL_GLOBS = [
    "eval/**/*.jsonl",
    "data/wisdom_market_benchmark/*.jsonl",
    "data/seib_entity_disjoint/*.jsonl",   # <-- add: the adopted entity-disjoint split
]
```
Then confirm it is also **lexically** train-disjoint (exact + shingle), not just entity-disjoint:
```bash
python tools/assert_decontam.py            # expect: OK (exit 0)
```
- [ ] `dataset_guard.EVAL_GLOBS` includes the new split.
- [ ] `assert_decontam.py` passes (no exact/shingle leak introduced).

## 5. Step 4 — Flip the entity gate to fail-closed (scoped to the clean split)

The whole-repo eval set is still contaminated by design, so gate the **new split
specifically**. In `.github/workflows/ci.yml`, in the *Data health + registry drift
gates* step, add:
```yaml
          python tools/assert_entity_decontam.py \
            --eval-file data/seib_entity_disjoint/heldout_v1.jsonl --fail-covered 0
```
Verify locally:
```bash
python tools/assert_entity_decontam.py \
  --eval-file data/seib_entity_disjoint/heldout_v1.jsonl --fail-covered 0   # exit 0
```
- [ ] CI now fails closed if the adopted split ever shares a fully-covered entity with train.

## 6. Step 5 — Regenerate artifacts + run the suite

The new `data/seib_entity_disjoint/manifest.json` is a tracked asset now:
```bash
python tools/build_data_registry.py        # registry picks up the new sealed asset
python tools/data_health_report.py         # lineage/reproducibility denominators update
python tools/assert_mix_balance.py         # unaffected (training mix unchanged) — expect OK
python -m pytest -q tests/test_assert_entity_decontam.py tests/test_build_data_registry.py \
  tests/test_data_health_report.py tests/test_data_analyst.py
```
- [ ] `build_data_registry --check`, `data_health_report --check` pass on the regenerated files.
- [ ] Full data-agent suite green.

## 7. Step 6 — Update the failure ledger (honest bookkeeping)

In `agi-proof/failure-ledger.md`:
- [ ] Flip `entity-decontam-candidate-staged-not-gated-2026-06-29` → **Closed** (split adopted,
      `--fail-covered 0` gated in CI), with the adoption commit referenced.
- [ ] Advance `seib-generalization-split-not-validated-2026-06-23`: a genuinely
      entity-disjoint split now exists — note remaining work (third-party authorship,
      `hidden-review-third-party-not-run`) so the claim is not overstated.
- [ ] `python tools/validate_failure_ledger.py --check` passes.

> **Do not** upgrade any public/`published-results.json` wording from this alone: an
> entity-disjoint split removes one contamination confound; it is **not** a frontier or
> AGI claim, and it is still maintainer-authored (not third-party). `canClaimAGI` stays false.

## 8. Rollback

If review or any gate fails, nothing is committed yet — `git checkout -- .` /
`git clean -fd data/seib_entity_disjoint`. The staged candidate under
`agi-proof/data-health/` is untouched and remains the source to re-carve from.

---

### One-glance promotion summary

| Step | Command / edit | Pass condition |
|---|---|---|
| Verify disjoint | `assert_entity_decontam --eval-file <candidate> --fail-covered 0` | exit 0 |
| Review | eyeball all 75 cases | every box in §2 |
| Promote | copy → `data/seib_entity_disjoint/` + sealed manifest | files exist, `candidateOnly:false` |
| Register | add glob in `dataset_guard.EVAL_GLOBS`; `assert_decontam.py` | exit 0 |
| Gate | add `--eval-file … --fail-covered 0` to `ci.yml` | exit 0 |
| Regenerate | `build_data_registry`, `data_health_report` | both `--check` pass |
| Ledger | close/advance the two items; `validate_failure_ledger --check` | exit 0 |
