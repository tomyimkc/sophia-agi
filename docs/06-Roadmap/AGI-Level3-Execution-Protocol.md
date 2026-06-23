# Sophia AGI — Level 3 Execution Protocol

This file explains, in operational detail, how to carry out the three remaining
Level-3 lanes that currently block `tools/run_agi_verification_gate.py --target level3`:

1. Hidden full comparison (`hidden_full_comparison`)
2. Distribution-shift learning (`distribution_shift`)
3. 30-minute long-horizon autonomy (`long_horizon_30m`)

> Claim boundary / 不誇大聲明: these procedures can produce **machine-checkable
> Level-3 evidence** only when they use real private packs, fresh post-tests,
> substantive autonomous work, and preserved artifacts. Local smoke/candidate runs
> are useful for rehearsal but **do not count** as Level-3 evidence.

---

## Current Level-3 gate rules

The gate checks these exact artifact conditions:

| Lane | Required artifact condition |
|---|---|
| `hidden_full_comparison` | an aggregate JSON under `agi-proof/hidden-reviewer-packs/**` with `sophiaDelta.scorePctVsRaw > 0`, `strictPassVsRaw >= 0`, `visibility == "private-hidden"`, and pack id not containing `smoke` |
| `distribution_shift` | a `*result*.json` under `agi-proof/learning-under-shift/` with `passingSignal == true` and `postTest.totalCases >= 10` |
| `long_horizon_30m` | a `*public-report.json` under `agi-proof/long-horizon-runs/` with `durationSec >= 1800`, `autonomy.substantive == true`, and `humanInterventionCount <= 2` |

Therefore: **do not manually copy smoke/candidate outputs into those gate-scanned
locations.** Only promote artifacts there after the real protocol clears.

---

## Quick local rehearsal benchmark

Use this before spending API/GPU/time:

```bash
python tools/run_level3_candidate_benchmark.py --json
```

It creates a local candidate run under:

```text
agi-proof/level3-candidate-benchmark/<date>-local-smoke/
```

It rehearses all three lanes:

- hidden full comparison with deterministic fake mode responses;
- distribution-shift report with deterministic pre/post scoring;
- long-horizon harness via a short self-test.

This is a **workflow benchmark only**. It intentionally writes to a non-gate path
and marks artifacts as `candidateOnly: true` / `visibility: revealed-after-eval`.

To create fillable real-run starter files in the gitignored `private/` tree:

```bash
python tools/run_level3_candidate_benchmark.py --emit-real-scaffold
```

This writes:

```text
private/hidden-evals/level3-<date>/PACK.json
private/hidden-evals/level3-<date>/responses.{raw,raw_tools,rag_only,gate_only,sophia_full}.json
private/shift/level3-shift-spec-<date>.json
private/long-horizon/30min-<date>.json
```

The whole `private/` tree is ignored by git so hidden prompts/specs do not leak
into the repo. Fill every `<...>` placeholder before a real run.

---

# Lane 1 — Hidden full comparison

## Purpose

Show that `sophia_full` beats ablations on a hidden reviewer pack:

- `raw`
- `raw_tools`
- `rag_only`
- `gate_only`
- `sophia_full`

This tests whether Sophia's full stack adds value beyond retrieval, tools, and the
gate alone.

## Required decisions

| Decision | Recommended default | Why it matters |
|---|---|---|
| Pack owner | someone other than the benchmark runner | prevents leakage and overfitting |
| Pack visibility | `private-hidden` until scoring is frozen | required by the Level-3 gate |
| Case count | 40+ recommended, 20 absolute minimum | small-N hidden packs are too noisy |
| Domains | at least 4: provenance, coding, planning/tool-use, learning/memory | tests generality, not just the source gate |
| Manual semantic review | two independent reviewers for semantic cases | avoids one-model/self-judge bias |
| API/model choices | one local/mock rehearsal, then real models | real result usually requires API keys |

## Pack design

A real pack should be stored outside git, e.g.:

```text
private/hidden-evals/level3-YYYY-MM-DD/PACK.json
```

Minimum shape:

```json
{
  "packId": "level3-private-hidden-YYYY-MM-DD",
  "visibility": "private-hidden",
  "cases": [
    {
      "id": "provenance_001",
      "domain": "philosophy",
      "prompt": "Did Confucius write the Dao De Jing?",
      "materials": [],
      "scoring": {
        "maxPoints": 5,
        "rubric": ["denies false attribution", "names Laozi/Daoist uncertainty"],
        "mustInclude": ["Confucius", {"match": "not", "aliases": ["did not", "No"]}, {"match": "Laozi", "aliases": ["老子"]}],
        "mustAvoid": ["Confucius wrote the Dao De Jing"]
      }
    }
  ]
}
```

Do not use examples copied from public docs as hidden cases. Public examples are
allowed for rehearsal only.

## Generate responses

For a real run, generate one response file per mode:

```text
private/hidden-evals/level3-YYYY-MM-DD/responses.raw.json
private/hidden-evals/level3-YYYY-MM-DD/responses.raw_tools.json
private/hidden-evals/level3-YYYY-MM-DD/responses.rag_only.json
private/hidden-evals/level3-YYYY-MM-DD/responses.gate_only.json
private/hidden-evals/level3-YYYY-MM-DD/responses.sophia_full.json
```

The response file shape is:

```json
{
  "model": "sophia-full-or-ablation-name",
  "responses": {
    "case_id": "answer text"
  },
  "toolLogs": {},
  "memoryDiffs": {}
}
```

If you use API models, you may need one or more of:

```bash
export ANTHROPIC_API_KEY=...
export OPENAI_API_KEY=...
export DEEPSEEK_API_KEY=...
```

The existing private runner `tools/run_hidden_eval_sophia.py` supports Grok/DeepSeek
paths and local CLI paths; choose the backend deliberately.

## Score aggregate

```bash
python tools/run_hidden_eval_full.py \
  --pack private/hidden-evals/level3-YYYY-MM-DD/PACK.json \
  --mode raw=private/hidden-evals/level3-YYYY-MM-DD/responses.raw.json \
  --mode raw_tools=private/hidden-evals/level3-YYYY-MM-DD/responses.raw_tools.json \
  --mode rag_only=private/hidden-evals/level3-YYYY-MM-DD/responses.rag_only.json \
  --mode gate_only=private/hidden-evals/level3-YYYY-MM-DD/responses.gate_only.json \
  --mode sophia_full=private/hidden-evals/level3-YYYY-MM-DD/responses.sophia_full.json \
  --out agi-proof/hidden-reviewer-packs/results/level3-full-aggregate-YYYY-MM-DD.json \
  --manual-review-out agi-proof/hidden-reviewer-packs/results/level3-manual-review-YYYY-MM-DD.md
```

## Promotion criteria

Promote the lane only if:

- pack remained private until scoring;
- `visibility == "private-hidden"`;
- not a smoke/candidate pack;
- `sophia_full` beats `raw`, and preferably also `raw_tools`, `rag_only`, and `gate_only`;
- semantic/manual cases have two-reviewer adjudication;
- no hidden prompt was leaked into training, examples, or public docs.

---

# Lane 2 — Distribution-shift learning

## Purpose

Show that Sophia can learn a new, source-grounded domain under shift:

1. fail or abstain on a pre-test;
2. ingest promoted learning records append-only;
3. improve on a fresh post-test;
4. avoid old-knowledge regression;
5. pass contamination and protected-knowledge checks.

## Required decisions

| Decision | Recommended default | Why it matters |
|---|---|---|
| New domain | a small third-party micro-domain unknown to current data | must actually test shift |
| Learning records | 10–30 reviewed, source-grounded facts | enough signal without turning into memorization |
| Pre-test size | >=10 cases | gate requires multi-case evidence |
| Post-test size | >=10 fresh cases | gate requires `postTest.totalCases >= 10` |
| Old benchmark pack | frozen Sophia provenance pack | checks no regression |
| Backend | `adapter`/local for rehearsal; API or trained adapter for real | real signal may need model/backend keys |

## Create spec

Start from a template:

```bash
python tools/run_distribution_shift.py --template private/shift/level3-shift-spec-YYYY-MM-DD.json
```

Then fill in:

- `learningRecords`: reviewed, promoted records only;
- `preTestPack`: hidden cases before learning;
- `postTestPack`: fresh hidden cases not verbatim in learning records;
- `oldBenchmarkPack`: frozen old-domain stability pack;
- `oldBenchmarkBaselineScorePct`: baseline score before learning.

## Run

```bash
python tools/run_distribution_shift.py private/shift/level3-shift-spec-YYYY-MM-DD.json \
  --backend adapter \
  --out agi-proof/learning-under-shift/shift-result-YYYY-MM-DD.public-report.json
```

If using hosted models, set the backend-specific API key. If using `adapter`, make
sure the adapter endpoint/checkpoint is available first.

## Promotion criteria

The lane is promotable only if:

- `passingSignal == true`;
- `postTest.totalCases >= 10`;
- post score improves over pre score;
- old benchmark does not regress beyond tolerance;
- protected old knowledge hash stays unchanged;
- contamination audit is clean;
- failed/non-promoted records were not appended.

---

# Lane 3 — 30-minute long-horizon autonomy

## Purpose

Show a substantive autonomous run, at least 30 minutes long, with limited human
intervention and durable logs.

## Required decisions

| Decision | Recommended default | Why it matters |
|---|---|---|
| Task | bounded repo repair or Skill Forge loop | enough substance without unsafe external side effects |
| Duration | >=1800 sec wall-clock | required by the gate |
| Human interventions | <=2 | required by the gate |
| Tool permissions | read/write repo + run tests only | avoids unsafe/unbounded actions |
| Output | public report + private/full JSONL log | proof artifact + audit trail |

## Spec

Use the existing template:

```text
agi-proof/long-horizon-runs/templates/30min-repo-repair.json
```

For a real run, edit a dated copy:

```bash
cp agi-proof/long-horizon-runs/templates/30min-repo-repair.json \
   private/long-horizon/30min-YYYY-MM-DD.json
```

Update:

- `runId`: unique dated id;
- `goal`: the exact autonomous objective;
- `steps`: enough real work to be substantive;
- `verification`: true for objective checks.

Do **not** add a meaningless `sleep 1800` step. Duration must come from real work,
not padding.

## Run

```bash
python tools/run_long_horizon.py \
  --spec private/long-horizon/30min-YYYY-MM-DD.json \
  --log agi-proof/long-horizon-runs/level3-30min-YYYY-MM-DD.log.jsonl \
  --report-out agi-proof/long-horizon-runs/level3-30min-YYYY-MM-DD.public-report.json \
  --timeout-sec 2400
```

If interrupted:

```bash
python tools/run_long_horizon.py \
  --resume agi-proof/long-horizon-runs/level3-30min-YYYY-MM-DD.log.jsonl \
  --report-out agi-proof/long-horizon-runs/level3-30min-YYYY-MM-DD.public-report.json
```

If you intervene, log it explicitly:

```bash
python tools/run_long_horizon.py \
  --resume agi-proof/long-horizon-runs/level3-30min-YYYY-MM-DD.log.jsonl \
  --intervene "Approved bounded patch after reviewing diff; no solution hints."
```

## Promotion criteria

The lane is promotable only if the public report has:

```json
{
  "durationSec": 1800,
  "autonomy": {"substantive": true},
  "humanInterventionCount": 0
}
```

with `humanInterventionCount <= 2`, passed verification steps, and artifacts/logs
preserved.

---

# Final Level-3 verification command

After all three real artifacts are produced:

```bash
python tools/run_agi_verification_gate.py --target level3 --run-local-smoke
```

Expected successful result:

```json
{
  "highestMachineVerifiedLevel": "level3",
  "targetPassed": true,
  "canClaimAGI": false
}
```

Even then, correct public wording is **"strong AGI-candidate evidence"**, not
"Sophia is proven AGI".

---

# Questions / decisions for Tom

Before real Level-3 runs, decide:

1. **Which backend/API for hidden full comparison?** Options: local adapter, Grok CLI,
   DeepSeek API, Anthropic/OpenAI wrapper. API keys may be required.
2. **Who creates the private hidden pack?** Ideally a third party or a separate
   reviewer process, not the same agent optimizing Sophia.
3. **Which new distribution-shift domain?** It should be unknown to current Sophia
   data and not public in the repo before the pre-test.
4. **Can a real 30-minute autonomous run write to the repo?** If yes, define the
   allowed write scope and max interventions.
5. **Can candidate artifacts ever be promoted?** Recommended answer: no. Rehearsal
   artifacts stay in `agi-proof/level3-candidate-benchmark/`; real artifacts must be
   regenerated under private-review conditions.

中文摘要: 本文件把 Level 3 三個剩餘證據 lane 的實際操作流程、artifact 條件、命令、
API key 需求、升級標準與禁止誇大聲明全部列清楚。候選/煙測 benchmark 只用來演練,
不能當作真正 Level 3 證據。
