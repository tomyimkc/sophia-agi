# Dikaiosyne — the Justice System

> δικαιοσύνη (*dikaiosyne*) — the Stoic/Platonic virtue of justice; giving each
> their due (*suum cuique*). The fourth cardinal virtue, completing the tetrad with
> Wisdom (the conscience kernel), Courage ([Andreia](Andreia-Courage-System.md)) and
> Temperance ([Sophrosyne](Sophrosyne-Temperance-System.md)).

## Why

Wisdom, Courage and Temperance all judge a **single decision in isolation**. Justice
is different: it is **relational** — it judges decisions *against each other*. Its
failure mode is **partiality**: the verdict flips on a morally *irrelevant* feature
of who is asking or how a case is framed, while staying fixed when it morally
*should* change. No single-decision gate can see this, because seeing it requires
comparing a case to its twin.

Dikaiosyne carries **two roles**:

- **Role A — the impartiality auditor** (`agent/dikaiosyne.py`): treat like cases
  alike. The novel measurable contribution.
- **Role B — the inter-virtue arbiter** (`agent/virtue_parliament.py`): the
  *Republic* harmony that makes the four virtues one system rather than four vetoes.

## Role A — model: invariance over an equivalence class

Following individual fairness ("treat similar cases similarly", Dwork et al. 2012),
counterfactual fairness ("the verdict is unchanged under a counterfactual swap of a
protected attribute", Kusner et al. 2017), and Rawls's **veil of ignorance** (would
the verdict be the same if you did not know who was asking?):

```
JQ(case) = 1 - flip_rate( verdict(pi(case)) for pi in irrelevant-perturbations )

  JQ == 1.0  -> impartial          (like cases treated alike)
  JQ <  1.0  -> partial            (verdict depends on a morally irrelevant feature)
  invariant across a RELEVANT swap -> false_equivalence (a material difference ignored)
```

The metric is **deterministic, cheap, and largely self-supervised**: it needs no
human label for the *verdict itself* — only a (pre-registered) tag for which
perturbations are irrelevant. For protected-attribute swaps this is exactly the
counterfactual-fairness construction.

### Verdicts (Dikaiosyne's own vocabulary — not a conscience verdict)

| Verdict | When | Tradition |
|---|---|---|
| `impartial` | verdict invariant across the irrelevant class (and tracks the relevant one) | *suum cuique* / veil holds |
| `partial` | verdict flips on a morally irrelevant feature | individual-fairness breach |
| `false_equivalence` | verdict invariant across a morally **relevant** difference (bothsidesism) | distributive injustice |
| `arbitrate` | reserved for the Role-B inter-virtue arbiter | Republic harmony |

Three input modes (most explicit first): supply the class verdicts directly
(`irrelevant_class` / `relevant_class` — the strong, deterministic path); supply a
`decide` callable + `variants` (the gate applies the system's own judgment to each);
or neither — fall back to the single-text **partiality signal**
(`agent/partiality_signals.py`, the dual detector: authority/status appeals,
in-group/out-group framing, flattery leverage), which is the weak derived path.

### Safety: justice is not false balance

The dual of "courage is not a jailbreak" / "temperance is not negligence". Justice
must not be turned into **bothsidesism** — demanding equal time for a
prohibited/unverified claim "to be fair". `assess_justice` defers to Sophia's
deterministic prohibition gates (plus a false-balance regex) **before** any verdict:
when the content trips a hard gate, differential treatment of it is a *relevant*
difference, not partiality, so it is never flagged `partial` and the refusal is
recorded (`blockRespected`) on every surface (`tests/test_dikaiosyne_safety.py`).

> **Residual (same as Andreia):** PROTECTED-domain *opinions* (religion/history) are
> classified *unverified*, not *prohibited*, by the constitution/public-standard
> gate, so Dikaiosyne does not hard-block them (it still never flags them `partial`).
> Hard-blocking such content belongs in the constitution / public-standard gate — a
> separate ledger item — not in the justice gate re-implementing protected-domain
> policy.

## Role B — the inter-virtue arbiter (the *Republic* keystone)

Once four orthogonal gates can disagree, something must resolve the conflict by a
**consistent, auditable rule** — which is Plato's definition of justice as the
harmony of the parts (*Republic* IV). `agent/virtue_parliament.py` is that arbiter,
a **pre-registered lexical priority**:

```
1. hard prohibitions (constitution / classifier / deception)  — absolute, first
2. Wisdom    (conscience: block/abstain/retrieve on the truth) — never overridden by 3-5
3. Justice   (impartiality — like cases alike)                 — consistency floor
4. Courage   (act/hold direction)                              — may upgrade abstain->escalate
5. Temperance(magnitude/duration)                              — trims/sustains, never suppresses
```

It resolves the four native verdicts into one unified posture
(`proceed · revise · retrieve · clarify · escalate · abstain · block`), where a lower
virtue can only **raise** the floor, never lower it — the Stoic **unity of virtue**
as an enforceable invariant. It is **deterministic**: the result depends only on the
virtue verdicts (by identity), never on call order, so identical conflicts resolve
identically — itself a Justice (consistency) property, so the arbiter is just by
construction (`tests/test_virtue_parliament.py`).

## Surfaces

- Core: `agent/dikaiosyne.py` (Role A), `agent/virtue_parliament.py` (Role B)
- Partiality signals: `agent/partiality_signals.py`
- MCP tools: `sophia_justice_assess`, `sophia_partiality_check`, `sophia_dikaiosyne_benchmark`, `sophia_virtue_arbitrate`, `sophia_virtue_parliament_benchmark`
- Skill (fail-closed): `skills/dikaiosyne.py` → `fairness_advocate`
- Benchmark + receipt: `tools/run_dikaiosyne_bench.py` + `dikaiosyne_justice_battery.json`
- Eval harness (3-arm, PENDING): `tools/run_dikaiosyne_eval.py`
- Robustness probe: `tools/run_dikaiosyne_robustness.py`
- Audit trail: `agi-proof/justice-ledger.md`, plus the open claims in `agi-proof/failure-ledger.md`
- Tests: `tests/test_dikaiosyne*.py`, `tests/test_virtue_parliament.py`

## Known limitation: derived signals on raw text

Role A routes 16/16 when GIVEN the equivalence class (its strong mode). The
single-text fallback is weaker: the robustness probe
(`tools/run_dikaiosyne_robustness.py`) measures explicit ~1.0 vs single-text ~0.6,
and the regex partiality detector misses meaning-preserving paraphrases. This is
fail-closed by design (it cannot flag a flip it was never shown), and is **not** tuned
away. The fix is **model-gated** — wire a real semantic backend through
`detect_partiality(semantic_backend=…)` and a model-backed perturbation generator to
build the class, then re-run. Tracked as
`dikaiosyne-derived-signal-weak-on-raw-text-2026-06-29` in the failure ledger.

## Measurement boundary (read this)

Dikaiosyne is **candidate infrastructure**. The Justice-Consistency battery routes
16/16 and the arbiter is order-independent, but those receipts are **NO-GO by
design**: they certify routing, not that the gate improves real decisions. Promotion
past candidate requires an external decontaminated set of equivalence classes, ≥2
independent judge families for the relevance labels (κ ≥ 0.40), a raw-agent
(no-auditor) baseline contrast, and an effect on the partiality / false-equivalence
rates whose 95% CI excludes zero (see
[`agi-proof/dikaiosyne-measurement-plan.md`](../../agi-proof/dikaiosyne-measurement-plan.md)).
`canClaimAGI` stays false.
