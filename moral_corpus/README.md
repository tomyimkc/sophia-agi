# Sophia Public Moral Standard Corpus

**Boundary:** This is *control infrastructure* for Sophia's moral gate, not a learned
moral sense and not a claim of moral consciousness or AGI. `candidateOnly: true`.

## What this is

An **overlapping-consensus** (Rawlsian) public moral standard. There is **no single
universal moral standard**; instead there are:

- a **hard floor** = the genuine *cross-tradition intersection* of moral minima
  (don't kill, don't deceive, don't exploit the vulnerable, honor consent/dignity);
- a **gray zone** = concerns that reasonable traditions weigh differently
  (autonomy vs. care/role duty, criteria of fairness). Gray-zone signals route to the
  **moral parliament**, never to a hard block.

## Two kinds of provenance (do not confuse them)

- **Legitimacy provenance** — *who endorsed a norm and through what process*
  (instruments, deliberation, professional ethics). Used here.
- **Truth provenance** — W3C-PROV evidence that a *factual* claim is correct
  (handled by the fact-check gate, not here).

Normative principles must **not** be routed through the factual provenance gate
(the is/ought distinction): a norm's warrant is legitimacy + reflective endorsement,
not source entailment.

## Layout

```
moral_corpus/
  public_standard.v1.json   # principles + source families (machine-read by the gate)
  sources/                  # legitimacy-provenance notes per source family
  principles/               # human-readable expansion of the hard floor
  contested_cases/          # gray-zone cases that should ESCALATE, with reasons
```

## How it is used

`agent/public_standard_gate.py` loads this corpus and returns one of the seven
conscience verbs (`allow | revise | retrieve | clarify | escalate | abstain | block`).
The corpus is the runtime **treatment**. Benchmark labels live separately in
`eval/moral_public_standard/` and are independently annotated, so the gate is never
scored against its own corpus (no-circularity discipline).

## Governance

The model never edits this corpus autonomously. Changes follow the human-gated
pipeline in `docs/11-Platform/Public-Moral-Standard.md` (candidate principle →
legitimacy provenance → public-standard mapping → adversarial tests → held-out tests
→ over-refusal check → maintainer approval → version bump).

## 中文摘要

本語料庫是 Sophia 道德閘的「功能性控制基礎設施」，不是主觀道德意識，也不是 AGI 證明。
採用「交疊共識」：硬底線為跨傳統交集的道德最低標準；灰色地帶交由道德議會審議。
規範性原則使用「正當性來源」（誰、以何程序背書），不可經事實查核閘處理（is/ought 區分）。
