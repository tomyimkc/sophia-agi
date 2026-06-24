# Legal Industry Fit — what Sophia can and cannot tackle

> **Stance.** This is a capability assessment, not a product claim or legal advice.
> Sophia is research code (Apache-2.0) that explicitly **does not claim to be AGI**
> ([README](../../README.md), [RESULTS.md](../../RESULTS.md)). The honest answer to
> "can it tackle legal AI?" is: *it tackles the **discipline** of legal AI; it does
> not yet tackle the **practice** of law.*

Source: *AI in the Legal Industry — A Hong Kong–Centred Global Assessment*. The
document's defining risk is **hallucinated citations** (≈712 decisions worldwide
addressing hallucinated content, ~90% written in 2025; *Mata v. Avianca*,
*Ayinde v Haringey*, and HK's own *Yu Hon Tong Thomas v Centaline* [2025] HKCFI 808
and *Licksun v Occupiers* [2025] HKDC 1287). Its reliability thesis: **anchor every
authority to an official primary source** (Hong Kong e-Legislation, HKLII) and treat
AI as a drafting/research assistant, **never a citator**.

That thesis is, almost word for word, Sophia's own: *"refuse to assert what it
cannot machine-check."* A fabricated case citation and "Confucius wrote the Dao De
Jing" are the **same failure** — a confident attribution with no machine-checkable
provenance.

## ✅ What Sophia can genuinely tackle

| Legal challenge | Sophia mechanism | Where |
|---|---|---|
| Hallucinated citations → sanctions | Verifier-gated loop + **calibrated abstention** (refuse rather than fabricate) | `agent/verifiers.py`, `agent/verifier_synthesis.py` |
| **"Confirm the authority exists"** (the *Mata* killer) | **`legal_citation_exists`** — extracts neutral citations (`[2025] HKCFI 808`) + ordinance refs (`Cap. 614`) and checks each against a trusted register; **fails closed** | `agent/legal_citations.py`, `agent/verifiers.py` |
| **"Confirm the holding supports the proposition"** (the *Ayinde* misstated-authority failure) | **`legal_holding_faithful`** — LLM-judge tier; flags a real authority cited for something its holding doesn't establish. Fail-closed/abstaining; single-judge = illustrative, gated for headlines | `agent/legal_faithfulness.py`, `agent/verifiers.py` |
| "Never ask the AI to verify itself" | No-overclaim gate: ≥2 independent judges (judge ≠ subject), ≥3 runs, inter-judge agreement | [RESULTS.md](../../RESULTS.md), [SECURITY.md](../../SECURITY.md) |
| Non-delegable accountability / human-in-the-loop | Law council `human_review_gatekeeper_seat` + `humanBoundary` (never final on liberty/custody/immigration; "not legal advice" label) | `data/law_council_figures.json` |
| Stale-law risk | `jurisdiction_detector_seat` (which law applies, as-of date, conflict of laws) | same |
| Bilingual common-law (HK's hardest problem) | Native EN + 中文 schema; 法家/HK seats keep historical Legalism and modern statute distinct | `AGENTS.md`, `data/law_council_figures.json` |
| HK-specific framing | Dedicated Hong Kong jurisdiction seat (PDPO, Basic Law) + Mainland/EU/UK/US + cross-border/GBA seats | same |
| Confidentiality / privilege leakage | M2 data-flow firewall: capability classification (READ/WRITE/EGRESS), taint tracking, `no_secret_leak`, airgap kill-switch | `agent/dataflow/`, `agent/verifiers.py` |
| Bias / access to justice | Always-seated `ethics_officer_seat` + `legal_aid_navigator_seat` for self-represented users | `data/law_council_figures.json` |
| Contradiction detection across authorities | OKF belief graph + contradiction detection + min-over-chain confidence | `okf/graph.py` |

The **`legal_citation_exists` verifier** (added with this assessment) is the concrete
bridge. It is the one check that would have stopped *Mata* and *Ayinde*. Honest
scope: it verifies **existence against the supplied register only** — not that the
authority is in the right jurisdiction, that the holding supports the proposition,
or that the law is current. Populate the register from an authoritative source via
`SOPHIA_LEGAL_AUTHORITIES`; the bundled `data/legal_authorities.json` is a tiny
illustrative snapshot, **not** a citator.

```bash
python -c "from agent.verifiers import check_text; \
  print(check_text('legal_citation_exists', 'Per Wong v Lee [2024] HKCFI 9999.'))"
# -> passed: False  (fabricated citation flagged)
```

Benchmark: `benchmark/legal_citations.json` (real-vs-fabricated, HK/UK/US). Tests:
`tests/test_legal_citation.py`.

## ⚠️ Partial — architecture present, substance missing

1. **Citation faithfulness is shallow** at the deterministic layer.
   `citation_faithful` checks lexical overlap (~35% content words) and cannot reason
   about whether a holding *supports* a proposition. *(Now complemented by a
   semantic tier — `legal_holding_faithful`, an LLM-judge check for misstated
   authority; see the build-order section. Honestly bounded: a single judge is
   illustrative, and it abstains when it lacks holding text or a judge.)* Ratio vs.
   obiter reasoning remains out of scope.
2. **The Law Council is scaffolding, not a validated product.**
   `law_council_figures.json` is rich *metadata*; there is no measured legal
   hallucination delta, and per RESULTS.md **0 results have cleared the gate** in any
   domain. The new `legal_citation_exists` benchmark is illustrative, not headline.
3. **RAG points at the wrong corpus.** The retrieval pipeline is real and
   provenance-carrying, but it indexes the 518-example philosophy corpus — not
   statutes or case law.

## ❌ What Sophia fundamentally cannot tackle (today)

1. **No live connection to an authoritative legal source.** The verifier checks a
   *snapshot*; it does **not** query HKLII or e-Legislation. Without a live citator
   backend it abstains from ignorance, not from verification against ground truth.
   *(Subsequently addressed — a live federated resolver now exists; see the
   connector + build-order sections below.)*
2. **No document ingestion for real legal work** — no PDF/DOCX/email parsing, so
   contract review, due diligence and e-discovery (the doc's biggest use cases) are
   out of scope. *(Subsequently addressed for TXT/MD/DOCX/HTML, and PDF with an
   optional parser, via `agent/legal_docs.py` — see build-order item 4. Full
   layout/OCR and e-discovery scale remain out of scope.)*
3. **No real legal training data or measured legal accuracy.** Nothing in the 518
   examples is law; there is no Sophia analogue to the Stanford 17–33% number.
4. **Cantonese (粵語)** — the doc's flagged HK niche. *(Now partially addressed:
   `agent/cantonese.py` detects written Cantonese by its distinctive particles
   (嘅/喺/唔係/咗…) — which Standard Written Chinese lacks — and the agent emits a
   粵語摘要 when the question is Cantonese or `--cantonese` is set. Generation
   quality is still the underlying model's; this adds detection + prompting, not a
   Cantonese-tuned model.)*
5. **Not a deployable, compliant product** — no enterprise confidentiality tier, no
   PDPO/PCPD Model Framework implementation, no audit trail meeting EU AI Act
   high-risk (Annex III) obligations.
6. **The hardest legal reasoning is unproven** — ratio extraction, distinguishing
   precedent, HK/Mainland statutory interpretation. The seats *name* these tasks;
   nothing demonstrates competence.

## The limitation of this "AGI"

This belongs in any honest legal-fit assessment, because the legal industry's whole
problem is **over-trusting confident output**:

- **It is not AGI, and the repo says so.** Sophia is an *AGI-candidate proof
  package* for **provenance-aware reasoning** — a deliberately narrow slice. The
  pre-registered AGI thresholds are **not met** ([agi-proof/](../../agi-proof/)),
  by design and by admission.
- **Its strength is refusal, not capability.** Sophia's contribution is *knowing
  when not to answer*. That is exactly right for a legal context — but it is a
  **guardrail, not a brain**. It does not draft novel legal arguments, weigh
  authorities, or reason about facts at lawyer level. A verifier that says "this
  citation is fabricated" has prevented harm; it has not practiced law.
- **The verifiers are deterministic and lexical.** They catch fabricated, out-of-
  range, or topically mismatched citations — not subtle wrong predicates where the
  subject still matches the source. Genuine semantic faithfulness needs a model
  judge, which reintroduces the very hallucination risk being guarded against.
- **The register is the ceiling.** `legal_citation_exists` is only as good as the
  authority list it is given. With the bundled snapshot it will **false-flag every
  real case not in the file** — safe (fail-closed) but useless until wired to a real
  primary source. This is the gap between the *discipline* and the *practice*.
- **0 validated numbers.** Every public figure is gated and currently illustrative.
  Sophia's honesty about its own limits is its most transferable feature to a field
  being sanctioned for the opposite.

**Bottom line:** Sophia's *epistemic engine* — abstention, provenance gate, citation
verifiers, human-boundary seats, bilingual schema, dataflow firewall, no-overclaim
measurement — is unusually well-suited to legal AI's defining risk. The *substance
layer* (authoritative live data, document ingestion, Cantonese, measured accuracy,
deployable compliance) is absent. It can enforce the **discipline** of legal AI; it
cannot yet do the **work** of a lawyer.

## The connector (live HKLII / e-Legislation backend)

The live backend now exists in `agent/legal_sources/` — a cache-first, fail-closed
resolver that turns `legal_citation_exists` from a snapshot tripwire into a real
citator. Design principles: **fail-closed** (network error/timeout/ambiguous match
→ UNVERIFIED, never a silent pass), **cache-first** (be a polite consumer of free
public-interest services; the cache *is* the snapshot and keeps CI deterministic),
and **send only the citation, never the matter** (the confidentiality rule, enforced
at the boundary).

```text
agent/legal_sources/
  base.py           # LegalSource protocol + Resolution (fail-closed helpers, injectable fetch)
  cache.py          # ResolutionCache — JSON cache; a miss is None (fail-closed)
  elegislation.py   # HK ordinance chapters (Cap. NNN); SOPHIA_ELEGISLATION_BASE
  hklii.py          # HK case law ([2025] HKCFI 808); SOPHIA_HKLII_BASE
  tna.py            # UK case law ([2025] EWHC 1383); SOPHIA_TNA_BASE
  courtlistener.py  # US reporter citations (925 F.3d 1339); SOPHIA_COURTLISTENER_BASE
  registry.py       # route by court token; SOPHIA_LEGAL_SOURCE = off | cache | live
tools/refresh_legal_authorities.py   # batch snapshot refresh (strategy A)
```

```python
from agent.legal_sources import make_resolver
from agent.verifiers import legal_citation_exists

verifier = legal_citation_exists(resolver=make_resolver())   # SOPHIA_LEGAL_SOURCE=live
```

- **Federation:** the registry routes each citation by **court token** —
  e-Legislation (HK ordinances), HKLII (HK case law), National Archives Find Case
  Law (UK), CourtListener (US reporter citations). A US `925 F.3d 1339` and a UK
  `[2025] EWHC 1383` go to different backends automatically.
- **Modes:** `off` (static register only), `cache` (default; cache-first, **no
  network** — a miss is UNVERIFIED), `live` (cache-first, then the routed source,
  then cache the result).
- **Snapshot refresh:** `SOPHIA_LEGAL_SOURCE=live python tools/refresh_legal_authorities.py --existing --write`
  re-verifies the bundled authorities and stamps each with its source URL +
  `retrievedAt`.
- Tests (`tests/test_legal_sources.py`) inject fake fetchers — **no real network in
  CI**.

> **Honest gaps that remain.** The default URL schemes are best-effort and
> overridable (`SOPHIA_HKLII_BASE`, `SOPHIA_ELEGISLATION_BASE`, `SOPHIA_TNA_BASE`,
> `SOPHIA_COURTLISTENER_BASE`) — confirm each source's robots/ToS and API shape
> (and supply `COURTLISTENER_API_TOKEN` for bulk US use) before production. And
> existence ≠ holding-supports-proposition: that still needs full-text + a model
> judge.

## Remaining build order

1. ~~Live `legal_citation_exists` backend~~ — **done** (above).
2. ~~A small legal benchmark run through the honest measurement path~~ —
   **done**. `tools/run_legal_citation_bench.py` scores `legal_citation_exists`
   over `benchmark/legal_citations.json` as an **objective** eval (ground-truth
   labels + deterministic verifier, no LLM judge) and publishes it under
   **verifierEvals** in [RESULTS.md](../../RESULTS.md): **100% accuracy, N=14**
   (every fabrication flagged, zero false alarms — including the actual *Mata*
   fake, *Varghese v. China Southern Airlines* 925 F.3d 1339). Honest bounds are
   published with it — tiny constructed N, capped by the register's completeness;
   it validates the extraction + fail-closed gate logic end-to-end, **not** a
   headline capability claim. A drift test (`tests/test_legal_citation_bench.py`)
   keeps the published number in sync with the runner.
3. ~~Federate other jurisdictions~~ — **done**. UK (National Archives Find Case
   Law) and US (CourtListener) `LegalSource` backends, routed by court token
   alongside HKLII / e-Legislation. US reporter citations (`925 F.3d 1339`,
   `576 U.S. 644`) are now extracted and resolved.
4. ~~Document ingestion~~ — **done**. `agent/legal_docs.py` extracts text from
   TXT/MD, DOCX (stdlib zip+XML), and HTML — plus PDF when an optional parser
   (`pypdf` / `pdfminer.six`) is installed — and runs the fail-closed citation
   scan over it. `tools/scan_legal_document.py` is a pre-filing gate: it exits
   non-zero if any cited authority is unverified (catches the *Mata* fake
   `925 F.3d 1339` in a real `.docx`/free text). Honest bound: extraction is
   citation-grade, not full layout/OCR — scanned-image PDFs need OCR first.
5. ~~Semantic faithfulness — "does the holding support the proposition?"~~ —
   **done** (the *Ayinde* misstated-authority failure). `agent/legal_faithfulness.py`
   pairs each proposition with its citations (masking citations so `U.S.`/`v.`
   periods don't fracture them), looks up an authoritative holding, and asks an LLM
   judge whether the holding supports the claim; `legal_holding_faithful` flags a
   `contradicted` verdict. `tools/check_legal_faithfulness.py` runs it over text or
   a document. **Honest bounds, by design:** the support judgment is a model call,
   so a single-judge number is *illustrative only* — a headline needs the
   no-overclaim gate (≥2 independent judges + CIs); it is **fail-closed and
   abstaining** (no holding text or no judge → unchecked, never a silent pass); and
   it does not do ratio-vs-obiter reasoning. Tests inject a deterministic stub judge
   to verify wiring without a model call.

## Wired into the agent (self-gating)

The verifiers are no longer standalone — the Sophia agent's epistemic gate
(`agent/gate.py:check_response`) runs them automatically on **every answer that
cites a legal authority**, so a legal answer self-checks before it reaches the
user:

- **Existence** (`legal_citation_exists`) runs always (deterministic, cheap); an
  unverifiable citation makes the gate **fail** — the *Mata* guardrail.
- **Faithfulness** (`legal_holding_faithful`) runs when a judge is configured
  (`SOPHIA_LEGAL_FAITHFULNESS=1`, judge via `SOPHIA_LEGAL_JUDGE`); a misstated
  authority makes the gate fail — the *Ayinde* guardrail.
- Non-legal answers (no citations) are a cheap no-op — `gate["legal"]` is `None`.
- The resolver respects `SOPHIA_LEGAL_SOURCE` (`off|cache|live`), so existence can
  be checked offline against the register or live against HKLII / e-Legislation /
  National Archives / CourtListener.

This is the step from "we have the machinery" to "the system uses it": `tools/sophia_agent.py`
now passes `legal_resolver` / `legal_judge` into the gate, and `tests/test_gate.py`
covers fabricated-citation failure, real-citation pass, the non-legal no-op, and
holding-faithfulness flagging (stub judge).

**Generalized beyond legal.** The same gate now also runs a domain-agnostic
**arithmetic-soundness** check (`gate["numeric"]`) on every answer — flagging a
finance/economy answer that claims `100000 / 5000 = 25 months` — and records the
detected `gate["sector"]` (e.g. `financial`). So the verifier-gating pattern is no
longer legal-only: any quantitative answer is self-checked, the natural extension
point for the finance/economy councils.

## Gated measurement of the semantic tier

The faithfulness tier is model-judged, so its accuracy is held to the same
**no-overclaim gate** as everything else (`tools/run_legal_faithfulness_bench.py`
over `benchmark/legal_holding_faithful.json`):

> validated = ≥2 independent judges (≥2 provider families, no mock) + mean pairwise
> Cohen's κ ≥ 0.40 + ≥3 runs + bootstrap 95% CI lower bound above chance (0.5).

The gate logic is tested offline with deterministic scripted judges
(`tests/test_legal_faithfulness_bench.py`) — it both **validates** a clean
multi-family run and **refuses** mock / single-family / low-agreement runs.
**No validated number is published** (RESULTS.md → *Semantic evals*: _None yet_):
producing one requires a local run with ≥2 real provider families. This is the
deliberate, honest end-state — the machinery to measure is in place; the headline
is withheld until it clears the gate.

```bash
python tools/run_legal_faithfulness_bench.py --judges mock --runs 1            # offline plumbing

# validated-grade — two separate provider keys:
python tools/run_legal_faithfulness_bench.py \
    --judges anthropic:claude-sonnet-4-6,deepseek:deepseek-chat --runs 3

# …or a SINGLE OpenRouter key (OPENROUTER_API_KEY) routing two vendors — the gate
# counts the underlying vendor as the family, so these are two independent judges:
python tools/run_legal_faithfulness_bench.py \
    --judges openrouter:anthropic/claude-sonnet-4-6,openrouter:deepseek/deepseek-chat --runs 3
```

## Growing the semantic benchmark (harder set)

The validated semantic number is small-N on clear-cut cases. The path to a real
stress test — subtle misstatement, ratio-vs-obiter, superseded authority, hard
faithful controls — is a curation effort, not new code: the runner already
supports self-contained cases (inline `holding` + provenance) and reports
**per-difficulty / per-failureType accuracy and κ**, so you can see exactly where
judges drop and disagree. A hard **seed** lives in
`benchmark/legal_holding_faithful_hard.json` (16 cases, AI-seeded and flagged
`seed-needs-verification` — holdings must be human-verified against their source
before any result is published). Full method, taxonomy, sourcing and pitfalls:
[Legal-Faithfulness-Benchmark.md](./Legal-Faithfulness-Benchmark.md).
