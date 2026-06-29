# Skills as a Self-Improving Substrate — Thesis, Code Review, and Triggering Roadmap

> Scope: how to make this repo's agents **write/update skills when they hit an issue**, and make
> skills **easier to trigger**. Grounded in the three skill systems already in the tree. Written to
> the repo's no-overclaim discipline: every claim about external work is a named reference; every
> claim about this codebase is a `file:line`.

## 0. TL;DR

You already have two of the three pieces of a self-improving skill library. What's missing is the
**wiring between experience (failures) and the skill library**, and a **unified, robust trigger
path**. Concretely:

1. A **verifier-gated skill forge** exists (`tools/sophia_skill_forge.py`, `gateway/skill_flywheel.py`)
   — skills only ship if a synthesized verifier clears held-out validation. This is the right thesis
   and the hard part is done.
2. A **description/keyword-triggered skill layer** exists in two flavors: model-chosen Agent Skills
   (`.claude/skills/*/SKILL.md`) and a token-overlap router (`agent/skills.py::select`).
3. **Nothing connects a logged issue to the forge**, forged skills are **never routable** by the
   agent, and the router (`select`) is brittle. Those three gaps are exactly your two asks.

---

## 1. The thesis: skills are the unit of recursive self-improvement, and verifiers are the gate

The strongest current framing in the agent literature is that **durable capability accrues in a
growing library of reusable skills, not in a single monolithic policy**:

- **Voyager** (Wang et al., 2023) is the canonical precedent for "the AI writes a skill when it
  encounters something new": a lifelong-learning agent that, on each novel situation, *authors a new
  skill as code*, stores it in a skill library keyed by an embedding of its description, and
  *retrieves by similarity* later. New skills are admitted only after they verifiably succeed in the
  environment. Skill = code + description + embedding; growth is driven by environment feedback.
- **Reflexion** (Shinn et al., 2023) is the precedent for "turn a failure into a written lesson":
  the agent converts a failed trajectory into *verbal* feedback it keeps and conditions on next time.
- **Anthropic Agent Skills** (the `SKILL.md` + frontmatter format this repo's `.claude/skills/` uses)
  add **progressive disclosure**: the model sees only each skill's short `description` until it
  chooses to load the body. So *trigger quality is entirely a function of the description*, because
  that is all the model sees at selection time.

This repo's distinctive contribution — and it is a genuinely good one — is to put a **verifier in
front of skill admission**. Voyager trusts environment success; Reflexion trusts the LLM's own
reflection. Here, `gateway/skill_flywheel.py::synthesize_gate` (lines 56-85) admits a skill only if a
*synthesized predicate* clears disjoint fit/validation/test splits with precision **and** recall
floors, with any LLM-proposed predicate AST-sandboxed first. That is the same anti-overclaim stance
as `tools/lint_claims.py`, applied to skills: **a skill earns its place by passing a held-out gate,
not by an LLM's say-so.**

So the thesis to build toward is:

> **An agent should treat every issue it hits as a labelled example, draft or revise a skill from it,
> and admit that skill only through the existing verifier gate. The skill library, the trigger
> router, and the failure ledger should be one closed loop — experience in, verifier-gated skills out,
> better triggering next time.**

Everything below is how to close that loop on top of what you already have.

---

## 2. The three skill systems you actually have (map before you build)

| Layer | Files | Unit | How it triggers today | Self-improving? |
|---|---|---|---|---|
| **A. Agent Skills (Claude Code)** | `.claude/skills/*/SKILL.md`, `settings.json` hooks | Markdown + YAML `description` | **Model-chosen** from the `description` each turn; **hooks** fire deterministically (`SessionStart`, `PreToolUse`) | No — authored by hand |
| **B. Harness skills (router)** | `skills/registry/*.json`, `agent/skills.py` | JSON spec (whenToUse/triggers/workflow/verification) | **Token-overlap scorer** `select()` (`agent/skills.py:86-114`) | No — curated by hand |
| **C. Skill Forge / Flywheel** | `tools/sophia_skill_forge.py`, `gateway/skill_flywheel.py`, `skills/core.py`, `skills/generated/`, `skills/registry/forge_index.json` | Verifier + program + eval suite | **Registered into a `Gateway`** by id; reliability-tracked | **Yes** — verifier-gated synthesis from labelled examples |

The "issue" source already exists too: `agi-proof/failure-ledger.md` ("Failures are claim evidence")
and `training/feedback/`. It is **manual markdown**, read by humans, not by the forge.

### The honest gap analysis

1. **No experience → skill bridge.** `gateway/skill_flywheel.py::improve_skill` (lines 92-117) is
   *exactly* "update a skill from failure examples" — and **nothing calls it from a failure stream**.
   `forge_skill` is invoked by hand with a spec JSON. The loop the user wants is half-built and unused.
2. **Forged skills are orphaned from triggering.** `forge_skill` writes `forge_index.json` and
   optionally registers into a **throwaway** `Gateway()` (`tools/sophia_skill_forge.py:328`), while the
   router deliberately **skips** `forge_index.json` (`agent/skills.py:60`, `_is_metadata_file`). A
   promoted forged skill is therefore never selectable by the agent harness, and its Gateway
   registration evaporates when the process exits. **The forge produces skills nothing routes to.**
3. **The router is brittle.** `select()` (`agent/skills.py:86-114`) is exact-token overlap with
   `min_score=1`: "debugging" does not match trigger "debug" (no stemming/synonyms), one shared common
   word ("test", "code") can win spuriously, only a single skill is returned (no composition), and ties
   go to iteration order. Paraphrased goals miss.
4. **Three registries that don't talk.** `.claude/skills/`, `skills/registry/*.json`, and
   `forge_index.json` are siblings with no projection between them. A skill authored in one layer is
   invisible to the other two.

---

## 3. Capability (a): "write/update a skill whenever it encounters an issue"

Build it as a **Reflexion→Forge** loop that reuses the verifier gate so nothing unproven ships.

### 3a. At the Claude Code layer (shipped in this PR — lowest risk, immediate value)

The new plaintext process skill **`.claude/skills/skill-author/SKILL.md`** operationalizes this for the
agent you are using right now. Its trigger is "you hit a non-trivial issue (a bug, a footgun, a
surprising repo rule, a wasted cycle) that a future session would repeat." Its workflow:

1. Decide layer: a *recurring process/discipline* lesson → new/updated **Agent Skill** (Layer A); a
   *checkable classification* (e.g. "is this output an attribution trap?") → **Skill Forge** spec
   (Layer C) so it goes through the verifier gate.
2. Search existing skills first — **prefer updating** an existing skill's description/body over adding a
   near-duplicate (de-dup is what keeps triggering sharp).
3. Write the description with the trigger discipline in §4. Log the issue + the skill id in
   `agi-proof/failure-ledger.md` so the experience is recorded as claim evidence.

This is deterministic, carries no IP, and is exempt from encryption (see the `.gitattributes` change)
so a fresh web session can read it without the git-crypt key — same rationale as the other three
process skills.

### 3b. At the Forge layer (proposed, ready to build on approval)

A small `tools/skill_from_failure.py` that:

1. Reads structured failure rows (extend the ledger with an optional sidecar
   `agi-proof/failures.jsonl`: `{id, text, expected_label, observed}`), or accepts them on stdin.
2. Groups by task/domain, builds `(text, label)` pairs, and calls the **existing**
   `forge_skill(spec, ...)` / `improve_skill(gateway, skill_id, examples, ...)`. No new gate logic —
   it reuses `synthesize_gate`, so a skill is admitted only if the verifier clears held-out validation.
3. On promotion, appends a row to `forge_index.json` (already supported) **and** projects a Layer-B
   `skills/registry/<task>.json` (see §3c) so the new skill is immediately routable.
4. On rejection (the common, honest case), writes the `promotion_report.json` and logs "no skill
   shipped — verifier failed validation" to the ledger. **A rejection is a valid outcome**, matching
   the repo's gate-decides-not-judgment rule (`AGENTS.md` Guardrails).

This is the closed loop: issue → labelled examples → verifier-gated forge → routable skill → recorded
in the ledger. It is ~80 lines because every hard part already exists.

### 3c. Projection: make a forged skill routable

Add `forge_skill` post-step (or a separate `tools/project_forge_to_registry.py`) that, for each
**promoted** entry, emits a Layer-B spec:

```jsonc
{
  "name": "skill.<task>",
  "whenToUse": "<spec.description>",
  "triggers": [/* task tokens + salient verifier rule tokens, e.g. contains:<token> */],
  "requiredTools": ["skillforge"],
  "workflow": ["Run the forged verifier skill.<task> via Gateway.", "..."],
  "ioSchema": {"input": {"text": "string"}, "output": {"answer": "boolean"}},
  "verification": ["forge_index best_validation accuracy >= threshold"],
  "commonFailures": ["paraphrases outside the validated distribution"],
  "examples": [/* a couple of eval_suite rows */]
}
```

Now a forged skill is discoverable by `agent/skills.py::select` and (if you also unify, §5) by the
Claude Code layer. This single projection closes gap #2.

---

## 4. Capability (b): make skills "more easily triggered"

Triggering is two mechanisms; improve both.

### 4a. Description discipline (Layer A — biggest lever, zero new code)

Because of progressive disclosure, **the `description` is the entire trigger surface**. Your own
`.claude/README.md:26-48` already states the rule; only the three process skills fully follow it. The
gold standard is `git-discipline` (`.claude/skills/git-discipline/SKILL.md:1-12`): it front-loads
exact verbs and situations and *names the "even if it looks unnecessary" case* ("Use even if the
conversation already contains a diagnosis"). Apply the same to every skill:

- First two sentences: **exact user phrasings + concrete actions** that should fire it.
- Name the **negative-override** cases ("even if X looks fine", "before Y, not after").
- Include the **slash-command** alias and any MCP tool names (raises recall on tool-shaped goals).

Make this enforceable: a `tools/lint_skill_descriptions.py` (sibling of `lint_claims.py`) that scores
each `description` for trigger-richness — presence of action verbs, a "use when" clause, and a
distinctive token not shared with sibling skills — and fails CI on thin descriptions. Triggering then
becomes a gated artifact, not a vibe.

### 4b. Replace the keyword router with hybrid retrieval (Layer B)

Upgrade `agent/skills.py::select` (currently `:86-114`):

- **Normalize/stem** tokens before overlap (so "debugging"≈"debug"); add a small synonym map for your
  domain ("provenance"≈"attribution"≈"citation").
- **Embedding fallback**: embed each skill's `description`+`triggers` once (reuse the existing RAG
  index under `rag/`), and when keyword score is low, rank by cosine similarity — this is the
  Voyager retrieval pattern and is robust to paraphrase.
- **Return ranked top-k**, not a single winner, so the planner can compose skills and break ties
  deterministically (current `score > best_score` silently favors iteration order).
- Raise the spurious-match floor (`min_score`) or down-weight tokens that appear in many skills
  (IDF over the skill corpus).

### 4c. Deterministic triggers via hooks (Layer A — guarantees firing)

Extend the `settings.json` hook pattern (today `SessionStart` + `PreToolUse` git guard,
`.claude/hooks/git_write_guard.sh`) with a **failure-aware nudge**: a `PostToolUse`/`Stop` hook that,
when a command or test exits non-zero, injects "an issue just occurred — consider the `skill-author`
skill to capture it." That makes capability (a) *fire on its own*, the same way the git guard does.
Keep it advisory (exit 0) to match the existing never-block contract.

### 4d. Close the loop: learn the triggers (the self-improving part)

Log every selection: `{goal, skill_id, fired, verifier_verdict}`. Tokens that co-occur with
*accepted* outcomes become promoted triggers; tokens that co-occur with misfires get demoted. Now
**triggering itself is verifier-gated and self-improving** — the same flywheel, applied to routing.

---

## 5. Unify the three registries (the structural fix behind both asks)

Define one manifest schema that all layers read, e.g. `skills/registry/index.json` with, per skill:
`id, layer (A|B|C), description, triggers, body_ref, verifier_ref, eval_ref, reliability`. Generate it
from the three sources. Then:

- The Claude Code agent, the harness router, and the forge all select from **one** surface.
- A forged skill (C) automatically gets a description (A) and trigger tokens (B).
- A hand-written Agent Skill (A) can carry a `verifier_ref` and be held to the same gate.

This is the single change that makes "write a skill once, trigger it everywhere" true.

---

## 6. Code review — specific findings

| # | Location | Finding | Suggested fix |
|---|---|---|---|
| 1 | `agent/skills.py:86-114` | `select()` is exact-token overlap, `min_score=1`, single-winner, no stemming/synonyms, ties by iteration order. Paraphrases miss; common words win spuriously. | Hybrid retrieval §4b: stem + synonym + embedding fallback + ranked top-k + IDF weighting. |
| 2 | `tools/sophia_skill_forge.py:60` (`_is_metadata_file`) vs `:328` (`Gateway()`), `agent/skills.py:60` | Forged skills are written to `forge_index.json` but the router **skips** it, and `--register-smoke` uses a **throwaway** Gateway, so the registration is ephemeral and the skill is never routable. | Projection step §3c; persist Gateway competence or document smoke registration as ephemeral. |
| 3 | `gateway/skill_flywheel.py:92-117` (`improve_skill`) | The "update a skill from failure examples" primitive exists but **has no caller** from any failure stream. | Wire `tools/skill_from_failure.py` §3b into it. |
| 4 | `tools/sophia_skill_forge.py:123-170` (generated `verifier.py`) | Generated skills import private internals `_compile_predicate`, `_is_prime` from `agent.verifier_synthesis`. Renaming those breaks every generated skill silently. | Expose a stable public predicate API, or vendor the predicate lib into the generated dir. |
| 5 | `tools/sophia_skill_forge.py:162-164` + generated `_one()` | `_compile_predicate(params["src"])` compiles LLM-proposed predicate source **at skill runtime**. Synthesis-time AST sandboxing is good; confirm the sandbox is re-applied on every load, not only at proposal time. | Verify `_compile_predicate` sandboxes on load; add a test that a malicious `src` is refused at runtime. (Flagged to verify, not a confirmed bug.) |
| 6 | `.claude/README.md:26-48` vs the skill set | The description-trigger discipline is documented but only the 3 process skills follow it well; nothing enforces it. | `tools/lint_skill_descriptions.py` §4a, wired into CI like `lint_claims`. |
| 7 | `skills/core.py:42-56` | `@sophia_skill` wrapper takes only `**kwargs`; a skill called positionally fails closed silently as "held" with a `TypeError`. Fine for safety, but indistinguishable from a real abstain in logs. | Tag wrapper-level signature errors distinctly from intentional abstains so §4d telemetry isn't polluted. |

### What is genuinely good (keep it)

- **Verifier-gated admission** (`synthesize_gate`, precision+recall floors, disjoint splits,
  AST-sandboxed proposals) is the right and rare thing. It is the anti-overclaim culture applied to
  skills. Do not weaken it to make the issue→skill loop "easier" — route rejections to the ledger.
- **Fail-closed skills** (`skills/core.py`) — abstain over fabricate — is correct for this repo.
- **Plaintext process / encrypted IP split** in `.gitattributes` is a clean trust boundary; the new
  `skill-author` skill follows it.

---

## 7. Suggested build order

1. **(shipped here)** `skill-author` Agent Skill + `.gitattributes` exception — capability (a) at the
   Claude Code layer, zero risk.
2. `tools/lint_skill_descriptions.py` + tighten every `description` — capability (b), biggest lever,
   no architecture change.
3. Hybrid `select()` (§4b) + projection of forged skills to Layer B (§3c) — makes forged skills
   routable and paraphrase-robust.
4. `tools/skill_from_failure.py` (§3b) + structured `failures.jsonl` — the issue→forge bridge,
   reusing the existing gate.
5. Unified registry (§5) + trigger telemetry/learning (§4d) — the closed self-improving loop.

Each step is independently useful and independently gateable. Nothing above asks you to trust an LLM
over a verifier — that is the whole point.
</content>
</invoke>
