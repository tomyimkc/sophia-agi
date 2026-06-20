# Hidden Prepared Pack Diagnosis — 2026-06-19

Pack: `prepared-pack-2026-06-19`  
Runner: `tools/run_hidden_eval_grok.py`  
Model/backend: `grok-cli`  
Public score: `28.75/40` automatic partial score, `2/8` strict pass.

## Claim Boundary

This run is a preliminary hidden benchmark, not AGI proof.

The run did **not** exercise the full Sophia architecture. It used Grok CLI as a
direct respondent with Sophia-style instructions. It did not use Sophia's normal
RAG retrieval, council orchestration, memory, gate repair loop, executor, or
append-only learning mechanism.

## Main Causes

| Cause | Evidence | Impact | Fix |
|---|---|---|---|
| Architecture bypass | Hidden runner directly calls Grok CLI | Score reflects prompt-only behavior, not full Sophia | Add a hidden-eval runner that calls Sophia's full agent pipeline |
| Exact-string scoring | Several strong answers missed one literal marker | Strict pass rate understates capability | Add aliases/regex and manual/LLM judge scoring |
| No repair loop | Answers were scored once, with no gate-triggered rewrite | Small phrasing misses became final failures | Add hidden-eval self-check and one bounded repair attempt |
| Tool-use not executed | Tool-use case was answered verbally while tools were disabled | Does not prove executor competence | Add tool-use tasks that require actual commands/logs |
| Learning not executed | Learning case was answered verbally; no append-only memory write was performed | Does not prove learning-under-shift | Add pre-test → append-only memory → post-test harness |
| One case per domain | Only 8 cases total | High variance, weak proof value | Expand to 100+ hidden cases with reviewer packs |
| Backend friction | Early Grok runs failed from sandbox/MCP/max-turn setup | Wasted attempts and noisy logs | Use raw model API or disable Grok MCP/session startup for evals |

## Case-Level Pattern

- Philosophy: substantively good lineage answer, but missed a required explicit
  source-discipline marker.
- Psychology: rejected overclaiming, but repeated a forbidden phrase while
  refuting it and missed a universal-claim marker.
- History: passed strict checks.
- Logic: passed strict checks.
- Coding: gave the right minimal patch, but missed one exact marker.
- Planning: good plan, but included a forbidden phrase while describing the
  constraint.
- Tool use: proposed a tool sequence, but did not name the expected repo tools
  and did not execute tools.
- Learning: answered the new fact and append-only behavior, but scoring missed a
  synonym/wording variant.

## Highest-Leverage Improvements

1. Build `tools/run_hidden_eval_sophia.py` that invokes the full Sophia path:
   retrieval context, council mode, gate, memory, executor logs, and artifacts.
2. Add a hidden-eval answer contract:
   `Required markers`, `Forbidden claims`, `Tool/action log`, `Decision`,
   `中文摘要`.
3. Upgrade scoring:
   aliases, regex, semantic judge, and human-review fields.
4. Add a bounded repair loop:
   if hidden scorer/gate detects missing markers, Sophia gets one rewrite using
   only the failure labels, not the answer key.
5. Make tool-use and learning tasks operational:
   actual command logs and append-only memory diffs must be required evidence.
6. Create a fresh third-party hidden pack after these fixes; the current pack is
   now spent.

## Next Success Threshold

For the next hidden run, target:

- at least `80/100` hidden cases;
- `>= 85%` strict pass;
- `>= 90%` partial/semantic score;
- zero old-knowledge overwrite failures;
- full tool-call and artifact logs for tool-use/planning/coding tasks.
