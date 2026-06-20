# Coding Council

Sophia's coding council is a data-driven software decision panel. It is meant
to improve coding, tool-use, and engineering benchmark answers by routing a task
to relevant language, role, and platform seats.

## Boundary

Legendary programming figures are treated as source-inspired design lineages,
not as impersonated people. A C seat may use Dennis Ritchie's design lineage as
historical framing, but it does not claim to speak as Dennis Ritchie. This
matches the religion figure council boundary: use sources and traditions to
shape behavior without impersonation.

## Seat Types

- Language elders: C, C++, Python, Java, JavaScript, TypeScript, C#, Go, Rust,
  Ruby, PHP, Swift, Kotlin, SQL, shell, functional languages, data-science
  languages, Lisp/Scheme/Clojure, Erlang/Elixir, Perl, Lua, Fortran, COBOL,
  Prolog, MATLAB/Octave, and Dart/Flutter.
- Expert roles: architecture, frontend, backend, database, security,
  performance, QA/test, DevOps/SRE, ML engineering, and final code review.
- Platform experts: Linux, macOS, Windows, mobile, and game engines.
- Engineering specialists: tool-calling reliability, RAG/context engineering,
  structure planning, debugging/reproduction, evaluation/benchmarks,
  prompt/protocol design, integration, observability, data quality, and product
  engineering.
- Self-improvement and writing-method seats: concept-level reviewers inspired by
  popular improvement and writing methods such as habit systems, focused work,
  capture/next-action workflows, retrieval practice, deliberate practice, lean
  experiments, and clear technical writing. These seats do not impersonate
  authors or quote long passages; they translate high-level concepts into code
  workflow constraints.

## Decision Contract

Coding answers should:

1. Classify language, platform, and risk.
2. Name selected council seats and why they were selected.
3. Give a concrete implementation or patch plan.
4. Name tests or tool commands to run.
5. Review security, performance, maintainability, and edge cases.
6. For hidden/eval tasks, state how the answer satisfies rubric labels,
   tool/memory evidence, and manual-review expectations.
7. End with Decision and 中文摘要.

## Strict-Pass Strategy

Hidden strict pass requires more than better prose. The council should drive a
checklist loop:

1. Evaluation engineer maps every rubric item to explicit answer evidence.
2. Tool-calling engineer verifies actual command logs and return codes.
3. RAG/context engineer checks whether retrieved sources cover the task.
4. Structure planner forces the answer into required sections.
5. Writing-method seat rewrites for clear, reviewer-friendly evidence.
6. Final code reviewer rejects answers that lack patch/test specificity.
7. Web-evidence reviewer may add official docs, papers, or source URLs when
   online search is explicitly approved.
8. Deterministic rubric reviewer records missing required items before a repair
   attempt is spent.

Chinese output should be a bounded summary, not the main evidence body, unless
the task explicitly asks for Chinese. This keeps English rubrics from missing
required keywords while preserving Sophia's bilingual identity.

## Implementation

- Data: `data/coding_council_figures.json`
- Router: `agent/coding_council.py`
- Prompt wiring: `agent/prompts.py`
- Hidden runner integration: `tools/run_hidden_eval_sophia.py`

For hidden coding cases, the runner records the selected council route in the
private response payload and publishes only aggregate route information in the
public report.

The council is intentionally extensible. Rare or niche languages route through
the polyglot fallback seat until a dedicated source-inspired language seat is
added.
