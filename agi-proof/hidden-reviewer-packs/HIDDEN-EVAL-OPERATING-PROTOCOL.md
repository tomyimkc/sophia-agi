# Hidden Evaluation Operating Protocol

Hidden tests are evidence only if Sophia cannot train on, retrieve, or memorize
the prompts before the run.

## Public vs Private

Public:

- schema;
- scoring protocol;
- aggregate result format;
- salted hash commitments;
- reviewer identity or signature after review.

Private until evaluation:

- prompts;
- source materials;
- rubrics detailed enough to leak answers;
- salts used to verify commitments.

## Required Domains

- hidden philosophy tests;
- hidden psychology tests;
- hidden history tests;
- hidden logic tests;
- hidden coding tests;
- hidden planning tests;
- hidden tool-use tests;
- hidden learning tests.

## Workflow

1. Reviewer creates a private pack that follows `schema.json`.
2. Reviewer publishes salted commitments only.
3. Run backend preflight before exposing hidden prompts. If preflight fails,
   stop and fix the backend; the hidden pack remains unspent.
4. Sophia is evaluated in a clean environment with no access to the private pack
   except during the actual timed run.
5. Responses, tool logs, memory writes, and artifacts are saved.
6. For Sophia-system runs, execute `tools/run_hidden_eval_sophia.py` so retrieval,
   gate checks, one repair attempt, tool logs, and learning memory diffs are
   captured.
7. Reviewer scores with `tools/hidden_eval_protocol.py` and completes the manual
   semantic review template before making strong claims.
8. Reviewer reveals the pack only after the run, or keeps it sealed for reuse.

## Commands

```bash
python tools/hidden_eval_protocol.py validate private/hidden-evals/<pack>/pack.json
python tools/hidden_eval_protocol.py template private/hidden-evals/<pack>/pack.json --out private/hidden-evals/<pack>/responses.template.json
python tools/hidden_eval_protocol.py score private/hidden-evals/<pack>/pack.json private/hidden-evals/<pack>/responses.json --out private/hidden-evals/<pack>/report.json
python tools/hidden_eval_protocol.py score private/hidden-evals/<pack>/pack.json private/hidden-evals/<pack>/responses.json \
  --manual-review private/hidden-evals/<pack>/manual-review-completed.json \
  --out private/hidden-evals/<pack>/reviewed-report.json
python tools/run_hidden_eval_sophia.py private/hidden-evals/<pack>/pack.json \
  --responses-out private/hidden-evals/<pack>/sophia-responses.json \
  --private-report-out private/hidden-evals/<pack>/sophia-private-report.json \
  --public-report-out agi-proof/benchmark-results/hidden-<pack>-sophia.public-report.json \
  --manual-review-out private/hidden-evals/<pack>/manual-review-template.json \
  --repair
```

The runner performs backend preflight by default. Use `--skip-preflight` only
for smoke tests on already-spent packs or for deliberately recording a backend
failure; it must not be used for a fresh hidden pack.

For Grok CLI runs, the runner sends prompts through temporary files in
`private/hidden-evals/.grok-isolated-cwd/` and deletes them after each call. This
isolated cwd avoids loading project MCP/plugin configuration during benchmark
answer generation.

For DeepSeek API runs, set `DEEPSEEK_API_KEY` in the environment or pass
`--deepseek-api-key-stdin` and provide the key through stdin. The default base
URL is `https://api.deepseek.com`; override with `DEEPSEEK_BASE_URL` or
`DEEPSEEK_MODEL` if needed.

## Required Evidence For Special Cases

- Tool-use tasks must include actual command/tool logs with return codes.
- Learning tasks must include pre-test, append-only memory write, post-test, and
  hashes proving protected old records were unchanged.
- Semantic/quality checks remain pending until a human reviewer fills the manual
  judgement file; automatic keyword/regex/alias scoring is only a first pass.
- Empty model answers score zero, even if avoid/tool/memory bookkeeping checks
  would otherwise produce partial credit.
