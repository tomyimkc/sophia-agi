# Long-Horizon Autonomy Runs

Sophia needs evidence that it can work beyond single-turn answers.

## Required Durations

| Tier | Minimum duration | Example task |
|---|---:|---|
| Short | 30 minutes | expand one benchmark domain and validate it |
| Medium | 2 hours | create hidden-pack tooling and run ablations |
| Long | 1 day | maintain a proof package across interruptions |

## Required Logs

- initial goal;
- plan;
- tool calls;
- state transitions;
- failed attempts;
- self-corrections;
- human intervention count;
- final artifact and verification command.

Runs with frequent human steering should be reported as partial autonomy, not
full autonomy.

## Harness

`tools/run_long_horizon.py` is the run logger that records every required field
above as append-only JSONL (so a run survives interruption and can be resumed),
and emits a public summary with the autonomy classification (full / mostly /
partial) and the duration tier.

```bash
# Short self-test demonstration of the harness (no credentials needed):
python3.12 tools/run_long_horizon.py --self-test

# A real timed run drives a longer spec; resume after interruption:
python3.12 tools/run_long_horizon.py --spec my-run.json
python3.12 tools/run_long_horizon.py --resume agi-proof/long-horizon-runs/<run>.log.jsonl \
  --intervene "operator restarted backend"
```

The included demo report is tier `below-short-demo` (shorter than the 30-minute
Short tier) — it proves the harness works, not that a 30-min/2h/1-day run has
been done. Unit tests: `tests/test_long_horizon.py`.
