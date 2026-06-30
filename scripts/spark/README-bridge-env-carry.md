# bridge env-carry (2026-06-30) — run cert/train THROUGH the bridge, results flow to the cloud

**Why:** direct-SSH cert/train jobs run *off* the bridge, so their results never reach a cloud
session — the owner ends up pasting/screenshotting verdicts. The bridge already carries `--bench-*`
results to the cloud (it read bench-a-03's full verdict from `bridge/results/`); the only gap was
that bench-B cert / bench-virtues need **config env** (`KEEP_SUFFIXES`, `QAT_ADAPTER`, `SEEDS`, …)
the command JSON couldn't carry. This adds an **allowlisted, value-sanitized `env`** to a command.

After this, a cert dispatched through the bridge writes its result (incl. the `VERDICT:` line in
`stdoutTail`) to `bridge/results/<id>.json` — which the cloud reads with
`python tools/spark_bridge.py result --id <id>`. No SSH, no pasting.

## Safety (defense in depth)
- Cloud side (`tools/spark_bridge.py`, already on the feature branch): `--env "K=V,K2=V2"` is
  accepted **only** for an allowlist of run_local_benchmarks/certify knobs, and each value must
  match `^[A-Za-z0-9_./:@,+= -]{0,200}$` (no shell metacharacters). Arbitrary keys (`PATH`,
  `LD_PRELOAD`) and unsafe values are refused at compose time.
- Poller side (`2026-06-30-bridge-env-carry.patch`): the poller **re-validates** the same way and
  silently drops anything not allowlisted/safe before exporting — it never trusts the command file
  to set arbitrary env. `--execute`/`--run-train` still require a human `approvedBy` (unchanged).

## Apply the poller patch (once, on the Spark's bridge checkout)
```bash
cd /home/tomyimkc/sophia-bridge
git checkout spark-bridge && git merge --ff-only origin/spark-bridge
git show origin/claude/sophia-positioning-gaps-84kb0v:scripts/spark/2026-06-30-bridge-env-carry.patch | git apply --check -
git show origin/claude/sophia-positioning-gaps-84kb0v:scripts/spark/2026-06-30-bridge-env-carry.patch | git apply -
python -c "import ast; ast.parse(open('tools/github_bridge_poll.py').read()); print('poller parses OK')"
git add tools/github_bridge_poll.py && git commit -m "bridge: carry allowlisted config env to the job (cert/train via bridge)"
git push origin spark-bridge
# restart the poller so it picks up the new code (use your existing launch line / tmux session)
```

## Then dispatch a cert through the bridge (the cloud composes; a human approves)
The cloud composes the command (env included), e.g. the T1 cert:
```bash
python tools/spark_bridge.py compose --id 2026-06-30-cert-t1 --args "--bench-b --execute" \
  --approved-by "user: T1 cert (2026-06-30)" \
  --env "KEEP_SUFFIXES=down_proj,QAT_ADAPTER=training/lora/checkpoints/olmoe-qat-spark-v3,CERT_NEVAL=256"
```
A human commits that JSON to `bridge/commands/<id>.json` on `spark-bridge` (the established
ritual). The poller runs it with the env exported; the result (with the `VERDICT:` line) lands in
`bridge/results/<id>.json`, which the cloud reads automatically. `canClaimAGI` stays false.
