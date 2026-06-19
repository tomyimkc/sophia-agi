# External Benchmark Plan

External benchmarks are required before any stronger AGI claim. Current status:
not run.

| Benchmark family | Capability tested | Required artifact |
|---|---|---|
| ARC-AGI / ARC-AGI-3 | Novel reasoning and skill-acquisition efficiency | official or reproducible score, solver config, per-task logs |
| GAIA-style tasks | Tool-using assistant reasoning | answer traces, tool logs, exact prompts, scoring script |
| SWE-bench-style repo tasks | Software maintenance agency | patches, test logs, resolved-task rate |
| METR-style autonomy | Long-horizon autonomous work | task suite, intervention count, full action logs |

## Result Template

```json
{
  "benchmark": "",
  "date": "",
  "system": "sophia-full",
  "model": "",
  "score": null,
  "total": null,
  "cost_usd": null,
  "time_minutes": null,
  "logs": [],
  "failures": []
}
```
