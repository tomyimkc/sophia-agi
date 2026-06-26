# RLVR adapter — multi-seed replication (candidate evidence)

_Standalone evidence — NOT the gated public RESULTS.md (which is generated from published-results.json)._

- Adapter: `sophia-rlvr-v1`; seeds: [0, 1, 2]; n=3
- Held-out capability (meanReward): 0.5896 → 0.7295 (mean Δ 0.1399, range 0.1058…0.1575, σ 0.0295)
- SSIL gate promotes: 2/3; protected regression: True; contaminated: False
- capabilityClaimReady: False
- Runs: https://github.com/tomyimkc/sophia-agi/actions/runs/28122120705,https://github.com/tomyimkc/sophia-agi/actions/runs/28128049054,https://github.com/tomyimkc/sophia-agi/actions/runs/28122135129

**Boundary:** aggregated gate result under the no-overclaim measurement gate; n is small; `candidateOnly: true`, `canClaimAGI: false`. Not a validated capability claim.
