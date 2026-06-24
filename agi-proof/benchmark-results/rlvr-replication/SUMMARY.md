# RLVR adapter — multi-seed replication (candidate evidence)

_Standalone evidence — NOT the gated public RESULTS.md (which is generated from published-results.json)._

- Adapter: `sophia-rlvr-v1`; seeds: [0, 1, 2]; n=3
- Held-out capability (meanReward): 0.5935 → 0.7609 (mean Δ 0.1674, range 0.1126…0.2058, σ 0.0487)
- SSIL gate promotes: 2/3; protected regression: True; contaminated: False
- capabilityClaimReady: False
- Runs: https://github.com/tomyimkc/sophia-agi/actions/runs/28101283220,https://github.com/tomyimkc/sophia-agi/actions/runs/28108334353,https://github.com/tomyimkc/sophia-agi/actions/runs/28113651413

**Boundary:** aggregated gate result under the no-overclaim measurement gate; n is small; `candidateOnly: true`, `canClaimAGI: false`. Not a validated capability claim.
