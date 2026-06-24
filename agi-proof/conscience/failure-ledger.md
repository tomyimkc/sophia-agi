# Conscience Failure Ledger

This ledger records known limits and non-clearing claims for the Conscience Kernel.

| Area | Current limit | Mitigation / next step |
|---|---|---|
| Semantic entropy | CI path uses lexical clustering/proxy, not full NLI equivalence. | Add NLI backend and hidden-state probe calibration per model. |
| Conformal abstention | Calibration pack is small and self-authored. | Expand to larger held-out and third-party packs; report Wilson intervals. |
| Constitutional classifier | Rule-based deterministic MVP. | Train/evaluate a local classifier on constitution-derived benign/attack/near-miss cases. |
| Deception probes | Activation probe path uses transparent text features, not real residual activations. | Add PyTorch-MPS/MLX hidden-state extraction and probe calibration. |
| Runtime enforcement | Core high-impact adapters are covered, but third-party integrations must opt into MCP/hook enforcement. | Add integration tests for every new tool/write/promotion surface. |
| Moral reasoning | Moral parliament is bounded heuristic aggregation, not a moral oracle. | Use only for gray zones after hard prohibitions clear. |
| AGI status | `canClaimAGI=false`; no hidden third-party AGI evidence. | Keep reflexive self-gate mandatory for public claims. |

