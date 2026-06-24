# Source-Discipline Engineering (course outline)

Learn the **method** behind Sophia: how to build AI that traces claims to sources, abstains
instead of fabricating, and reports honestly-bounded numbers. This teaches a *practice*, not
a product — and definitely not "how to build AGI."

> Built from the open corpus, the OKF wiki, and the verifier gate in this repo. Everything
> taught here is reproducible against the Apache-2.0 core.

## Who it's for
Engineers and researchers shipping LLM features who need outputs they can defend — legal,
medical, finance, education, or any high-stakes domain.

## What you'll be able to do
- Design a **fail-closed verifier gate** (`record_claim → verify_claim`) for your pipeline.
- Build a **provenance corpus** without contaminating it.
- Run a **no-overclaim benchmark** (≥2 judge families, κ, ≥3 runs, CIs) and read it honestly.
- Distinguish *abstention* from *fabrication* and measure the trade-off (calibration/ECE).
- Keep a **failure ledger** so your claims never exceed your evidence.

## Modules
1. The fabrication problem & why keyword scoring lies.
2. Provenance & attribution: who wrote what, and how to verify it.
3. The verifier gate: accept / hold / reject, fail-closed.
4. Abstention vs fabrication: calibration and the no-overclaim gate.
5. Benchmarking honestly: judges, agreement, CIs, and the failure ledger.
6. Shipping it: MCP, governance contract, and human-in-the-loop boundaries.

## Formats (indicative)
| Product | Price |
|---|---|
| eBook / written guide | $29 |
| Self-paced course | $197 |
| Live cohort (incl. office hours) | $997 |

Prices are indicative and may change.

## Honesty note
This course teaches source-discipline engineering. It does **not** teach or claim AGI, and
completing it does not certify any capability beyond the method itself.
