# Measurement-contract convenience targets. `make claim-check` runs the full deterministic
# enforcement suite locally — the same gates fast-ci runs on every PR. `make hooks` installs the
# pre-commit hook (opt-in via core.hooksPath).
.PHONY: claim-check claim-check-fast hooks bench-local aats-experiments

# Full contract: no-overclaim copy, habit-not-fact training rows, independent decontam, power
# self-test, and the GO/NO-GO claim receipts for the headline recipes.
claim-check:
	python tools/lint_claims.py
	python tools/lint_training_rows.py
	python tools/assert_decontam.py
	python tools/eval_stats.py
	python tools/claim_gate.py --prefix M3-pilot
	python tools/claim_gate.py --prefix M3-transfer
	python tools/build_tool_disclosure.py --check
	python tools/leiden_receipt.py --check

# Fast subset for the pre-commit hook (skips the ~3s shingle decontam scan).
claim-check-fast:
	python tools/lint_claims.py
	python tools/lint_training_rows.py

# Local DGX Spark + Mac Studio benchmark runbook (Benchmark A: two-box 2-family judging,
# Benchmark B: NVFP4 low-RAM cert). Dry-run by default — pass ARGS=--execute to actually run
# on the Spark. Owned hardware is free; RunPod is the only metered path (read wisdom-gpu-prebaked).
bench-local:
	SPARK_HOST=$(SPARK_HOST) MAC_HOST=$(MAC_HOST) bash scripts/run_local_benchmarks.sh $(ARGS)

hooks:
	git config core.hooksPath .githooks
	@echo "pre-commit hook installed (core.hooksPath=.githooks). Run 'git config --unset core.hooksPath' to remove."

# AATS auto-approval experiment harnesses (docs/research/ai-auto-approval-thesis.md §5).
# Deterministic + offline (no model): verify/generate census, diverse-ensemble agreement +
# error-correlation, conformal price-of-guarantee curve, canary circuit-breaker. All
# candidate-only; the two-model-family judge arms + real labeled rows are model-gated.
aats-experiments:
	python tools/verify_generate_census.py --check
	python tools/ensemble_agreement_study.py --demo
	python tools/aats_conformal_calibration.py --synthetic 600 --target-false-approve 0.05
	python tools/run_canary_breaker.py --approver sound --state agi-proof/aats/_test-tmp/breaker-ci.json
	python tests/test_aats_experiments.py
