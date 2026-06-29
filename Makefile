# Measurement-contract convenience targets. `make claim-check` runs the full deterministic
# enforcement suite locally — the same gates fast-ci runs on every PR. `make hooks` installs the
# pre-commit hook (opt-in via core.hooksPath).
.PHONY: claim-check claim-check-fast hooks

# Full contract: no-overclaim copy, habit-not-fact training rows, independent decontam, power
# self-test, and the GO/NO-GO claim receipts for the headline recipes.
claim-check:
	python tools/lint_claims.py
	python tools/lint_training_rows.py
	python tools/assert_decontam.py
	python tools/eval_stats.py
	python tools/claim_gate.py --prefix M3-pilot
	python tools/claim_gate.py --prefix M3-transfer

# Fast subset for the pre-commit hook (skips the ~3s shingle decontam scan).
claim-check-fast:
	python tools/lint_claims.py
	python tools/lint_training_rows.py

hooks:
	git config core.hooksPath .githooks
	@echo "pre-commit hook installed (core.hooksPath=.githooks). Run 'git config --unset core.hooksPath' to remove."
