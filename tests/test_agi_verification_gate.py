#!/usr/bin/env python3
from pathlib import Path
import json
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def test_agi_verification_gate_smoke() -> None:
    out = Path('/tmp/sophia-agi-verification-test.json')
    proc = subprocess.run(
        [sys.executable, 'tools/run_agi_verification_gate.py', '--allow-open', '--out', str(out)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    data = json.loads(out.read_text())
    assert data['schema'] == 'sophia.agi_verification.report.v1'
    assert data['canClaimAGI'] is False
    ids = {c['id'] for c in data['checks']}
    for required in {
        'provenance_validation',
        'hidden_full_comparison',
        'rlvr_adapter_eval',
        'external_benchmarks',
        'verifier_synthesis_integrity',
        'cross_domain_transfer',
    }:
        assert required in ids
    assert isinstance(data['remainingNonHumanSteps'], list)


if __name__ == '__main__':
    test_agi_verification_gate_smoke()
    print('test_agi_verification_gate: OK')
