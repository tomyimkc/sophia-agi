#!/usr/bin/env python3
from __future__ import annotations
import json, subprocess, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def test_conscience_proof_package() -> None:
    r=subprocess.run([sys.executable,'tools/build_conscience_proof_package.py'],cwd=ROOT,text=True,capture_output=True)
    assert r.returncode==0, r.stdout+r.stderr
    report=json.loads((ROOT/'agi-proof/conscience/conscience-eval.public-report.json').read_text())
    assert report['ok'] is True
    assert report['candidateOnly'] is True and report['level3Evidence'] is False and report['canClaimAGI'] is False
    assert len(report['implementedPriorities']) == 7

def main() -> int:
    test_conscience_proof_package(); print('test_conscience_proof_package: OK'); return 0
if __name__=='__main__': raise SystemExit(main())
