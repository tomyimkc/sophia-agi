#!/usr/bin/env python3
from __future__ import annotations
import subprocess, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def run(cmd):
    r=subprocess.run([sys.executable, *cmd], cwd=ROOT, text=True, capture_output=True)
    assert r.returncode==0, r.stdout+r.stderr

def test_eval_tools() -> None:
    run(['tools/run_conformal_conscience_eval.py','--out','/tmp/conformal.json'])
    run(['tools/run_constitutional_eval.py','--out','/tmp/constitutional.json'])
    run(['tools/run_deception_eval.py','--out','/tmp/deception.json'])
    run(['tools/run_semantic_entropy_eval.py','--out','/tmp/semantic.json'])
    run(['tools/run_probe_eval.py','--out','/tmp/probe.json'])

def main() -> int:
    test_eval_tools(); print('test_conscience_eval_tools: OK'); return 0
if __name__=='__main__': raise SystemExit(main())
