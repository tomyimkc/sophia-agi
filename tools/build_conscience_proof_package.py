#!/usr/bin/env python3
from __future__ import annotations
import json, subprocess, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from agent.public_sanitize import sanitize_public_artifact
OUT=ROOT/'agi-proof'/'conscience'/'conscience-eval.public-report.json'
COMMANDS=[
    [sys.executable,'tools/run_conscience_demo.py','--out',str(ROOT/'agi-proof/conscience/conscience.public-report.json')],
    [sys.executable,'tools/run_conformal_conscience_eval.py'],
    [sys.executable,'tools/run_constitutional_eval.py'],
    [sys.executable,'tools/run_deception_eval.py'],
    [sys.executable,'tools/run_semantic_entropy_eval.py'],
    [sys.executable,'tools/run_probe_eval.py'],
    [sys.executable,'tools/run_moral_public_standard_eval.py'],
]
FILES={
    'conscience':'conscience.public-report.json',
    'conformal':'conformal-conscience.public-report.json',
    'constitutional':'constitutional-eval.public-report.json',
    'deception':'deception-eval.public-report.json',
    'semanticEntropy':'semantic-entropy.public-report.json',
    'semanticEntropyProbe':'semantic-entropy-probe.public-report.json',
    'activationProbe':'activation-probe.public-report.json',
    'moralPublicStandard':'moral-public-standard-eval.public-report.json',
}
def run():
    for cmd in COMMANDS:
        r=subprocess.run(cmd,cwd=ROOT,text=True,capture_output=True)
        if r.returncode!=0:
            raise SystemExit(r.stdout+r.stderr)
    components={}
    for k,f in FILES.items():
        p=ROOT/'agi-proof'/'conscience'/f
        components[k]=json.loads(p.read_text()) if p.exists() else {'available':False}
    invariants={
        'candidate_boundary': all(c.get('candidateOnly') is True and c.get('level3Evidence') is False for c in components.values() if isinstance(c,dict) and c.get('schema')),
        'conscience_benchmark_ok': components['conscience'].get('ok') is True,
        'conformal_ok': components['conformal'].get('ok') is True,
        'constitutional_ok': components['constitutional'].get('ok') is True,
        'deception_ok': components['deception'].get('ok') is True,
        'semantic_entropy_ok': components['semanticEntropy'].get('ok') is True,
        'activation_probe_candidate_ok': components['activationProbe'].get('metrics',{}).get('recall',0) >= 0.75,
        'moral_public_standard_ok': components['moralPublicStandard'].get('ok') is True,
    }
    report={
        'schema':'sophia.conscience_proof_package.v1',
        'candidateOnly':True,
        'level3Evidence':False,
        'canClaimAGI':False,
        'claimBoundary':'Candidate moral + epistemic control infrastructure. This does not prove AGI.',
        'implementedPriorities':[
            'mandatory conscience enforcement', 'conformal abstention gate', 'semantic entropy upgrade',
            'constitutional benchmark', 'deception benchmark', 'activation probe path', 'conscience proof package',
            'public moral standard gate (overlapping consensus)'
        ],
        'components':components,
        'invariants':invariants,
        'ok':all(invariants.values()),
    }
    OUT.parent.mkdir(parents=True,exist_ok=True); OUT.write_text(json.dumps(sanitize_public_artifact(report),indent=2,ensure_ascii=False)+'\n')
    print(json.dumps({'ok':report['ok'],'out':str(OUT),'invariants':invariants},indent=2))
    return 0 if report['ok'] else 1
if __name__=='__main__': raise SystemExit(run())
