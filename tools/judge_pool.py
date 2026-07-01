#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Judge POOL — multiple endpoint replicas (lanes) per judge FAMILY, for the local judge farm.

The throughput sim (`tools/cluster_schedule_sim.py`) shows the judged-sweep queue plateaus at
~1.36x regardless of Spark count because ~7/10 jobs funnel through ONE serialized Mac judge. The
fix is to let each judge FAMILY be served by MULTIPLE endpoint REPLICAS (lanes) — e.g. the 70B
"mlx" family served by the Mac AND a 70B running on 2 spare Sparks — so judge requests distribute
across lanes instead of queueing on one box.

This module is the routing layer. It is **pure, offline, deterministic** (no `random`, no clock,
no GPU, no network). Each replica is the existing judge-spec string
``provider:model@http://host:port/v1`` that ``agent.model.default_client`` already builds an
endpoint client from. A ``JudgePool`` maps ``family -> [endpoint specs]``.

CRITICAL INVARIANT (do not violate): this changes **how many lanes serve each family, NEVER which
families judge**. Replicas of the SAME model are the SAME family — adding lanes does NOT add
families and must NOT change any verdict. The 2-family VALIDATED gate (κ≥0.40, ≥2 DISTINCT
families, judge≠subject) is untouched. ``families()`` counts via
``run_lora_uplift_validation._family_key`` so it stays in LOCKSTEP with the gate's family counting;
``validate_pool`` REFUSES a "family" whose replicas key to different families (a misconfig that
would silently change the family count) and REFUSES a pool with <2 distinct families.

design/infra; no capability claim; canClaimAGI stays false.

CLI:
    python tools/judge_pool.py --self-test
    python tools/judge_pool.py --config config/inference.local.judge-pool.json --families
    python tools/judge_pool.py --config config/inference.local.judge-pool.json --validate
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Reuse the gate's family keying VERBATIM — do NOT fork it. This is the single source of truth for
# "what family is this judge spec", so the pool counts families EXACTLY as the κ gate does.
from tools.run_lora_uplift_validation import _family_key  # noqa: E402


# A pool is a plain dict: {family_label: [judge_spec, ...]}. The family_label is a human label
# (e.g. "mlx", "qwen"); the AUTHORITATIVE family is always recomputed from the spec via _family_key
# so a mislabeled key cannot fool the gate. We keep the label only for readability / config shape.
Pool = "dict[str, list[str]]"


def load_pool(config_dict: dict) -> "dict[str, list[str]]":
    """Build a pool ``{family_label: [spec, ...]}`` from a judge-pool config dict.

    Accepted shapes (both keep the existing judge-config JSON style):
      * ``{"families": {"mlx": ["mlx:...@http://a/v1", ...], "qwen": [...]}}``
      * ``{"families": {"mlx": {"replicas": ["..."]}, ...}}``  (replicas under a sub-key, so a
        family entry can also carry comments/metadata like the existing configs do)

    Pure: no I/O. The caller reads the JSON; this only normalizes it. Order is preserved
    (round-robin / least-loaded tie-breaks are deterministic regardless, but a stable order keeps
    the worked example legible)."""
    if not isinstance(config_dict, dict):
        raise ValueError("judge-pool config must be a JSON object")
    fams = config_dict.get("families")
    if not isinstance(fams, dict) or not fams:
        raise ValueError("judge-pool config must have a non-empty 'families' object")
    pool: "dict[str, list[str]]" = {}
    for label, entry in fams.items():
        if isinstance(entry, dict):
            specs = entry.get("replicas") or entry.get("specs") or entry.get("endpoints")
        else:
            specs = entry
        if not isinstance(specs, list) or not specs:
            raise ValueError(f"family {label!r} must list >=1 endpoint spec (a 'replicas' list)")
        clean = [str(s).strip() for s in specs if str(s).strip()]
        if not clean:
            raise ValueError(f"family {label!r} has no usable endpoint specs")
        pool[str(label)] = clean
    return pool


def families(pool: "dict[str, list[str]]") -> "list[str]":
    """The DISTINCT judge family-keys in the pool, via ``_family_key`` (the gate's keying). This is
    the number the 2-family gate counts — NOT the count of config labels. Replicas of the same
    model collapse to one family. Sorted for determinism."""
    fams = set()
    for specs in pool.values():
        for spec in specs:
            fams.add(_family_key(spec))
    return sorted(fams)


def endpoints_for(pool: "dict[str, list[str]]", family: str) -> "list[str]":
    """All replica specs whose family-key == ``family`` (across every config label). So asking for
    family ``"mlx"`` returns every lane that the gate would count as ``mlx``, even if the config
    split them across labels. Order = config order. Empty list if no such family."""
    out: "list[str]" = []
    for specs in pool.values():
        for spec in specs:
            if _family_key(spec) == family:
                out.append(spec)
    return out


def next_endpoint(pool: "dict[str, list[str]]", family: str,
                  in_flight_per_endpoint: "dict[str, int] | None") -> str:
    """Pick the LEAST-LOADED replica for ``family`` given current in-flight counts. Deterministic:
    ties break by the spec string (lexicographic), NO random. ``in_flight_per_endpoint`` maps a
    spec -> number of requests currently outstanding on it (missing = 0). This is the only routing
    decision; it changes WHICH lane serves a request, never the request or its verdict."""
    eps = endpoints_for(pool, family)
    if not eps:
        raise ValueError(f"no endpoints for family {family!r} in pool")
    counts = in_flight_per_endpoint or {}
    # least in-flight first; deterministic tie-break by spec string. (key over the config-ordered
    # list so a stable, reproducible lane is chosen every call.)
    return min(eps, key=lambda spec: (int(counts.get(spec, 0)), spec))


def validate_pool(pool: "dict[str, list[str]]") -> "tuple[bool, dict]":
    """Assert the pool is gate-safe. RAISES ``ValueError`` on a misconfig; returns ``(True, info)``
    when valid. Two hard checks:

      1. **>=2 DISTINCT families** — the κ gate needs at least two. (Replicas do NOT add families,
         so a pool that is "one family on many lanes" is REFUSED here.)
      2. **replicas within a config-family are the SAME family-key** — a config label whose
         replicas mix families (e.g. a qwen spec sneaked into the "mlx" family) is a silent
         family-count corruption and is REFUSED.

    These two together guarantee adding lanes never changes the family count the gate sees."""
    if not pool:
        raise ValueError("empty judge pool")
    # check 2 first: each config label must be internally single-family.
    for label, specs in pool.items():
        keys = {_family_key(s) for s in specs}
        if len(keys) != 1:
            raise ValueError(
                f"family label {label!r} mixes judge families {sorted(keys)} across its replicas; "
                f"replicas of one family must all key to the SAME family (adding LANES must not add "
                f"FAMILIES)")
    fams = families(pool)
    if len(fams) < 2:
        raise ValueError(
            f"judge pool has {len(fams)} distinct families {fams}; the VALIDATED gate needs >=2 "
            f"DISTINCT families (κ>=0.40, judge!=subject). Adding replicas/lanes does NOT add a "
            f"family.")
    return True, {
        "families": fams,
        "lanesPerFamily": {f: len(endpoints_for(pool, f)) for f in fams},
        "totalLanes": sum(len(s) for s in pool.values()),
    }


def total_lanes(pool: "dict[str, list[str]]") -> int:
    """Total number of endpoint replicas (lanes) across all families — what the sim treats as the
    judge concurrency."""
    return sum(len(s) for s in pool.values())


def lanes_for_family(pool: "dict[str, list[str]]", family: str) -> int:
    """Number of lanes serving ``family`` (the sim's mac-bound-family concurrency)."""
    return len(endpoints_for(pool, family))


# --- offline invariants / self-test -------------------------------------------------------------
def _example_pool() -> "dict[str, list[str]]":
    """The worked example: qwen = 1 lane; the 70B family = 3 lanes (Mac + 2 spare Sparks).

    IMPORTANT family-key subtlety (this is WHY all three 70B lanes use the SAME provider+vendor):
    ``_family_key`` keys an AGGREGATOR provider (vllm/sglang/llamacpp/openai/openrouter) by the
    model VENDOR prefix, but keys a non-aggregator engine (mlx/ollama) by the ENGINE. So
    ``vllm:mlx-community/...`` -> ``mlx-community`` while ``mlx:mlx-community/...`` -> ``mlx``: the
    SAME weights served by two different engines would key to two DIFFERENT families and silently
    inflate the family count. To keep the 70B a SINGLE family across all lanes, every replica is
    served the SAME way — here ``vllm`` (keyless) on every box, so all three key to the vendor
    family ``mlx-community``. ``validate_pool`` ENFORCES this (it refuses a label whose replicas key
    differently)."""
    return {
        "qwen": ["vllm:Qwen/Qwen2.5-7B-Instruct@http://169.254.10.1:8000/v1"],
        "mlx-community": [
            "vllm:mlx-community/Llama-3.3-70B-Instruct-4bit@http://169.254.26.171:8081/v1",
            "vllm:mlx-community/Llama-3.3-70B-Instruct-4bit@http://169.254.10.2:8001/v1",
            "vllm:mlx-community/Llama-3.3-70B-Instruct-4bit@http://169.254.10.3:8001/v1",
        ],
    }


def offline_invariants() -> "tuple[bool, dict]":
    checks: "dict[str, bool]" = {}
    pool = _example_pool()

    # 1) families() counts via the gate's _family_key (2 distinct families: qwen + mlx-community).
    #    NB: all 3 70B replicas use vllm (aggregator) -> vendor family 'mlx-community': ONE family,
    #    3 lanes. (The Mac's mlx-ENGINE spec would key to 'mlx' instead — see _example_pool.)
    fams = families(pool)
    checks["families_via_family_key"] = (fams == ["mlx-community", "qwen"])
    checks["replicas_same_family"] = (
        len(endpoints_for(pool, "mlx-community")) == 3 and
        len(families({"mlx-community": pool["mlx-community"]})) == 1)

    # 2) validate_pool accepts the 2-family pool.
    try:
        ok, info = validate_pool(pool)
        checks["validate_accepts_2family"] = (ok and info["totalLanes"] == 4)
    except ValueError:
        checks["validate_accepts_2family"] = False

    # 3) validate_pool REFUSES a pool with <2 distinct families (one family, many lanes).
    one_fam = {"mlx-community": pool["mlx-community"]}
    refused_1fam = False
    try:
        validate_pool(one_fam)
    except ValueError:
        refused_1fam = True
    checks["refuses_under_2_families"] = refused_1fam

    # 4) validate_pool REFUSES a config label that MIXES families (qwen spec under "mlx-community").
    mixed = {"mlx-community": pool["mlx-community"] + ["vllm:Qwen/Qwen2.5-7B-Instruct@http://x:1/v1"],
             "qwen": pool["qwen"]}
    refused_mixed = False
    try:
        validate_pool(mixed)
    except ValueError:
        refused_mixed = True
    checks["refuses_mixed_family_label"] = refused_mixed

    # 5) next_endpoint is least-loaded and deterministic; ties break by spec string (no random).
    mlx_lanes = pool["mlx-community"]
    empty_load = {s: 0 for s in mlx_lanes}
    first = next_endpoint(pool, "mlx-community", empty_load)
    checks["tie_break_lexicographic"] = (first == min(mlx_lanes))
    # least-loaded: load the lexicographically-first lane, the next pick must be a different lane.
    loaded = dict(empty_load); loaded[first] = 5
    second = next_endpoint(pool, "mlx-community", loaded)
    checks["picks_least_loaded"] = (second != first)
    # deterministic across calls
    checks["next_endpoint_deterministic"] = (
        next_endpoint(pool, "mlx-community", loaded) == next_endpoint(pool, "mlx-community", loaded))

    # 6) round-robin via repeatedly incrementing the chosen lane spreads load evenly across lanes.
    rr_load = {s: 0 for s in mlx_lanes}
    for _ in range(9):  # 9 requests over 3 lanes -> 3 each
        pick = next_endpoint(pool, "mlx-community", rr_load)
        rr_load[pick] += 1
    checks["round_robin_balanced"] = (sorted(rr_load.values()) == [3, 3, 3])

    # 7) load_pool accepts both shapes (flat list + {"replicas": [...]}).
    flat = load_pool({"families": {"a": ["mlx:m@http://h1/v1"], "b": ["ollama:q@http://h2/v1"]}})
    nested = load_pool({"families": {"a": {"replicas": ["mlx:m@http://h1/v1"]},
                                     "b": {"replicas": ["ollama:q@http://h2/v1"]}}})
    checks["load_pool_shapes"] = (flat == nested)

    # 8) endpoints_for(family) gathers a family even if split across config labels.
    split_cfg = {"mlx-a": [pool["mlx-community"][0]], "mlx-b": pool["mlx-community"][1:],
                 "qwen": pool["qwen"]}
    checks["endpoints_for_crosses_labels"] = (len(endpoints_for(split_cfg, "mlx-community")) == 3)

    return all(checks.values()), {"checks": checks}


# --- CLI ----------------------------------------------------------------------------------------
def _load_config_file(path: Path) -> "dict[str, list[str]]":
    return load_pool(json.loads(path.read_text(encoding="utf-8")))


def main(argv: "list[str] | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, help="judge-pool config JSON")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--families", action="store_true", help="print distinct families (gate keying)")
    ap.add_argument("--validate", action="store_true", help="validate the pool (>=2 families etc.)")
    args = ap.parse_args(argv)

    if args.self_test:
        ok, detail = offline_invariants()
        print("judge_pool invariants:", "PASS" if ok else "FAIL")
        for k, v in detail["checks"].items():
            print(f"  [{'ok' if v else 'XX'}] {k}")
        return 0 if ok else 1

    if not args.config:
        ap.print_help()
        return 0
    try:
        pool = _load_config_file(args.config)
    except (ValueError, json.JSONDecodeError, OSError) as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return 2

    if args.validate:
        try:
            ok, info = validate_pool(pool)
        except ValueError as e:
            print(f"REFUSED: {e}", file=sys.stderr)
            return 2
        print(json.dumps(info, indent=2, ensure_ascii=False))
        return 0

    if args.families:
        print(json.dumps({"families": families(pool),
                          "lanesPerFamily": {f: lanes_for_family(pool, f) for f in families(pool)},
                          "totalLanes": total_lanes(pool)}, indent=2, ensure_ascii=False))
        return 0

    # default: dump the normalized pool
    print(json.dumps(pool, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
