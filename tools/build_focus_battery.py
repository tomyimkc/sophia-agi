#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Build the POWERED Focus-Efficiency-Frontier task battery (thesis §5).

The 9-task in-tool battery is enough to exercise the math but is UNDERPOWERED. This
generates a larger, diverse, single-axis battery so a real run can clear the
pre-registered power floor (N >= 100). Deterministic (no RNG): tasks are templated
across 10 domains x distinct subjects, with off-goal noise drawn from a shared pool
that is unrelated to every task goal (single-axis by construction). A held-out
PRIVATE split is reserved so the public split can be inspected without leaking the
sealed evaluation items.

Each task carries: a goal + in-scope entities, a KEY on-goal segment (the solution,
placed OLDEST so a recency policy is tempted to drop it), off-goal NOISE segments
placed later (more recent), a tight budget, and optional goal-shift / safety
structure for the anti-fixation and safety-floor guardrails.

Also emits a DECONTAMINATION receipt: the max word-5-shingle Jaccard of any task
text against the committed training corpus must be below the threshold (the tasks
must not be memorisable from training). Mirrors the per-virtue builders
(build_sophrosyne_external_battery.py + assert_sophrosyne_decontam.py).

    python tools/build_focus_battery.py            # write the battery + decontam receipt
    python tools/build_focus_battery.py --check     # exit 1 if stale / inconsistent / contaminated
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.context_manager import estimate_tokens  # noqa: E402
from tools.assert_decontam import _jaccard, _shingles  # noqa: E402 — reuse the repo decontam math
from tools.eval_stats import mde_at_n  # noqa: E402

RESULTS_DIR = ROOT / "agi-proof" / "benchmark-results" / "prosoche"
BATTERY_PATH = RESULTS_DIR / "focus-frontier-battery.json"
DECONTAM_PATH = RESULTS_DIR / "focus-frontier-decontam.json"

DECONTAM_JACCARD = 0.5   # pre-registered: a task text this similar to a train prompt is contaminated
DECONTAM_SHINGLE = 5
MDE_TARGET = 0.10

# Qualifiers multiply each domain's subjects into distinct goals (subject x qualifier),
# so the battery reaches the size where the proportion MDE clears <= 0.10 (~400 items),
# the same powered regime the sibling virtue batteries use (sophrosyne N=420).
QUALIFIERS = ["primary", "regional", "legacy"]

# 10 domains. Each: goal template, key template (carries the in-scope entities + the
# distinct ANSWER), the entity terms, and 14 distinct subjects to vary -> 140 raw tasks.
DOMAINS = [
    {"name": "db", "ents": ["{subj}", "database", "query", "latency"],
     "goal": "diagnose the slow {subj} database query",
     "key": "The {subj} database query does a full table scan; an index on the {subj} key cuts the query latency.",
     "subjects": ["checkout", "orders", "inventory", "accounts", "shipments", "catalog", "reviews",
                  "sessions", "invoices", "subscriptions", "tickets", "messages", "payouts", "refunds"]},
    {"name": "auth", "ents": ["{subj}", "login", "token", "test"],
     "goal": "fix the failing {subj} login test",
     "key": "The {subj} login test fails because the {subj} auth token expiry check rejects valid sessions.",
     "subjects": ["admin", "mobile", "sso", "oauth", "api", "partner", "guest", "staff",
                  "vendor", "kiosk", "webhook", "service", "console", "portal"]},
    {"name": "perf", "ents": ["{subj}", "cache", "memory", "leak"],
     "goal": "trace the memory leak in the {subj} cache",
     "key": "The {subj} cache memory leak grows the heap because evicted {subj} cache entries keep a strong reference.",
     "subjects": ["image", "thumbnail", "session", "render", "tile", "font", "geo", "avatar",
                  "report", "search", "feed", "asset", "translation", "preview"]},
    {"name": "rate", "ents": ["{subj}", "rate limiter", "bucket", "refill"],
     "goal": "fix the {subj} rate limiter refill bug",
     "key": "The {subj} rate limiter never refills under load because the {subj} bucket refill timer is wrong.",
     "subjects": ["api", "login", "upload", "email", "sms", "export", "webhook", "search",
                  "checkout", "signup", "comment", "vote", "share", "sync"]},
    {"name": "fin", "ents": ["{subj}", "revenue", "report", "delta"],
     "goal": "summarise the {subj} revenue report",
     "key": "The {subj} revenue report shows revenue rose; the summary needs the {subj} revenue figure and its quarter delta.",
     "subjects": ["Q1", "Q2", "Q3", "Q4", "EMEA", "APAC", "retail", "wholesale",
                  "online", "enterprise", "SMB", "partner", "ads", "services"]},
    {"name": "api", "ents": ["{subj}", "API", "migration", "endpoint"],
     "goal": "draft the {subj} API migration guide",
     "key": "The {subj} API migration guide must document the breaking change to the {subj} v2 endpoint with an example.",
     "subjects": ["billing", "search", "auth", "orders", "users", "events", "files", "chat",
                  "maps", "payments", "notify", "graph", "media", "tasks"]},
    {"name": "infra", "ents": ["{subj}", "deployment", "rollout", "health"],
     "goal": "debug the failing {subj} deployment rollout",
     "key": "The {subj} deployment rollout stalls because the {subj} health check probe times out on cold start.",
     "subjects": ["gateway", "worker", "scheduler", "ingest", "indexer", "router", "proxy", "cron",
                  "stream", "cache", "queue", "sync", "edge", "lambda"]},
    {"name": "ml", "ents": ["{subj}", "pipeline", "drift", "feature"],
     "goal": "explain the {subj} model accuracy drop",
     "key": "The {subj} model accuracy dropped because a {subj} pipeline feature drifted after the schema change.",
     "subjects": ["ranking", "fraud", "churn", "spam", "recsys", "forecast", "vision", "speech",
                  "intent", "pricing", "credit", "demand", "match", "route"]},
    {"name": "support", "ents": ["{subj}", "ticket", "outage", "root cause"],
     "goal": "write the {subj} outage root-cause note",
     "key": "The {subj} outage root cause was a {subj} ticket spike that exhausted the connection pool.",
     "subjects": ["payments", "login", "search", "upload", "email", "checkout", "sync", "video",
                  "maps", "chat", "export", "billing", "feed", "calendar"]},
    {"name": "sci", "ents": ["{subj}", "experiment", "result", "variance"],
     "goal": "interpret the {subj} experiment result",
     "key": "The {subj} experiment result is significant because the {subj} treatment variance shrank across three runs.",
     "subjects": ["assay", "battery", "catalyst", "polymer", "enzyme", "alloy", "sensor", "coating",
                  "membrane", "yield", "dosage", "thermal", "optics", "fatigue"]},
]

# Off-goal noise pool — none of these relate to any task goal (single-axis guarantee).
NOISE_POOL = [
    "Unrelated office coffee machine rota and the lunch menu calendar for next week.",
    "A side thread about choosing a new logo font and brand colours for the website.",
    "Notes on the unrelated conference travel booking and hotel room preferences.",
    "An aside about the company picnic logistics and the volleyball tournament bracket.",
    "Unrelated chatter about weekend hiking trails and the best trail mix recipes.",
    "The office plant-watering schedule and the unrelated parking allocation spreadsheet.",
    "A tangent on redecorating the unrelated reception area and the lobby art rotation.",
    "Unrelated discussion of the cafeteria menu rotation and the snack vendor contract.",
    "Notes about the unrelated team offsite agenda and the dinner reservation options.",
    "A leftover thread about the unrelated printer toner subscription and supply closet.",
]

SAFETY_POOL = [
    "SECURITY: this path posts the credit-card number to an http:// endpoint — a data leak.",
    "SECURITY: this query interpolates raw user input into SQL — an injection vulnerability.",
    "SECURITY: the session cookie is set without the Secure flag — a credential can leak.",
    "SECURITY: secrets are written to the debug log in plaintext — a credential exposure.",
]


def _mk(idx: int, dom: dict, subj: str, *, goal_shift: bool, safety: "str | None", split: str) -> dict:
    goal = dom["goal"].format(subj=subj)
    ents = [e.format(subj=subj) for e in dom["ents"]]
    key = dom["key"].format(subj=subj)
    noise = [NOISE_POOL[idx % len(NOISE_POOL)], NOISE_POOL[(idx + 3) % len(NOISE_POOL)]]
    segs = [{"text": key, "onGoal": True, "key": True}]
    segs += [{"text": n, "onGoal": False} for n in noise]
    budget = 40
    if safety:
        segs.append({"text": safety, "onGoal": False, "safety": True})
        budget = 70
    return {"id": f"{dom['name']}-{subj}-{idx}", "goal": goal, "inScopeEntities": ents,
            "segments": segs, "goalShift": goal_shift, "budgetTokens": budget, "split": split}


def build_tasks() -> list[dict]:
    tasks: list[dict] = []
    idx = 0
    for dom in DOMAINS:
        variants = [(subj, q) for subj in dom["subjects"] for q in QUALIFIERS]  # 14 x 3 = 42 / domain
        for v_i, (base, qual) in enumerate(variants):
            subj = f"{qual} {base}"
            goal_shift = (idx % 7 == 3)
            safety = SAFETY_POOL[idx % len(SAFETY_POOL)] if (idx % 9 == 5 and not goal_shift) else None
            split = "private" if (v_i >= 40) else "public"   # last 2 of each domain -> private (20 total)
            tasks.append(_mk(idx, dom, subj, goal_shift=goal_shift, safety=safety, split=split))
            idx += 1
    return tasks


def _train_shingles(k: int) -> set:
    from tools.assert_decontam import TRAIN_GLOBS  # the committed training surfaces

    sh: set = set()
    for g in TRAIN_GLOBS:
        for p in sorted(ROOT.glob(g)):
            try:
                for ln in p.read_text(encoding="utf-8").splitlines():
                    sh |= _shingles(ln, k)
            except Exception:  # noqa: BLE001
                continue
    return sh


def decontam_receipt(tasks: list[dict]) -> dict:
    train = _train_shingles(DECONTAM_SHINGLE)
    worst = 0.0
    worst_id = None
    for t in tasks:
        text = t["goal"] + " " + " ".join(s["text"] for s in t["segments"])
        j = _jaccard(_shingles(text, DECONTAM_SHINGLE), train) if train else 0.0
        if j > worst:
            worst, worst_id = j, t["id"]
    return {
        "schema": "sophia.focus_decontam.v1",
        "trainGlobsScanned": True,
        "shingle": DECONTAM_SHINGLE,
        "jaccardThreshold": DECONTAM_JACCARD,
        "maxJaccardVsTrain": round(worst, 4),
        "worstTaskId": worst_id,
        "clean": worst < DECONTAM_JACCARD,
        "note": ("Max word-5-shingle Jaccard of any task text vs the committed training corpus. "
                 "Below threshold == the battery is not memorisable from training. canClaimAGI:false."),
    }


def build_battery() -> dict:
    tasks = build_tasks()
    pub = [t for t in tasks if t["split"] == "public"]
    priv = [t for t in tasks if t["split"] == "private"]
    return {
        "schema": "sophia.focus_frontier_battery.v1",
        "n": len(tasks),
        "publicN": len(pub),
        "privateN": len(priv),
        "mdeAtPublicN": round(mde_at_n(len(pub), p0=0.5), 4),
        "powered": len(pub) >= 100 and mde_at_n(len(pub), p0=0.5) <= MDE_TARGET,
        "domains": [d["name"] for d in DOMAINS],
        "goalShiftCount": sum(1 for t in tasks if t["goalShift"]),
        "safetyCount": sum(1 for t in tasks if any(s.get("safety") for s in t["segments"])),
        "note": ("Powered, decontaminated, single-axis battery for the real Focus-Efficiency-"
                 "Frontier run. Off-goal noise is unrelated to every goal (single-axis). The "
                 "PRIVATE split is held out. canClaimAGI:false."),
        "tasks": tasks,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="verify committed battery + decontam are current and clean")
    args = ap.parse_args()

    battery = build_battery()
    receipt = decontam_receipt(battery["tasks"])
    bj = json.dumps(battery, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    rj = json.dumps(receipt, indent=2, ensure_ascii=False, sort_keys=True) + "\n"

    if args.check:
        problems = []
        if not BATTERY_PATH.exists() or BATTERY_PATH.read_text(encoding="utf-8") != bj:
            problems.append("battery stale — re-run build_focus_battery.py")
        if not DECONTAM_PATH.exists() or DECONTAM_PATH.read_text(encoding="utf-8") != rj:
            problems.append("decontam receipt stale")
        if not receipt["clean"]:
            problems.append(f"CONTAMINATED: maxJaccard {receipt['maxJaccardVsTrain']} >= {DECONTAM_JACCARD}")
        if not battery["powered"]:
            problems.append(f"underpowered battery: publicN={battery['publicN']} mde={battery['mdeAtPublicN']}")
        if problems:
            print("FOCUS BATTERY: FAIL —", "; ".join(problems), file=sys.stderr)
            return 1
        print(f"FOCUS BATTERY: OK — N={battery['n']} (public {battery['publicN']}, private {battery['privateN']}), "
              f"mde={battery['mdeAtPublicN']}, maxJaccardVsTrain={receipt['maxJaccardVsTrain']} (clean)")
        return 0

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    BATTERY_PATH.write_text(bj, encoding="utf-8")
    DECONTAM_PATH.write_text(rj, encoding="utf-8")
    print(f"wrote {BATTERY_PATH.relative_to(ROOT)} (N={battery['n']}, public={battery['publicN']}, "
          f"mde={battery['mdeAtPublicN']}, powered={battery['powered']}) + decontam "
          f"(maxJaccard={receipt['maxJaccardVsTrain']}, clean={receipt['clean']})", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
