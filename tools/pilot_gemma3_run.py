#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""M3 PILOT runner (executes ON a CUDA pod) — gemma-3-4b LoRA train + evaluate.

Self-contained so a rented GPU pod can: (1) LoRA-fine-tune ONLY the language tower of
the multimodal `google/gemma-3-4b-it` on the corpus-bound 965-row gate-passed dataset,
then (2) evaluate base-vs-adapter on the held-out wisdom-market benchmark using the SAME
deterministic scoring as the M1 instrument (`tools/run_same_size_market_baselines.py`),
batched through transformers (no vLLM, no serving) so the N=354 x 3-run protocol stays
tractable. Writes one report JSON the pre-registered M3-pilot go/no-go reads.

Pre-registered spec: docs/06-Roadmap/Sophia-Wisdom-4B-M3-Pilot.md. This script makes NO
claim; it produces the numbers. A null result is a legitimate outcome.

Usage (on the pod):
    HF_TOKEN=... python tools/pilot_gemma3_run.py --smoke           # cheap load + 2 steps + 1 gen
    HF_TOKEN=... python tools/pilot_gemma3_run.py --train --eval --runs 3
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Reuse the M1 instrument's TESTED scoring + system prompts (judge-independent, deterministic).
import importlib.util  # noqa: E402
_spec = importlib.util.spec_from_file_location("ssmb", ROOT / "tools" / "run_same_size_market_baselines.py")
SSMB = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(SSMB)

BENCH = ROOT / "data" / "wisdom_market_benchmark" / "heldout_v1.jsonl"
TRAIN = ROOT / "training" / "local_sophia_v3" / "mlx" / "train.jsonl"
DEFAULT_MODEL = "google/gemma-3-4b-it"
DEFAULT_ADAPTER = ROOT / "training" / "adapters" / "sophia-wisdom-4b-pilot"
LANG_LORA_REGEX = r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$"


def log(msg: str) -> None:
    print(f"[pilot {time.strftime('%H:%M:%S')}] {msg}", flush=True)


# --------------------------------------------------------------------------- #
# Model loading (multimodal gemma-3: train/generate the LANGUAGE tower only)   #
# --------------------------------------------------------------------------- #
def load_base(model_id: str, *, dtype="bfloat16"):
    import torch
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    td = getattr(torch, dtype)
    model = None
    errs = []
    # Gemma3 is multimodal (Gemma3ForConditionalGeneration). Try the most specific
    # loaders first; all expose .generate for text-only inputs and a .language_model tower.
    for loader in ("AutoModelForImageTextToText", "Gemma3ForConditionalGeneration", "AutoModelForCausalLM"):
        try:
            import transformers
            cls = getattr(transformers, loader)
            model = cls.from_pretrained(model_id, torch_dtype=td, device_map="auto", trust_remote_code=True)
            log(f"loaded base via {loader}")
            break
        except Exception as exc:  # try the next loader
            errs.append(f"{loader}: {type(exc).__name__}: {exc}")
    if model is None:
        raise RuntimeError("could not load gemma-3 base; tried:\n  " + "\n  ".join(errs))
    return model, tok


def _chat_ids(tok, messages, *, max_len, add_generation_prompt):
    """Tokenize a chat. Gemma's template has no system role — fold any system text into
    the first user turn so apply_chat_template never raises. Returns a flat list[int]
    regardless of whether the template yields a list, a nested list, a tensor, or a
    BatchEncoding/dict (gemma-3's multimodal template returns the latter)."""
    msgs = [dict(m) for m in messages]
    if msgs and msgs[0].get("role") == "system":
        sys_txt = msgs.pop(0)["content"]
        for m in msgs:
            if m.get("role") == "user":
                m["content"] = f"{sys_txt}\n\n{m['content']}"
                break
    out = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=add_generation_prompt,
                                  truncation=True, max_length=max_len)
    if hasattr(out, "input_ids"):          # BatchEncoding
        out = out["input_ids"]
    if isinstance(out, dict):              # plain dict
        out = out["input_ids"]
    try:
        import torch
        if isinstance(out, torch.Tensor):
            out = out.tolist()
    except Exception:
        # torch may be absent or `out` already a plain list; leave `out` as-is
        pass
    if out and isinstance(out[0], (list, tuple)):  # nested [[...]] -> first row
        out = out[0]
    return [int(t) for t in out]


# --------------------------------------------------------------------------- #
# Train                                                                       #
# --------------------------------------------------------------------------- #
def train(model_id: str, adapter_out: Path, *, rows_path: Path, seq_len: int, epochs: int,
          seed: int, lr: float, smoke: bool, lora_rank: int = 16, lora_alpha: int = 32,
          use_rslora: bool = False, kl_coef: float = 0.0) -> None:
    """LoRA SFT on the language tower. Stability knobs (anti-forgetting, 2026-06-26):
    - lora_rank/lora_alpha + use_rslora: LOWER capacity overwrites less of the base.
    - kl_coef>0: KL-ANCHOR the adapter to the FROZEN base on the GENERAL-RETENTION rows only
      (reference logits via peft `disable_adapter()` — no 2nd model), so the general slice is
      held near base while the discipline rows still move freely."""
    import torch
    import torch.nn.functional as F
    from peft import LoraConfig, get_peft_model
    from torch.utils.data import Dataset, DataLoader
    from transformers import get_cosine_schedule_with_warmup

    model, tok = load_base(model_id)
    model.config.use_cache = False
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    lora = LoraConfig(r=lora_rank, lora_alpha=lora_alpha, lora_dropout=0.05, bias="none",
                      task_type="CAUSAL_LM", target_modules=LANG_LORA_REGEX, use_rslora=use_rslora)
    model = get_peft_model(model, lora)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"trainable params: {n_train:,} (rank={lora_rank} alpha={lora_alpha} rslora={use_rslora} kl_coef={kl_coef})")
    if n_train == 0:
        raise RuntimeError("LoRA matched 0 modules — the language-tower regex needs adjusting")

    rows = [json.loads(l) for l in rows_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    if smoke:
        rows = rows[:8]
    n_gen = sum(1 for r in rows if (r.get("metadata") or {}).get("task_family") == "general_retention")
    log(f"training rows: {len(rows)} ({n_gen} general_retention — KL-anchored if kl_coef>0)")

    class DS(Dataset):
        def __len__(self): return len(rows)
        def __getitem__(self, i):
            msgs = rows[i]["messages"]
            full = _chat_ids(tok, msgs, max_len=seq_len, add_generation_prompt=False)
            # mask everything before the assistant turn (prompt-masked loss)
            prompt_only = _chat_ids(tok, [m for m in msgs if m["role"] != "assistant"],
                                    max_len=seq_len, add_generation_prompt=True)
            labels = list(full)
            for j in range(min(len(prompt_only), len(labels))):
                labels[j] = -100
            is_general = (rows[i].get("metadata") or {}).get("task_family") == "general_retention"
            return {"input_ids": full, "labels": labels, "is_general": is_general}

    def collate(batch):
        maxlen = max(len(b["input_ids"]) for b in batch)
        pad = tok.pad_token_id
        ids, lbl, att, gen = [], [], [], []
        for b in batch:
            n = maxlen - len(b["input_ids"])
            ids.append(b["input_ids"] + [pad] * n)
            lbl.append(b["labels"] + [-100] * n)
            att.append([1] * len(b["input_ids"]) + [0] * n)
            gen.append(bool(b["is_general"]))
        return (torch.tensor(ids), torch.tensor(lbl), torch.tensor(att), gen)

    torch.manual_seed(seed)
    dl = DataLoader(DS(), batch_size=1, shuffle=True, collate_fn=collate)
    steps = (2 if smoke else len(dl) * epochs)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    sched = get_cosine_schedule_with_warmup(opt, int(0.03 * steps), steps)
    model.train()
    dev = next(model.parameters()).device
    done = 0
    for _ep in range(epochs):
        for ids, lbl, att, gen in dl:
            ids, lbl, att = ids.to(dev), lbl.to(dev), att.to(dev)
            out = model(input_ids=ids, attention_mask=att, labels=lbl)
            loss = out.loss
            kl_val = 0.0
            # KL-anchor ONLY the general-retention rows to the frozen base (preserve general
            # capability) while discipline rows move freely under the plain SFT loss.
            if kl_coef > 0 and any(gen):
                with torch.no_grad():
                    with model.disable_adapter():
                        ref_logits = model(input_ids=ids, attention_mask=att).logits
                mask = (att == 1).unsqueeze(-1)  # ignore pad positions
                lp = F.log_softmax(out.logits, dim=-1)
                rp = F.softmax(ref_logits, dim=-1)
                kl_tok = (rp * (rp.clamp_min(1e-9).log() - lp)).sum(-1, keepdim=True)  # KL(base||adapter)
                kl = (kl_tok * mask).sum() / mask.sum().clamp_min(1)
                loss = loss + kl_coef * kl
                kl_val = float(kl.detach())
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step(); sched.step(); opt.zero_grad()
            done += 1
            if done % 25 == 0 or smoke:
                log(f"step {done}/{steps} loss={out.loss.item():.4f} kl={kl_val:.4f}")
            if smoke and done >= 2:
                break
        if smoke:
            break
    adapter_out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(adapter_out))
    tok.save_pretrained(str(adapter_out))
    (adapter_out / "pilot_train_meta.json").write_text(json.dumps(
        {"baseModel": model_id, "rows": len(rows), "generalRows": n_gen, "epochs": epochs,
         "seed": seed, "seqLen": seq_len, "lr": lr, "loraRank": lora_rank, "loraAlpha": lora_alpha,
         "useRslora": use_rslora, "klCoef": kl_coef, "trainableParams": n_train, "smoke": smoke}, indent=2))
    log(f"saved adapter -> {adapter_out}")


# --------------------------------------------------------------------------- #
# Eval (batched generation; reuse instrument scoring)                          #
# --------------------------------------------------------------------------- #
def _batched_generate(model, tok, systems_users, *, max_new=400, batch_size=32, seq_len=1024):
    import torch
    tok.padding_side = "left"
    outs = []
    dev = next(model.parameters()).device
    for i in range(0, len(systems_users), batch_size):
        chunk = systems_users[i:i + batch_size]
        enc = []
        for system, user in chunk:
            ids = _chat_ids(tok, [{"role": "system", "content": system}, {"role": "user", "content": user}],
                            max_len=seq_len, add_generation_prompt=True)
            enc.append(ids)
        maxlen = max(len(e) for e in enc)
        pad = tok.pad_token_id
        input_ids = torch.tensor([[pad] * (maxlen - len(e)) + e for e in enc]).to(dev)
        attn = torch.tensor([[0] * (maxlen - len(e)) + [1] * len(e) for e in enc]).to(dev)
        with torch.no_grad():
            gen = model.generate(input_ids=input_ids, attention_mask=attn, max_new_tokens=max_new,
                                 do_sample=True, temperature=0.2, top_p=0.95, pad_token_id=pad)
        for j in range(len(chunk)):
            text = tok.decode(gen[j][input_ids.shape[1]:], skip_special_tokens=True)
            outs.append(text)
        log(f"  gen {min(i + batch_size, len(systems_users))}/{len(systems_users)}")
    return outs


def _run_conditions_for_model(model, tok, cases, runs, *, capture=None):
    """Return {condition: [per-run metric dicts]} reusing SSMB scoring. Generates ONCE per
    (raw, advisor) system per run and scores advisor output as both prompt & prompt_gate.
    If ``capture`` is a list, the run-0 ADVISOR (prompt-condition, no-gate) answers are
    appended to it for later independent LLM-judge scoring (the primary-signal layer)."""
    raw_sys = SSMB.RAW_SYSTEM
    adv_sys = SSMB.system_for("prompt")
    out = {"raw": [], "prompt": [], "prompt_gate": []}
    for r in range(runs):
        raw_ans = _batched_generate(model, tok, [(raw_sys, c["prompt"]) for c in cases])
        adv_ans = _batched_generate(model, tok, [(adv_sys, c["prompt"]) for c in cases])
        if capture is not None and r == 0:
            capture.extend(adv_ans)
        out["raw"].append(SSMB.aggregate_metrics(
            [SSMB.score_case(c, a, gated=False) for c, a in zip(cases, raw_ans)]))
        out["prompt"].append(SSMB.aggregate_metrics(
            [SSMB.score_case(c, a, gated=False) for c, a in zip(cases, adv_ans)]))
        out["prompt_gate"].append(SSMB.aggregate_metrics(
            [SSMB.score_case(c, a, gated=True) for c, a in zip(cases, adv_ans)]))
        log(f"  run {r + 1}/{runs} done")
    return out


def evaluate(model_id: str, adapter_dir: Path, *, runs: int, limit, out_path: Path,
             answers_path: "Path | None" = None, bench_path: Path = BENCH) -> None:
    import torch  # noqa: F401
    from peft import PeftModel
    cases = SSMB.load_cases(bench_path, limit)
    log(f"eval cases: {len(cases)} x {runs} runs x (base, adapter)")

    base, tok = load_base(model_id)
    base.eval()
    base_cap = [] if answers_path else None
    base_runs = _run_conditions_for_model(base, tok, cases, runs, capture=base_cap)

    log("loading adapter onto base ...")
    adapted = PeftModel.from_pretrained(base, str(adapter_dir))
    adapted.eval()
    adapter_cap = [] if answers_path else None
    adapter_runs = _run_conditions_for_model(adapted, tok, cases, runs, capture=adapter_cap)

    if answers_path is not None:
        rows = []
        for i, c in enumerate(cases):
            rows.append({
                "id": c.get("id"), "task_family": c.get("task_family"),
                "language": c.get("language"), "prompt": c["prompt"],
                "gold_route": c.get("gold_route"),
                "forbidden_assertions": c.get("forbidden_assertions"),
                "acceptable_answer_features": c.get("acceptable_answer_features"),
                "base_answer": base_cap[i] if i < len(base_cap) else "",
                "adapter_answer": adapter_cap[i] if i < len(adapter_cap) else "",
            })
        answers_path.parent.mkdir(parents=True, exist_ok=True)
        answers_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
        log(f"wrote {len(rows)} base/adapter answer pairs -> {answers_path}")

    def pack(cond_runs):
        return {c: {"metrics": SSMB.aggregate_runs(rs)} for c, rs in cond_runs.items()}

    report = {
        "pilot": "sophia-wisdom-4b-m3",
        "baseModel": model_id,
        "benchmark": str(BENCH.relative_to(ROOT)), "nCases": len(cases), "runs": runs,
        "base": pack(base_runs), "adapter": pack(adapter_runs),
        # The pilot's PRIMARY signal: adapter(prompt) vs base(prompt) — did the WEIGHTS move the
        # no-gate behavior toward the gated target? Plus protected-suite guardrails.
        "adapterPromptVsBasePrompt": SSMB.deltas_vs_raw(adapter_runs["prompt"], base_runs["prompt"]),
        "adapterGateVsBaseGate": SSMB.deltas_vs_raw(adapter_runs["prompt_gate"], base_runs["prompt_gate"]),
        "boundary": ("Pilot feasibility numbers; deterministic structural metrics; no LLM judge; "
                     "not market-beating, not validated, not AGI."),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    log(f"wrote eval report -> {out_path}")
    _print_prereg(report)


def _print_prereg(report: dict) -> None:
    d = report["adapterPromptVsBasePrompt"]
    keys = ["tradition_merge_rate", "qualification_rate_on_contested", "false_attribution_rate", "citation_fidelity"]
    log("=== PRE-REGISTERED PRIMARY (adapter(prompt) - base(prompt); * = CI excludes 0) ===")
    any_improve = False
    for k in keys:
        v = d.get(k, {})
        star = "*" if v.get("improves") else " "
        if v.get("improves"):
            any_improve = True
        log(f"  {k:34s} Δ {v.get('delta')} ci={v.get('ci')} {star}")
    ab = report["adapter"]
    for g in ("protected_history_regression", "protected_religion_regression", "over_abstention_rate"):
        log(f"  adapter {g:34s} {ab['prompt_gate']['metrics'].get(g, {}).get('mean')}")
    log(f"PRIMARY habit-transfer signal present: {any_improve}")


def evaluate_retention(model_id: str, adapter_dir: Path, *, out_path: Path,
                       tasks_path: Path = ROOT / "data" / "generality_tasks.json",
                       answers_path: "Path | None" = None) -> None:
    """Catastrophic-forgetting check for criterion #3 (learning-shift STABILITY): score base vs
    adapter on the HELD-OUT generality probe (data/generality_tasks.json — abstraction/reasoning/
    analogy/out-of-domain, NO provenance tasks). Scored DETERMINISTICALLY (numeric/exact/regex)
    via tools/eval_generality.score — no LLM judge, can't be Goodharted. retains = the adapter
    does NOT drop more than 5pts vs base. This is the gemma-3-native stand-in for
    run_learning_shift's stability phase (that tool's adapter backend is vLLM/GLM, not gemma-3)."""
    import importlib.util as _ilu
    from peft import PeftModel
    _g = _ilu.module_from_spec(_ilu.spec_from_file_location("evg", ROOT / "tools" / "eval_generality.py"))
    _g.__spec__.loader.exec_module(_g)

    doc = _g.load_tasks(tasks_path)
    tasks = doc["tasks"]
    contam = _g.contamination_report(tasks, TRAIN)
    if contam:
        log(f"WARN: {len(contam)} generality prompt(s) leaked into train — retention number is suspect")
    sys_prompt = "You are a helpful, precise assistant. Answer concisely and follow the format requested."
    su = [(sys_prompt, t["prompt"]) for t in tasks]
    log(f"retention: {len(tasks)} held-out generality tasks x (base, adapter)")

    base, tok = load_base(model_id)
    base.eval()
    base_ans = _batched_generate(base, tok, su, max_new=96)

    log("loading adapter onto base for retention ...")
    adapted = PeftModel.from_pretrained(base, str(adapter_dir))
    adapted.eval()
    adapter_ans = _batched_generate(adapted, tok, su, max_new=96)

    def grade(answers):
        rows, correct = [], 0
        for t, a in zip(tasks, answers):
            cleaned = _g.strip_prompt_echo(a, t["prompt"])
            ok = _g.score(cleaned, t["answer"], t.get("match", "exact"))
            correct += int(ok)
            rows.append({"id": t["id"], "category": t.get("category"), "match": t.get("match"),
                         "gold": t["answer"], "reply": cleaned[:200], "correct": ok})
        return correct / len(tasks) if tasks else 0.0, rows

    base_acc, base_rows = grade(base_ans)
    adapter_acc, adapter_rows = grade(adapter_ans)
    delta = round(adapter_acc - base_acc, 4)
    retains = delta >= -0.05  # criterion #3: adapter stability >= base - 5pts (point estimate)

    # Paired bootstrap CI on the delta (the analysis flagged N is small -> report uncertainty).
    import random as _rnd
    _rnd.seed(0)
    bc = [int(r["correct"]) for r in base_rows]
    ac = [int(r["correct"]) for r in adapter_rows]
    N = len(bc)
    if N:
        boot = []
        for _ in range(4000):
            s = 0
            for _ in range(N):
                i = _rnd.randrange(N); s += ac[i] - bc[i]
            boot.append(s / N)
        boot.sort()
        delta_ci = [round(boot[int(0.025 * len(boot))], 4), round(boot[int(0.975 * len(boot))], 4)]
    else:
        delta_ci = [None, None]
    # CI-aware verdict: forgetting is ESTABLISHED only if the whole CI is below -0.05.
    retains_ci = (delta_ci[0] is None) or (delta_ci[1] >= -0.05)
    forgetting_established = (delta_ci[1] is not None) and (delta_ci[1] < -0.05)

    if answers_path is not None:
        answers_path.parent.mkdir(parents=True, exist_ok=True)
        answers_path.write_text(json.dumps(
            {"base": base_rows, "adapter": adapter_rows}, ensure_ascii=False, indent=2), encoding="utf-8")

    import collections
    def by_cat(rows):
        c = collections.Counter(); n = collections.Counter()
        for r in rows:
            n[r["category"]] += 1; c[r["category"]] += int(r["correct"])
        return {k: {"correct": c[k], "n": n[k], "acc": round(c[k] / n[k], 3)} for k in n}

    report = {
        "pilot": "sophia-wisdom-4b-m3-retention",
        "criterion": "learning-shift stability: adapter general-capability >= base - 5pts",
        "baseModel": model_id,
        "adapter": str(adapter_dir.relative_to(ROOT) if adapter_dir.is_relative_to(ROOT) else adapter_dir),
        "probe": str(tasks_path.relative_to(ROOT)), "nTasks": len(tasks),
        "scoring": "deterministic (numeric/exact/regex) — NO LLM judge",
        "contaminationLeaks": contam,
        "base_accuracy": round(base_acc, 4), "adapter_accuracy": round(adapter_acc, 4),
        "delta": delta, "delta_ci95": delta_ci, "retains": retains,
        "retains_ci": retains_ci, "forgetting_established": forgetting_established,
        "byCategoryBase": by_cat(base_rows), "byCategoryAdapter": by_cat(adapter_rows),
        "boundary": (f"Held-out generality probe (N={len(tasks)}); deterministic scoring; paired "
                     "bootstrap 95% CI on delta. 'retains' is the point-estimate criterion; "
                     "'forgetting_established' requires the whole CI below -0.05."),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    log(f"wrote retention report -> {out_path}")
    log(f"=== RETENTION (criterion #3): base {base_acc:.3f} -> adapter {adapter_acc:.3f} "
        f"(Δ {delta:+.3f} ci95={delta_ci}) retains={retains} forgetting_established={forgetting_established} ===")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--adapter", type=Path, default=DEFAULT_ADAPTER)
    ap.add_argument("--rows", type=Path, default=TRAIN)
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="cheap: load + 2 train steps + 1-case eval")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--out", type=Path, default=ROOT / "agi-proof" / "benchmark-results" / "wisdom-market" / "M3-pilot-eval.json")
    ap.add_argument("--save-answers", type=Path, default=None,
                    help="also write per-case base/adapter advisor answers here (for the LLM-judge pass)")
    ap.add_argument("--retention", action="store_true",
                    help="also score base vs adapter on the held-out generality probe (criterion #3)")
    ap.add_argument("--benchmark", type=Path, default=BENCH,
                    help="eval benchmark JSONL (default = heldout_v1; point at transfer_v1 for the "
                         "external-validity test on NOVEL entities)")
    # Stability knobs (anti-forgetting). Defaults reproduce the original M3 recipe.
    ap.add_argument("--lora-rank", type=int, default=16, help="LoRA rank (lower = less forgetting)")
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--use-rslora", action="store_true", help="rank-stabilized LoRA scaling")
    ap.add_argument("--kl-coef", type=float, default=0.0,
                    help="KL-anchor weight on general-retention rows (0 = off)")
    args = ap.parse_args()

    def _ret_paths():
        # retention-ONLY (no --eval): write to --out/--save-answers so the launcher stages the
        # exact files. With --eval: derive sibling names from the eval --out stem.
        if not args.eval:
            return args.out, args.save_answers
        base = args.out.name.replace("-eval", "").replace(".json", "")
        return (args.out.with_name(f"{base}-retention.json"),
                args.out.with_name(f"{base}-retention-answers.json"))

    if args.smoke:
        log("SMOKE: train 2 steps then eval 2 cases x1 run")
        train(args.model, args.adapter, rows_path=args.rows, seq_len=args.seq_len,
              epochs=1, seed=args.seed, lr=args.lr, smoke=True, lora_rank=args.lora_rank,
              lora_alpha=args.lora_alpha, use_rslora=args.use_rslora, kl_coef=args.kl_coef)
        evaluate(args.model, args.adapter, runs=1, limit=2, out_path=args.out.with_name("M3-pilot-smoke.json"),
                 bench_path=args.benchmark)
        if args.retention:
            evaluate_retention(args.model, args.adapter,
                               out_path=args.out.with_name("M3-pilot-retention-smoke.json"))
        log("SMOKE OK")
        return 0
    if args.train:
        train(args.model, args.adapter, rows_path=args.rows, seq_len=args.seq_len,
              epochs=args.epochs, seed=args.seed, lr=args.lr, smoke=False, lora_rank=args.lora_rank,
              lora_alpha=args.lora_alpha, use_rslora=args.use_rslora, kl_coef=args.kl_coef)
    if args.eval:
        evaluate(args.model, args.adapter, runs=args.runs, limit=args.limit, out_path=args.out,
                 answers_path=args.save_answers, bench_path=args.benchmark)
    if args.retention:
        ret_out, ret_ans = _ret_paths()
        evaluate_retention(args.model, args.adapter, out_path=ret_out, answers_path=ret_ans)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
