# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Rollout-driven GRPO for the retrieval-faithfulness reward.

The vanilla TRL ``GRPOTrainer`` reward callback only sees completion text, but the
faithfulness reward needs a full trajectory (retrieval context + the counterfactual
citation-drop regeneration). So the SAMPLING unit here is a whole
``faithfulness_rollout.rollout`` — for each case we sample a GROUP of G rollouts,
score each with ``retrieval_faithfulness.reward_for_trajectory``, compute the GRPO
group-relative advantage, and apply it as the policy-gradient weight on each
rollout's *answer* tokens (no value network — the group mean is the baseline, à la
DeepSeek-R1 GRPO). The ablation regeneration is reward-only (no_grad); we never
train on the ablated answer.

Two layers, separated so the math is CI-testable without a GPU:

  PURE (asserted offline, no torch):
    * group_advantages / within_group_std — the value-free baseline.
    * grpo_objective — the per-rollout loss + its analytic gradient wrt the answer
      log-prob, incl. the k3 KL-to-reference estimator. This is the SAME formula the
      torch loop minimizes, so the test validates the real objective. Headline
      invariant: a faithful rollout (positive advantage) gets a gradient that
      INCREASES its answer's log-prob; a leaky one's DECREASES — i.e. the
      faithfulness reward trains the policy toward retrieval-grounded answers, and
      it does so on all-correct groups where a correctness-only reward collapses.

  TORCH (structure-validated; runs on a CUDA GPU, not in CI):
    * TorchPolicy — sample answers + recompute grad-enabled answer log-probs.
    * train / run_live — the live loop: load LoRA policy, sample groups, advantage,
      objective, step. Reward seams are live retrieval (agent.ai_search) + a
      deterministic lexical entailment placeholder so the reward is computable
      on-box; a real entailment LLM is the Open upgrade (failure-ledger).

See agi-proof/reasoning-core-design.md. candidate_only; canClaimAGI:false — this
ships a runnable loop, not a measured uplift (that needs a pre-registered gated run).
"""

from __future__ import annotations

import math
import statistics
from typing import Any, Callable


# --------------------------------------------------------------------------- #
# PURE loop math (no torch) — group baseline + objective + its gradient.
# --------------------------------------------------------------------------- #

def within_group_std(rewards: list) -> float:
    """Population std of a group's rewards — the GRPO learning signal. Zero means
    every rollout got the same reward (advantage collapses to zero)."""
    if len(rewards) < 2:
        return 0.0
    return statistics.pstdev(rewards)


def group_advantages(rewards: list, *, eps: float = 1e-6) -> list:
    """GRPO group-relative, value-free advantage: ``(r - mean) / std``. A collapsed
    group (std <= eps) yields all-zero advantages (no gradient), which is exactly the
    failure mode the faithfulness term is designed to avoid on all-correct groups."""
    if not rewards:
        return []
    mean = statistics.fmean(rewards)
    sd = within_group_std(rewards)
    if sd <= eps:
        return [0.0] * len(rewards)
    return [(r - mean) / sd for r in rewards]


def _k3_kl(ref_lp: float, pol_lp: float) -> float:
    """Schulman k3 KL estimator on a (mean) log-prob pair: ``exp(r) - r - 1`` with
    ``r = ref_lp - pol_lp``. Always >= 0; 0 iff the policy matches the reference."""
    r = ref_lp - pol_lp
    return math.exp(r) - r - 1.0


def grpo_objective(
    advantages: list,
    mean_logprobs: list,
    *,
    ref_mean_logprobs: list | None = None,
    beta: float = 0.0,
) -> dict:
    """The GRPO loss over one group and its analytic gradient wrt each rollout's mean
    answer log-prob. Pure (operates on floats) so it is unit-testable; the torch loop
    minimizes the identical expression on tensors.

    Per rollout i (N = group size):
        pg_i   = -adv_i * lp_i
        kl_i   = beta * k3(ref_lp_i, lp_i)            (0 if no reference)
        loss   = mean_i (pg_i + kl_i)
        d loss/d lp_i = (1/N) * (-adv_i + beta * (1 - exp(ref_lp_i - lp_i)))

    So with positive advantage the gradient is negative — gradient DESCENT raises the
    answer's log-prob (the policy is pushed toward that rollout); negative advantage
    lowers it. The KL term pulls back toward the reference."""
    n = len(advantages)
    if n == 0:
        return {"loss": 0.0, "grad_logprob": [], "pg": [], "kl": []}
    refs = ref_mean_logprobs if ref_mean_logprobs is not None else [None] * n
    pg, kl, grad = [], [], []
    for adv, lp, ref in zip(advantages, mean_logprobs, refs):
        pg_i = -adv * lp
        if ref is not None and beta:
            kl_i = beta * _k3_kl(ref, lp)
            grad_i = (-adv + beta * (1.0 - math.exp(ref - lp))) / n
        else:
            kl_i = 0.0
            grad_i = (-adv) / n
        pg.append(pg_i)
        kl.append(kl_i)
        grad.append(grad_i)
    loss = sum(p + k for p, k in zip(pg, kl)) / n
    return {"loss": loss, "grad_logprob": grad, "pg": pg, "kl": kl}


def sample_group(
    case: dict,
    policies: list,
    *,
    seams: dict,
    reward_fn: Callable | None = None,
) -> list:
    """Roll out one GROUP for ``case`` (one rollout per policy in ``policies``) and
    return ``[{traj, reward, detail}, ...]``.

    Offline, the group's diversity comes from passing distinct deterministic
    ``policies``; live, it is ``[stochastic_policy] * G`` sampled at temperature. ``seams``
    supplies ``retrieve`` / ``extract_claims`` / ``verify_claim`` (+ optional
    ``check_correct``); ``reward_fn`` defaults to the faithfulness reward."""
    from provenance_bench.faithfulness_rollout import rollout
    from provenance_bench.retrieval_faithfulness import reward_for_trajectory

    rf = reward_fn or reward_for_trajectory
    out = []
    for policy in policies:
        traj = rollout(case, generate=policy, **seams)
        reward, detail = rf(traj)
        out.append({"traj": traj, "reward": reward, "detail": detail})
    return out


def offline_invariants() -> tuple[bool, dict]:
    """Assert the GRPO advantage math, the anti-collapse property, AND the objective's
    gradient direction (no torch/GPU).

    Builds a group of four all-CORRECT rollouts — two retrieval-using (faithful), two
    weights-leaking — on an identical answer, then shows: a correctness-only reward
    collapses (zero advantage), the faithfulness reward keeps a learning signal, and
    the GRPO objective's gradient RAISES the faithful answers' log-prob while LOWERING
    the leaky ones'."""
    from provenance_bench import faithfulness_rollout as fr

    case = {"prompt": "Who wrote the Project Phoenix Charter?",
            "should_retrieve": True, "answerable": True, "gold": "founding committee"}
    seams = dict(retrieve=fr._mock_retrieve, extract_claims=fr._mock_extract,
                 verify_claim=fr._mock_verify, check_correct=fr._check_correct)
    policies = [fr._faithful_policy, fr._faithful_policy, fr._leaky_policy, fr._leaky_policy]

    group = sample_group(case, policies, seams=seams)
    rewards = [g["reward"] for g in group]
    adv = group_advantages(rewards)
    correctness_only = [1.0 if g["traj"].get("task_correct") else 0.0 for g in group]

    # Gradient direction under the objective (uniform synthetic log-probs + a slightly
    # higher reference so the KL term is exercised and non-zero).
    lp = [-1.0, -1.0, -1.0, -1.0]
    ref = [-0.9, -0.9, -0.9, -0.9]
    obj = grpo_objective(adv, lp, ref_mean_logprobs=ref, beta=0.02)
    grad = obj["grad_logprob"]

    checks = {
        "allRolloutsCorrect": all(c == 1.0 for c in correctness_only),
        "correctnessOnlyCollapses": within_group_std(correctness_only) == 0.0,
        "faithfulnessGivesSignal": within_group_std(rewards) > 0.0,
        "advantagesSumZero": abs(sum(adv)) < 1e-6,
        "faithfulPositiveLeakyNegative": adv[0] > 0.0 > adv[2],
        "collapsedGroupZeroAdvantage": group_advantages([0.5, 0.5, 0.5]) == [0.0, 0.0, 0.0],
        # gradient DESCENT (lp -= grad) raises faithful log-prob (grad<0), lowers leaky (grad>0).
        "gradientRaisesFaithful": grad[0] < 0.0,
        "gradientLowersLeaky": grad[2] > 0.0,
        "klNonNegative": all(k >= 0.0 for k in obj["kl"]),
    }
    detail = {
        "checks": checks,
        "rewards": [round(r, 4) for r in rewards],
        "advantages": [round(a, 4) for a in adv],
        "gradLogprob": [round(g, 4) for g in grad],
        "withinGroupStd": round(within_group_std(rewards), 4),
        "correctnessOnlyStd": within_group_std(correctness_only),
        "loss": round(obj["loss"], 4),
        "note": "correctness-only collapses to zero advantage; faithfulness separates "
                "retrieval-using from weights-leaking rollouts and the objective gradient "
                "moves the policy toward the faithful answers.",
    }
    return all(checks.values()), detail


# --------------------------------------------------------------------------- #
# TORCH loop (structure-validated; CUDA GPU, not CI). Lazy imports throughout.
# --------------------------------------------------------------------------- #

def _build_prompt(query: str, context_chunks: list) -> str:
    """Format the retrieved context + question into a single grounding prompt. The
    instruction is the design's core rule: answer ONLY from the sources, else abstain."""
    src = "\n".join(f"[{c['chunk_id']}] {c['text']}" for c in context_chunks) or "(no sources)"
    return (
        "Answer the question using ONLY the sources below. Cite the source id you use. "
        "If the sources do not support an answer, say you cannot answer.\n\n"
        f"Sources:\n{src}\n\nQuestion: {query}\nAnswer:"
    )


class TorchPolicy:
    """Wraps a HF causal-LM + tokenizer as a rollout ``generate`` seam that also exposes
    grad-enabled answer log-probs. Records each (prompt, answer) call so the loop can
    take the MAIN answer (call 0) for the gradient and ignore the ablation regens."""

    def __init__(self, model: Any, tokenizer: Any, *, max_new_tokens: int = 96,
                 temperature: float = 0.9):
        self.model = model
        self.tok = tokenizer
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.calls: list[tuple[str, str]] = []

    def generate(self, query: str, context_chunks: list) -> str:
        import torch

        prompt = _build_prompt(query, context_chunks)
        ids = self.tok(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(
                **ids, max_new_tokens=self.max_new_tokens, do_sample=True,
                temperature=self.temperature, top_p=0.95,
                pad_token_id=self.tok.pad_token_id,
            )
        answer = self.tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        self.calls.append((prompt, answer))
        return answer

    def answer_logprob(self, prompt: str, answer: str, *, use_reference: bool = False):
        """Mean log-prob of the answer tokens given the prompt. ``use_reference`` runs
        with the LoRA adapter disabled (the frozen base = KL reference) and detaches."""
        import torch

        full = self.tok(prompt + " " + answer, return_tensors="pt").to(self.model.device)
        plen = self.tok(prompt, return_tensors="pt")["input_ids"].shape[1]
        ctx = self.model.disable_adapter() if use_reference else _nullctx()
        with ctx:
            with (torch.no_grad() if use_reference else _nullctx()):
                logits = self.model(**full).logits[:, :-1, :]
                logprobs = torch.log_softmax(logits, dim=-1)
                targets = full["input_ids"][:, 1:]
                tok_lp = logprobs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
                ans_lp = tok_lp[:, plen - 1:]
                val = ans_lp.mean()
        return val.detach() if use_reference else val


class _nullctx:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False


def cases_from_rl_dataset(*, eval_frac: float = 0.2, seed: int = 0, limit: int | None = None) -> list:
    """Map the provenance RL dataset into faithfulness cases (prompt + gold author +
    answerability). Contested attributions are answerable-from-wiki and worth retrieving."""
    from provenance_bench import rl_dataset

    data = rl_dataset.build_rl_dataset(eval_frac=eval_frac, seed=seed)
    cases = []
    for row in data["train_rows"]:
        cases.append({
            "prompt": row["prompt"],
            "should_retrieve": True,
            "answerable": True,
            "gold": row.get("gold_author", ""),
        })
    return cases[:limit] if limit else cases


def train(args: Any) -> int:
    """Live GRPO loop on a CUDA GPU. Structure-validated here; the offline invariants
    assert the objective/advantage math this minimizes."""
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from provenance_bench import faithfulness_seams as seams
    from provenance_bench.retrieval_faithfulness import reward_for_trajectory

    if not torch.cuda.is_available():
        print("CUDA GPU required for the live faithfulness GRPO loop (use --model mock "
              "for the offline invariants).")
        return 1

    model_name = getattr(args, "model", "Qwen/Qwen2.5-7B-Instruct")
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16).cuda()

    # LoRA target modules by family (GLM fused gate_up vs split gate/up), matching run_rlvr.
    from tools.run_rlvr import resolve_target_modules

    peft_cfg = LoraConfig(
        r=getattr(args, "lora_r", 16), lora_alpha=getattr(args, "lora_alpha", 32),
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        target_modules=resolve_target_modules(model_name, getattr(args, "lora_target_modules", None)),
    )
    model = get_peft_model(base, peft_cfg)
    model.train()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=getattr(args, "lr", 1e-5))

    policy = TorchPolicy(model, tok)
    # Verify seam: a REAL entailment LLM when a provider is configured (keys in the
    # gitignored private/secrets/), else the deterministic lexical placeholder. Network
    # failures fail closed to "irrelevant" inside the adapter, so a flaky API never
    # crashes the rollout.
    entail_provider = getattr(args, "entailment_provider", None)
    if entail_provider:
        entailment = seams.entailment_from_provider(
            entail_provider, model=getattr(args, "entailment_model", None))
        print(f"verify seam: live entailment via {entail_provider}")
    else:
        entailment = seams.lexical_entailment
        print("verify seam: lexical placeholder (no --entailment-provider given)")
    seams_d = dict(
        retrieve=seams.make_ai_search_retrieve(top_k=getattr(args, "top_k", 6)),
        extract_claims=seams.heuristic_extract_claims,
        verify_claim=seams.make_entailment_verify(entailment),
        check_correct=lambda a, g: bool(g) and str(g).lower() in a.lower(),
    )
    cases = cases_from_rl_dataset(seed=getattr(args, "seed", 0),
                                  limit=getattr(args, "limit", None))
    from provenance_bench.faithfulness_rollout import rollout

    g = getattr(args, "num_generations", 8)
    beta = getattr(args, "beta", 0.02)
    max_steps = getattr(args, "max_steps", 0) or len(cases)
    std_log = []

    for step in range(max_steps):
        case = cases[step % len(cases)]
        group = []
        for _ in range(g):
            policy.calls = []
            traj = rollout(case, generate=policy.generate, **seams_d)
            reward, _ = reward_for_trajectory(traj)
            main = policy.calls[0] if policy.calls else (_build_prompt(case["prompt"], []), "")
            group.append({"reward": reward, "prompt": main[0], "answer": main[1]})

        rewards = [m["reward"] for m in group]
        adv = group_advantages(rewards)
        std_log.append(within_group_std(rewards))
        if all(a == 0.0 for a in adv):
            continue  # collapsed group: no gradient this step

        opt.zero_grad()
        losses = []
        for m, a in zip(group, adv):
            lp = policy.answer_logprob(m["prompt"], m["answer"])
            ref_lp = policy.answer_logprob(m["prompt"], m["answer"], use_reference=True)
            kl = torch.exp(ref_lp - lp) - (ref_lp - lp) - 1.0
            losses.append(-a * lp + beta * kl)
        loss = torch.stack(losses).mean()
        loss.backward()
        opt.step()
        if step % 5 == 0:
            print(f"step {step} loss {loss.item():.4f} meanReward {sum(rewards)/len(rewards):.3f} "
                  f"groupStd {std_log[-1]:.3f}")

    out_dir = str(getattr(args, "output", "training/faithfulness/checkpoints/sophia-faithful-v1"))
    model.save_pretrained(out_dir)
    print(f"Saved adapter -> {out_dir}. Held-out faithfulness eval + gating is a separate, "
          f"pre-registered step (canClaimAGI:false; no uplift claimed here).")
    return 0


def run_live(args: Any) -> int:
    """Entry point for the GPU path. Requires the RL deps; delegates to ``train``."""
    try:
        import torch  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        print(f"Install RL deps: pip install -r requirements-rl.txt ({type(exc).__name__}: {exc})")
        return 1
    return train(args)


__all__ = [
    "within_group_std", "group_advantages", "grpo_objective", "sample_group",
    "offline_invariants", "TorchPolicy", "cases_from_rl_dataset", "train", "run_live",
]
