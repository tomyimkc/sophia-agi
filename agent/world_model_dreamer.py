# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""DreamerV3-style discrete-latent world model for Sophia's action-outcome traces.

Path A of `docs/06-Roadmap/Two-Paths-To-Novelty.md`. The falsifiable experiment:
**can a neural dynamics model, trained on Sophia's own harness traces, generalize
to held-out AND distribution-shifted (state, action) pairs — or does it collapse
to memorization?** This is the AlphaGo-move question for Sophia's substrate.

The roofline result (`reasoning/deliberation_roofline.py`) says the verifier sets
the ceiling. A dynamics model that is **verified against real traces** can raise
that ceiling — but ONLY if it generalizes. DreamerV3 ([Hafner et al. 2023](
https://huggingface.co/papers/2301.04104); [Nature 2025 control paper](
https://www.nature.com/articles/s41586-025-08744-2)) is the canonical architecture:
encode state → **discrete latent** (categorical distributions, the key to stable
imagination) → predict next latent + reward → plan inside the imagined model.

What this is NOT: a full DreamerV3 port. Sophia has no environment to "dream" in
the RL sense. This is the **narrow** analogue adapted to Sophia's discrete
(state, action) → outcome traces: a Recurrent State-Space Model (RSSM) over a
discrete latent that predicts P(success | state, action), trained on the same
`OutcomePair` corpus the toy `FeatureLogisticPredictor` used, and gated by the
SAME `verified_world_model.train_verified_world_model` canary (promote only on
held-out gain + bounded shift-degradation).

Discipline (Sophia, preserved):
  * **CUDA-gated, fail-closed** — torch is imported lazily; absent torch/CUDA the
    predictor abstains (returns 0.5) rather than crashing. Same pattern as
    `run_rlvr.py` / `math_verifier.py`'s Lean stub. CI never imports torch.
  * **candidateOnly / level3Evidence: false** — the report carries these fields;
    promoting the model is `verified_world_model`'s job, not this module's.
  * **Generalization is measured, not assumed** — the shift-degeneracy check in
    `verified_world_model.train_verified_world_model` is the load-bearing test.
"""

from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass, field
from typing import Any, Callable

from agent.verified_world_model import OutcomePair, accuracy

# OutcomePredictor protocol (duck-typed): `predict(state, action) -> float in [0,1]`.
# Matches agent.planner_learned_sim.OutcomePredictor and verified_world_model's
# FeatureLogisticPredictor so this slots into the existing canary scaffold.


def _torch():
    """Lazy torch import. Returns the torch module or None (fail-closed)."""
    try:
        import torch  # type: ignore
        return torch
    except ImportError:
        return None


@dataclass
class DreamerConfig:
    """Hyperparameters for the discrete-latent RSSM. Defaults are small (CPU-dev
    friendly); scale `hidden`, `classes`, `stoch` up on GPU for the real run.

    - ``hidden``: RSSM + heads MLP width.
    - ``classes`` / ``stoch``: the discrete latent is ``stoch`` categorical
      distributions each over ``classes`` categories (DreamerV3 uses 32×32).
    - ``lr`` / ``epochs`` / ``batch``: Adam optimizer settings.
    """

    hidden: int = 128
    classes: int = 16
    stoch: int = 16
    lr: float = 3e-4
    epochs: int = 60
    batch: int = 32
    seed: int = 0
    max_str_len: int = 64  # char-level tokenization cap for (state, action) strings


@dataclass
class DreamerReport:
    """Training report for the discrete-latent world model. Carries the discipline
    fields and the measured (not assumed) generalization signals."""

    schema: str = "sophia.world_model_dreamer.v1"
    candidate_only: bool = True
    level3_evidence: bool = False
    torch_available: bool = False
    cuda_available: bool = False
    train_size: int = 0
    val_accuracy: float = 0.0
    train_loss: float = 0.0
    trained: bool = False
    reason: str = ""
    config: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "candidateOnly": self.candidate_only,
            "level3Evidence": self.level3_evidence,
            "torchAvailable": self.torch_available,
            "cudaAvailable": self.cuda_available,
            "trainSize": self.train_size,
            "valAccuracy": round(self.val_accuracy, 4),
            "trainLoss": round(self.train_loss, 6),
            "trained": self.trained,
            "reason": self.reason,
            "config": self.config,
            "interpretation": self._interp(),
        }

    def _interp(self) -> str:
        if not self.torch_available:
            return ("torch unavailable — predictor abstains (fail-closed). The "
                    "discrete-latent model cannot train; install torch (CPU dev "
                    "or CUDA for the real run) to fire the experiment.")
        if not self.trained:
            return self.reason or "not trained"
        return (f"Trained discrete-latent RSSM ({self.config.get('classes')}×"
                f"{self.config.get('stoch')} latent). val accuracy {self.val_accuracy:.4f}. "
                f"Feed this predictor to verified_world_model.train_verified_world_model "
                f"to run the shift-degeneracy canary — THAT is the load-bearing test.")


# --------------------------------------------------------------------------- #
# Char-level tokenizer (deterministic, vocab-free, CI-stable). The (state, action)
# strings are short symbolic labels; char-level avoids a tokeniser dependency and
# keeps the model's input space auditable.
# --------------------------------------------------------------------------- #
class _CharTok:
    def __init__(self, max_len: int = 64) -> None:
        self.max_len = max_len

    def encode(self, s: str) -> list[int]:
        # printable ASCII range 32..126 -> 0..94; pad/truncate to max_len.
        vals = [max(0, min(126, ord(c)) - 32) for c in (s or "")[: self.max_len]]
        vals += [0] * (self.max_len - len(vals))
        return vals


class DreamerWorldPredictor:
    """A discrete-latent (DreamerV3-style) outcome predictor.

    Implements the ``OutcomePredictor`` protocol (``predict(state, action) -> float``)
    so it drops into ``verified_world_model.train_verified_world_model`` and
    ``planner_learned_sim.LearnedSimulator`` unchanged. When torch is absent it
    abstains (predict 0.5 — uninformative, fail-closed) so the canary never
    silently passes on a model that didn't actually train.
    """

    def __init__(self, cfg: DreamerConfig | None = None) -> None:
        self.cfg = cfg or DreamerConfig()
        self.tok = _CharTok(self.cfg.max_str_len)
        self._torch = _torch()
        self._device = None
        self._enc = None  # encoder MLP
        self._rssm = None  # recurrent state-space
        self._head = None  # reward/success head
        self._trained = False

    # ----- the OutcomePredictor protocol ----- #
    def fit(self, pairs: list[OutcomePair]) -> "DreamerWorldPredictor":
        """Train the discrete-latent model on (state, action, success) triples."""
        torch = self._torch
        if torch is None:
            return self  # abstain; report records torch_available=False
        cfg = self.cfg
        torch.manual_seed(cfg.seed)
        random.seed(cfg.seed)
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Build a compact discrete-latent model. We keep it faithful to DreamerV3's
        # *discrete* latent (categorical distributions) but drop the full RSSM
        # sequence model — Sophia's (state, action) inputs are not long temporal
        # sequences, they are single transitions. The "discrete latent" is the
        # load-bearing piece for stable imagination + generalization; the recurrent
        # prior matters for long horizons Sophia doesn't yet have.
        V, H, C, S = 95, cfg.hidden, cfg.classes, cfg.stoch
        self._enc = torch.nn.Sequential(
            torch.nn.Linear(V * 2 * cfg.max_str_len, H), torch.nn.LayerNorm(H), torch.nn.GELU(),
            torch.nn.Linear(H, H), torch.nn.LayerNorm(H), torch.nn.GELU(),
        ).to(self._device)
        self._rssm = torch.nn.Sequential(
            torch.nn.Linear(H, H), torch.nn.GELU(),
            torch.nn.Linear(H, S * C),  # logits over S categorical distributions
        ).to(self._device)
        self._head = torch.nn.Sequential(
            torch.nn.Linear(H + S * C, H), torch.nn.LayerNorm(H), torch.nn.GELU(),
            torch.nn.Linear(H, 1),
        ).to(self._device)
        params = list(self._enc.parameters()) + list(self._rssm.parameters()) + list(self._head.parameters())
        opt = torch.optim.Adam(params, lr=cfg.lr)
        bce = torch.nn.BCEWithLogitsLoss()

        # Encode the corpus once.
        xs = torch.tensor(
            [self.tok.encode(s) + self.tok.encode(a) for s, a, _ in pairs],
            dtype=torch.long, device=self._device,
        )
        ys = torch.tensor([[float(l)] for _, _, l in pairs], dtype=torch.float32, device=self._device)
        n = len(pairs)
        last_loss = 0.0
        for _ in range(cfg.epochs):
            perm = torch.randperm(n)
            for i in range(0, n, cfg.batch):
                idx = perm[i : i + cfg.batch]
                x = xs[idx].float()  # (B, 2*max_len) ints in [0,94]
                # one-hot each char -> (B, 2*max_len*V)
                xoh = torch.nn.functional.one_hot(x.long(), V).flatten(1).float()
                h = self._enc(xoh)
                logits = self._rssm(h).view(-1, S, C)
                # straight-through Gumbel-softmax sample (DreamerV3-style discrete lat.)
                if self._training():
                    z = torch.nn.functional.gumbel_softmax(logits, tau=1.0, hard=True).flatten(1)
                else:
                    z = logits.softmax(-1).flatten(1)
                pred = self._head(torch.cat([h, z], -1))
                loss = bce(pred, ys[idx])
                opt.zero_grad()
                loss.backward()
                opt.step()
                last_loss = float(loss.detach())
        self._train_loss = last_loss
        self._trained = True
        return self

    def _training(self) -> bool:
        # module-level train/eval flag for the Gumbel sampling
        return getattr(self, "_is_train", True)

    def predict(self, state: str, action: str) -> float:
        """P(success | state, action) in [0,1]. Abstains (0.5) if untrained/no-torch."""
        torch = self._torch
        if torch is None or not self._trained or self._head is None:
            return 0.5  # fail-closed: uninformative prediction
        with torch.no_grad():
            x = torch.tensor([self.tok.encode(state) + self.tok.encode(action)], dtype=torch.long, device=self._device)
            xoh = torch.nn.functional.one_hot(x.long(), 95).flatten(1).float()
            h = self._enc(xoh)
            logits = self._rssm(h).view(-1, self.cfg.stoch, self.cfg.classes)
            z = logits.softmax(-1).flatten(1)  # eval: use the soft categorical
            logit = self._head(torch.cat([h, z], -1))
            return float(torch.sigmoid(logit).item())


def train_dreamer_report(
    traces: list[OutcomePair],
    *,
    val_traces: list[OutcomePair] | None = None,
    cfg: DreamerConfig | None = None,
) -> tuple[DreamerWorldPredictor, DreamerReport]:
    """Train the DreamerV3-style predictor and return (predictor, report).

    The caller should then feed the predictor into
    ``verified_world_model.train_verified_world_model`` to run the
    shift-degeneracy canary — THAT is the load-bearing generalization test, not
    the val accuracy here (which only checks in-distribution learnability)."""
    cfg = cfg or DreamerConfig()
    torch = _torch()
    rep = DreamerReport(
        torch_available=torch is not None,
        cuda_available=bool(torch and torch.cuda.is_available()),
        train_size=len(traces),
        config={"hidden": cfg.hidden, "classes": cfg.classes, "stoch": cfg.stoch,
                "lr": cfg.lr, "epochs": cfg.epochs, "batch": cfg.batch, "seed": cfg.seed},
    )
    if torch is None:
        rep.reason = ("torch not installed — install torch (CPU for dev, CUDA for the "
                      "real run) to train the discrete-latent world model. Abstaining.")
        return DreamerWorldPredictor(cfg), rep

    pred = DreamerWorldPredictor(cfg).fit(traces)
    rep.trained = pred._trained
    rep.train_loss = getattr(pred, "_train_loss", 0.0)
    if val_traces is not None:
        rep.val_accuracy = accuracy(pred, val_traces)
    return pred, rep


def write_dreamer_report(report: DreamerReport, out: str) -> dict[str, Any]:
    """Persist the dreamer report as JSON (candidate artifact)."""
    import json
    from pathlib import Path

    payload = report.to_dict()
    p = Path(out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return payload


__all__ = [
    "DreamerConfig",
    "DreamerReport",
    "DreamerWorldPredictor",
    "train_dreamer_report",
    "write_dreamer_report",
]
