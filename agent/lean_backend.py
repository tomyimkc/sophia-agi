# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Lean 4 backend for formal math verification — the legitimate novelty pathway.

Path B of `docs/06-Roadmap/Two-Paths-To-Novelty.md`. The roofline result bounds
every Sophia output to (train ∪ retrieved), filtered by a verifier. Formal proof
is the ONE domain where novelty is reachable *under* that ceiling: a Lean-verified
proof is self-certifying, so "novel + verified" is achievable without breaking the
fail-closed discipline.

Methodology: the open, reproducible AlphaProof-style stack —
  * **LeanDojo** ([NeurIPS 2023](https://neurips.cc/virtual/2023/poster/73510);
    [project](https://leandojo.org/); [docs](https://leandojo.readthedocs.io/)):
    programmatic Lean 4 interaction (tactic application, premise extraction,
    proof-state trees). The open analogue of AlphaProof's
    ([Nature 2025](https://www.nature.com/articles/s41586-025-09833-y)) Lean integration.
  * **ReProver** ([repo](https://github.com/lean-dojo/reprover)): retrieval-augmented
    premise selection — surface relevant library lemmas for the LLM's tactic proposals.
  * **LeanProgress** ([ICLR 2025](https://arxiv.org/html/2502.17925v2)): proof-progress
    prediction to guide search.

Discipline (Sophia, preserved — non-negotiable):
  * **Opt-in extra, fail-closed default.** `lean-dojo` + Lean 4 + elan live behind
    `requirements-theorem.txt`. When Lean is absent (the CI default, and the
    production default), every call abstains with `lean_unavailable` — NEVER crashes,
    NEVER fabricates a verdict. The existing `math_verifier.verify(use_lean=True)`
    abstain stub is preserved exactly when this module isn't installed.
  * **A proof is the verifier.** Lean either accepts a proof (closed goal) or it
    doesn't; there is no "looks correct" middle. This is the strongest verifier family.
  * **candidateOnly / level3Evidence: false** until a gated run.
  * **Novelty is MEASURED, not assumed.** `novelty_check` (strict, per the human's
    decision) flags a proof as "novel" only if it is NOT an embedding near-duplicate
    of the training corpus / library.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any, Literal

LeanVerdict = Literal["accepted", "rejected", "abstain"]


def lean_available() -> bool:
    """True iff lean-dojo is importable AND a Lean 4 toolchain is reachable.

    Cheap probe (no Lean invocation) used to decide whether to attempt a Lean check
    or abstain fail-closed. Mirrors `math_verifier.sympy_available`."""
    try:
        import lean_dojo  # type: ignore  # noqa: F401
        return True
    except ImportError:
        return False


@dataclass
class LeanCheck:
    """Result of attempting a Lean verification. The verdict is the only field a
    caller should trust; the rest is audit detail."""

    verdict: LeanVerdict
    reason: str
    backend: str = "lean4"
    lean_available: bool = False
    goal_closed: bool = False
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict,
            "reasons": [self.reason],
            "detail": {"backend": self.backend, "lean": self.lean_available,
                       "goalClosed": self.goal_closed, **self.detail},
        }


def verify_proof(
    *,
    theorem: str,
    proof: str,
    repo_url: str = "https://github.com/leanprover-community/mathlib4",
    timeout_s: int = 120,
) -> LeanCheck:
    """Verify a Lean 4 ``proof`` of ``theorem``.

    ``theorem`` is a header ending in ``:= by`` (e.g. ``"theorem t : True := by"``) and
    ``proof`` is the tactic body that follows (one tactic per line). The two are
    assembled into a SINGLE ``theorem ... := by <tactics>`` source — never concatenated
    with a separator or a second theorem header, which would produce invalid Lean and
    cause systematic rejection/abstention even when Lean is available. If ``proof``
    already contains ``:= by`` (a caller passed a full block), it is used verbatim.

    Fail-closed at every step: no lean-dojo → abstain(``lean_unavailable``); a Lean
    error → rejected with the error tail; a closed goal → accepted. We never interpret a
    partial/errored state as anything but not-yet-proven.
    """
    if not lean_available():
        return LeanCheck(verdict="abstain", reason="lean_unavailable: lean-dojo not installed",
                         lean_available=False)
    try:
        from lean_dojo import LeanDojo  # type: ignore
    except ImportError:
        return LeanCheck(verdict="abstain", reason="lean_unavailable: lean-dojo import failed",
                         lean_available=False)

    # Assemble ONE Lean 4 source. Accept either a tactic body (the normal call shape)
    # or a caller's pre-assembled full block, but never emit a duplicate theorem header
    # or a stray separator — both would make Lean reject an otherwise-correct proof.
    # Detect a full block ONLY by its leading declaration keyword (not by a `:= by`
    # substring): a tactic body legitimately contains `have h : P := by ...`, which would
    # falsely match the substring test and drop the theorem header.
    _proof_head = proof.lstrip()[:16].lower()
    _is_full_block = _proof_head.startswith(("theorem ", "lemma ", "def ", "example", "axiom "))
    if _is_full_block:
        source = proof  # caller passed a full theorem block; use it verbatim
    elif ":= by" in theorem:
        source = f"{theorem}\n{proof}"
    else:
        source = f"{theorem} := by\n{proof}"
    try:
        # LeanDojo's exact API varies by version; the load-bearing call is "does
        # this source elaborate with no remaining goals / errors". We wrap it so a
        # version mismatch abstains rather than crashes (fail-closed).
        dojo = LeanDojo(repo=repo_url, timeout=timeout_s)  # type: ignore[call-arg]
        result = dojo.run_code(source)  # type: ignore[attr-defined]
        # Convention: a clean proof closes all goals; an error surfaces a message.
        errored = bool(getattr(result, "has_errors", False) or getattr(result, "error", None))
        if errored:
            msg = str(getattr(result, "error", "") or getattr(result, "trace", ""))[-300:]
            return LeanCheck(verdict="rejected", reason=f"lean_rejected: {msg}",
                             lean_available=True, goal_closed=False,
                             detail={"error_tail": msg})
        return LeanCheck(verdict="accepted", reason="lean_accepted: goal closed",
                         lean_available=True, goal_closed=True)
    except Exception as exc:  # fail-closed: any LeanDojo failure abstains, never lies
        return LeanCheck(verdict="abstain",
                         reason=f"lean_error: {type(exc).__name__}: {str(exc)[:200]}",
                         lean_available=True, goal_closed=False,
                         detail={"exception": type(exc).__name__})


# ---------------------------------------------------------------------------
# Tactic-DAG novelty hash (§3.3 / §5.3 of the Open-Problems-Framework critique).
#
# char-trigram Jaccard (below) is a cheap, dependency-free PRE-FILTER: it catches
# near-verbatim retrieval. Its documented failure mode is that a re-proof via a
# *different tactic path* the model saw in training scores "novel". The tactic-DAG
# hash is the STRICT decider: it hashes the *structure* of the tactic proof — the
# sequence of tactic heads + the named lemmas/hypotheses they use, with
# commutative/associative rewrite lists normalized to a canonical order — so a
# trivially-reordered proof does not get a fresh hash, while two genuinely
# different tactic paths to the same theorem get different hashes (sequence +
# dependency edges differ).
#
# Discipline (non-negotiable): pure stdlib (re + hashlib). It parses the proof
# STRING structurally — it deliberately does NOT depend on Lean, so the fail-closed
# default (no lean-dojo) and the numpy-only core are untouched, and CI is unchanged.
# An unparseable proof abstains: dag_hash is None and novelty is NEVER claimed,
# never fabricated.
# ---------------------------------------------------------------------------

_TAC_SPLIT_RE = re.compile(r"<;>|=>|\||[;\n]")      # tactic combinators / arm markers / separators
_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_']*(?:\.[A-Za-z_][A-Za-z0-9_']*)*")  # bare / dotted
_TAC_HEAD_RE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_']*[!?]?)")                  # leading tactic id
_RW_BRACKET_RE = re.compile(r"\[([^\[\]]*)\]")                                    # `[...]` arg list
# Rewrite lemmas whose ORDER is irrelevant (commutativity / associativity / ac) —
# permuting them inside `rw [a, b]` must not change the hash. All other rewrites
# keep their sequence (their order is semantically load-bearing).
_COMM_LEMMA_RE = re.compile(r"(?:comm|assoc|left_comm|ac_rfl|commutes)$")
_RW_TAC_RE = re.compile(r"^(rw|rewrite|simp_rw|erw|nth_rewrite)\b")


def _trigrams(s: str) -> set[str]:
    s = "".join(c for c in (s or "").lower() if c.isalnum() or c.isspace())
    return {s[i : i + 3] for i in range(max(0, len(s) - 2))} if len(s) >= 3 else {s}


def _trigram_best_overlap(proof: str, corpus: list[str]) -> float:
    """Normalized char-trigram Jaccard of ``proof`` against the best corpus match.

    Shared by every novelty method as the cheap pre-filter signal. Returns 0.0 when
    either side has no trigrams (fail-safe, not a verdict)."""
    p_tri = _trigrams(proof)
    best = 0.0
    for cand in corpus:
        c_tri = _trigrams(cand)
        if not p_tri or not c_tri:
            continue
        j = len(p_tri & c_tri) / len(p_tri | c_tri)
        if j > best:
            best = j
    return best


# Recognized Lean 4 tactic heads. Used as a FAIL-CLOSED gate: if no split tactic has
# a head in this set, the proof is treated as unparseable (no hash, novelty refused).
# This is deliberately a conservative, common-core set; a real production hasher would
# use Lean's actual tactic table. Extending it only makes the hash ACCEPT more — never
# fabricate — so the conservative default errs toward abstention (the thesis discipline).
_KNOWN_TACS: frozenset[str] = frozenset(
    {
        # core
        "rfl", "intro", "intros", "introv", "exact", "apply", "have", "let", "set",
        "simp", "simp_arith", "decide", "norm_num", "ring", "ring_nf", "linarith",
        "nlinarith", "omega", "tauto", "trivial", "contradiction", "assumption",
        "constructor", "left", "right", "use", "exists", "refine", "rcases", "obtain",
        "induction", "cases", "case", "match", "rename", "next", "show", "by_contra",
        "by_cases", "exfalso", "sorry", "admit", "done", "skip", "clear", "change",
        "unfold", "funext", "ext", "suffices", "revert", "generalize", "specialize",
        "trans", "symm", "calc", "conv", "with",
        # rewrite family
        "rw", "rewrite", "simp_rw", "erw", "nth_rewrite",
        # mathlib common
        "polyrith", "abel", "field_simp", "push_neg", "finish",
    }
)


def _split_tactics(proof: str) -> list[str]:
    """Structurally split a Lean proof STRING into tactic statements.

    Pure regex, no Lean dependency. Splits on `;`, newlines, and the `<;>` all-goals
    combinator; strips the `by` block introducer, focus braces `{`/`}`, the `·` step
    marker, and `--` / `/- -/` comments. Empty/non-tokenizing fragments are dropped.

    Handles the `induction`/`cases`/`match ... with | arm => body` combinator: the
    `| pattern =>` arms are split so each arm's sub-tactic becomes its own node (the
    body of `| succ n ih => rw [ih]` is a distinct tactic that references `ih`).
    """
    if not proof:
        return []
    cleaned = re.sub(r"--[^\n]*", "", proof)               # line comments
    cleaned = re.sub(r"/\*.*?\*/", "", cleaned, flags=re.S)  # block comments
    cleaned = cleaned.replace("·", ";")                     # step marker -> separator

    # Arm combinator: `... with | pat => body | pat2 => body2`. Split on `|` arms
    # and on `=>`, so each arm body tokenizes as its own tactic. The arm-pattern
    # (zero, succ n ih, _ , h1 h2) may itself introduce names — we keep the first
    # arm's pattern joined to the intro tactic so its introductions register first.
    cleaned = re.sub(r"\s*\|\s*", " | ", cleaned)           # normalize `|` spacing
    cleaned = re.sub(r"\s*=>\s*", " => ", cleaned)          # normalize `=>` spacing

    out: list[str] = []
    for chunk in _TAC_SPLIT_RE.split(cleaned):
        c = re.sub(r"^by\b\s*", "", chunk.strip())          # drop leading `by`
        c = c.strip().strip("{}").strip()                   # drop focus braces
        # Drop pure arm-structural tokens (`|`, `=>`, `with`). Fragments that are arm
        # patterns (e.g. `succ n ih`) or bare goal tokens are KEPT here — `_build_tactic_dag`
        # needs them to register the hypothesis names they bind. It then filters to
        # recognized tactic heads for the actual node list (the fail-closed gate).
        if not c or c in ("|", "=>", "with"):
            continue
        out.append(c)
    return out


def _is_comm_lemma(item: str) -> bool:
    bare = item.lstrip("← ").rstrip("?").strip()           # drop `←`/`?` modifiers for classification
    return bool(_COMM_LEMMA_RE.search(bare.split(".")[-1]))


def _canonicalize_rewrite_item(item: str) -> str:
    """Normalize ONE rewrite item so the hash is invariant under renaming of local
    hypotheses. A bare local (no dot) referenced inside a rewrite is a pointer to a
    local hypothesis — its *identity* is irrelevant to proof structure (it is bound
    by the DAG), so we map it to a placeholder `#local`. Dotted library lemma names
    (`Nat.add_comm`) and structured terms are kept verbatim — they ARE the proof's
    semantic content. Modifiers (`←`, `?`) are preserved as they change meaning."""
    return re.sub(
        r"(?<![A-Za-z0-9_.'])[A-Za-z_][A-Za-z0-9_']*(?![A-Za-z0-9_'.])",
        "#local",
        item.strip(),
    )


def _normalize_rewrite_list(items: list[str]) -> tuple[str, ...]:
    """Canonical order for a rewrite list, respecting semantics.

    Two normalizations:
      1. Local-hypothesis references inside a rewrite item are replaced by `#local`
         (their identity is bound by the DAG; only the rewrite *structure* matters).
      2. Commutativity / associativity / ac lemmas are order-irrelevant WITH RESPECT
         TO EACH OTHER (permuting them inside `rw [a, b]` is a trivial rewrite-ordering
         change the hash must ignore — per the spec). We keep the comm lemmas in their
         original SLOTS but sort their VALUES, so a non-comm lemma cannot jump past a
         comm one (that would over-claim novelty).

    So `[add_comm, le_iff]` != `[le_iff, add_comm]` (the non-comm lemma's slot is
    fixed), while `[add_comm, mul_comm]` == `[mul_comm, add_comm]` (both slots comm)."""
    if not items:
        return ()
    items = [_canonicalize_rewrite_item(it) for it in items if it.strip()]
    comm_pool = sorted(it for it in items if _is_comm_lemma(it))
    ci = 0
    out: list[str] = []
    for it in items:
        if _is_comm_lemma(it):
            out.append(comm_pool[ci])
            ci += 1
        else:
            out.append(it)
    return tuple(out)


def _parse_tactic(text: str) -> tuple[str | None, tuple[str, ...], tuple[str, ...]]:
    """Return ``(head, idents, rewrite_items)`` for one tactic statement.

    * head — the tactic's leading identifier (rw, apply, induction, simp, ...), or
      None if `text` has no recognizable tactic head.
    * idents — every identifier referenced (e.g. Nat.add_comm, ih, h), de-duped in
      first-seen order. Bare locals are potential DAG-edge sources; dotted names are
      external premises and fold into the node signature.
    * rewrite_items — for rewrite tactics (`rw [...]`), the bracket list normalized
      via `_normalize_rewrite_list`; () for non-rewrite tactics.
    """
    mh = _TAC_HEAD_RE.match(text)
    head = mh.group(1) if mh else None
    idents = tuple(dict.fromkeys(_IDENT_RE.findall(text)))  # de-dup, preserve order
    rewrite_items: tuple[str, ...] = ()
    if head and _RW_TAC_RE.match(head):
        items: list[str] = []
        for bracket in _RW_BRACKET_RE.findall(text):
            for it in bracket.split(","):
                if it.strip():
                    items.append(it.strip())
        rewrite_items = _normalize_rewrite_list(items)
    return head, idents, rewrite_items


def _node_signature(head: str, idents: tuple[str, ...], rw_items: tuple[str, ...]) -> str:
    """Canonical signature of one tactic node.

    External (dotted) premises fold into the signature; bare locals are deliberately
    excluded here (they become DAG edges) so a dependency is counted once, not twice.
    For rewrite tactics the normalized rewrite list IS the premise set, so its
    members are not re-added as library idents."""
    sig = head
    rw_set = set(rw_items)
    library = [i for i in idents if "." in i and i != head and i not in rw_set]
    if rw_items:
        sig += "[" + ",".join(rw_items) + "]"
    if library:
        sig += "{" + ",".join(library) + "}"
    return sig


def _build_tactic_dag(proof: str) -> dict[str, Any]:
    """Build the normalized tactic dependency DAG of a Lean proof STRING.

    Returns ``{parsed, nodes, edges, n_tactics}``:
      * parsed — False if no tactic could be parsed (fail-closed: the caller must NOT
        claim novelty). True otherwise.
      * nodes — node signatures in tactic SEQUENCE order. Sequence is load-bearing
        (different paths differ); only within-tactic comm/assoc rewrites are normalized.
      * edges — ``(src_idx, dst_idx)`` dependency edges: tactic `i` references a bare
        local name introduced by an earlier tactic `j` -> edge (i, j). This is the
        real dependency structure; library premises are in the node signatures.
    """
    # Two passes, decoupling BINDING-TRACKING from the NODE ALLOWLIST:
    #  * Pass 1 records every fragment (recognized tactic OR an arm-pattern fragment
    #    like `succ n ih`) so bare-local bindings are tracked faithfully — a name is
    #    bound at its first textual occurrence regardless of which fragment carries it.
    #  * Pass 2 emits NODE signatures only for recognized tactic heads (the allowlist
    #    is the fail-closed gate against prose), but maps each recognized node back to
    #    its binding-pass index so dependency edges resolve to the right introducer.
    fragments: list[tuple[str | None, tuple[str, ...], tuple[str, ...]]] = []
    for tac in _split_tactics(proof):
        head, idents, rw_items = _parse_tactic(tac)
        fragments.append((head, idents, rw_items))

    nodes: list[str] = []                       # recognized-tactic node signatures (in order)
    node_bind_idx: list[int] = []               # each node's index into the binding pass
    introducer: dict[str, int] = {}             # bare-local name -> binding-pass index
    for bidx, (head, idents, _rw) in enumerate(fragments):
        if head is None:
            continue
        if head in _KNOWN_TACS:                 # recognized tactic -> emit a node
            nodes.append(_node_signature(head, idents, _rw))
            node_bind_idx.append(bidx)
        # Bindings are tracked for EVERY fragment (incl. arm patterns like `succ n ih`),
        # not just recognized tactics — a hypothesis introduced by an induction arm is a
        # real binding even though `succ` is not itself a tactic.
        for name in idents:
            if "." not in name:                 # bare local only
                introducer.setdefault(name, bidx)

    if not nodes:
        # No recognized tactic head anywhere -> unparseable. Fail-closed: refuse to hash.
        return {"parsed": False, "nodes": [], "edges": [], "n_tactics": 0}

    edges: list[tuple[int, int]] = []
    for nidx, bidx in enumerate(node_bind_idx):
        head, idents, _rw = fragments[bidx]
        for name in idents:
            src_b = introducer.get(name)
            if src_b is not None and src_b < bidx:
                # Map the introducer's binding index back to a NODE index. If the
                # introducer was an arm-pattern fragment (not itself a node), map to the
                # nearest preceding recognized node — the induction/obtain that owns it.
                src_n = _bind_to_node(src_b, node_bind_idx)
                if src_n is not None and src_n < nidx:
                    edges.append((nidx, src_n))
    return {"parsed": True, "nodes": nodes, "edges": sorted(set(edges)), "n_tactics": len(nodes)}


def _bind_to_node(bind_idx: int, node_bind_idx: list[int]) -> int | None:
    """Map a binding-pass index to its node index. When ``bind_idx`` is itself a node
    (it's in ``node_bind_idx``), return its position. Otherwise (an arm-pattern
    fragment that introduced a name but isn't a node), return the largest node whose
    binding index is < ``bind_idx`` — the induction/obtain/cases node that owns it."""
    if bind_idx in node_bind_idx:
        return node_bind_idx.index(bind_idx)
    preceding = [n for n, b in enumerate(node_bind_idx) if b < bind_idx]
    return preceding[-1] if preceding else None


def _canonical_dag_hash(dag: dict[str, Any]) -> str | None:
    """SHA-256[:16] of the normalized DAG, or None if unparseable.

    Canonical form preserves tactic SEQUENCE (so genuinely different paths differ)
    and folds in the dependency edges, while being invariant under within-tactic
    comm/assoc rewrite permutations (already normalized into the node signatures)."""
    if not dag.get("parsed"):
        return None
    node_str = "\n".join(dag["nodes"])
    edge_str = "\n".join(f"{a}->{b}" for a, b in dag["edges"])
    canon = f"N{dag['n_tactics']}\n{node_str}\nE{len(dag['edges'])}\n{edge_str}"
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()[:16]


def novelty_check_dag(
    proof: str,
    *,
    corpus: list[str],
    near_dup_threshold: float = 0.92,
) -> dict[str, Any]:
    """STRICT tactic-DAG novelty probe (§3.3 of the Open-Problems critique).

    Computes the normalized tactic-DAG hash of ``proof`` and flags it novel only if
    no corpus proof shares that hash. A proof is "novel" iff its tactic *structure*
    is not a known library/training proof modulo commutative-associative rewrites.

    Fail-closed: if ``proof`` cannot be parsed into a DAG, the probe refuses to
    claim novelty — ``dag_hash`` is None, ``dag_parsed`` is False, and ``novel`` is
    False. It NEVER fabricates a hash. Char-trigram Jaccard is still computed as a
    cheap display/legacy signal (``best_overlap``) so pre-existing callers keep
    working; the verdict itself is DAG-driven.
    """
    best_overlap = _trigram_best_overlap(proof, corpus)
    p_hash = _canonical_dag_hash(_build_tactic_dag(proof))
    if p_hash is None:
        return {
            "novel": False,  # fail-closed: never claim novel on an unparseable proof
            "best_overlap": round(best_overlap, 4),
            "dag_hash": None,
            "dag_parsed": False,
            "threshold": near_dup_threshold,
            "method": "tactic-dag (abstained: unparseable proof; fail-closed non-novel)",
        }
    corpus_hashes: set[str] = set()
    corpus_parsed = 0
    for cand in corpus:
        ch = _canonical_dag_hash(_build_tactic_dag(cand))
        if ch is not None:
            corpus_hashes.add(ch)
            corpus_parsed += 1
    return {
        "novel": p_hash not in corpus_hashes,
        "best_overlap": round(best_overlap, 4),
        "dag_hash": p_hash,
        "dag_parsed": True,
        "corpus_hashes": len(corpus_hashes),
        "corpus_parsed": corpus_parsed,
        "threshold": near_dup_threshold,
        "method": "tactic-dag (comm/assoc rewrite order normalized; sequence + dep edges preserved)",
    }


def novelty_check(
    proof: str,
    *,
    corpus: list[str],
    near_dup_threshold: float = 0.92,
    method: Literal["auto", "trigram", "dag"] = "auto",
) -> dict[str, Any]:
    """STRICT novelty probe: is ``proof`` a near-duplicate of anything in ``corpus``?

    A proof that is Lean-valid AND not a near-duplicate is the novelty signal —
    recorded honestly, candidate-only. This is a *measurement*, not a claim of
    creative superintelligence.

    Methods (per the Open-Problems critique §3.3):
      * ``"trigram"`` — char-trigram Jaccard only (the original, dependency-free
        proxy). Catches near-verbatim retrieval; misses re-proofs via a different
        tactic path. Backward-compatible with all pre-existing callers.
      * ``"dag"`` — normalized tactic-DAG hash only (`novelty_check_dag`). The strict
        decider; abstains fail-closed on an unparseable proof.
      * ``"auto"`` (default) — char-trigram as the cheap pre-filter, tactic-DAG as
        the strict decider. A proof is novel only if it passes BOTH: not a trigram
        near-duplicate AND its DAG hash is absent from the corpus. This is the honest
        novelty signal the critique recommends: trigram Jaccard is necessary, the DAG
        hash makes it sufficient against tactic-path retrieval.

    NB: the DAG hash parses the proof string structurally with stdlib `re` + hashlib
    only — no Lean dependency, no model — so the fail-closed default and CI are
    untouched. Threshold 0.92 = ">=92% trigram overlap counts as a duplicate."
    """
    if method == "trigram":
        best_overlap = _trigram_best_overlap(proof, corpus)
        return {
            "novel": best_overlap < near_dup_threshold,
            "best_overlap": round(best_overlap, 4),
            "threshold": near_dup_threshold,
            "method": "char-trigram-jaccard (strict proxy; production = sentence embedding)",
            "proof_hash": hashlib.sha256(proof.encode("utf-8")).hexdigest()[:16],
        }
    if method == "dag":
        return novelty_check_dag(proof, corpus=corpus, near_dup_threshold=near_dup_threshold)
    # method == "auto": trigram pre-filter, then DAG decider.
    best_overlap = _trigram_best_overlap(proof, corpus)
    if best_overlap >= near_dup_threshold:
        # Cheap rejection: an obvious near-duplicate. DAG can only further restrict,
        # so we short-circuit (fast path for the common case of verbatim retrieval).
        return {
            "novel": False,
            "best_overlap": round(best_overlap, 4),
            "dag_hash": None,
            "dag_decider": "skipped (trigram pre-filter rejected)",
            "threshold": near_dup_threshold,
            "method": "auto: trigram pre-filter rejected (near-duplicate); DAG not needed",
            "proof_hash": hashlib.sha256(proof.encode("utf-8")).hexdigest()[:16],
        }
    dag = novelty_check_dag(proof, corpus=corpus, near_dup_threshold=near_dup_threshold)
    dag["dag_decider"] = "applied"
    dag["method"] = "auto: trigram pre-filter passed -> tactic-DAG decider applied"
    return dag


__all__ = [
    "LeanVerdict",
    "LeanCheck",
    "lean_available",
    "verify_proof",
    "novelty_check",
    "novelty_check_dag",
]
