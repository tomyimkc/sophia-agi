# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Voyager/AutoSkill-style executable skill library with an anti-forgetting gate.

This is **not** a claim of AGI and it changes **no weights**. It is the missing
rung for compositional, retention-tested skill acquisition: a skill is *executable
code* (a sandboxed ``def solve(x): ...`` function) plus a *declarative verifier
spec* (preconditions / effects / verifier cases), versioned and gated. Skills may
*compose* one another by declaring deps; a composite skill calls already-admitted,
already-verified deps that are injected into its execution namespace.

The trust boundary is the measured verifier, not the proposal source, and the
admission gate is the same anti-forgetting tripwire we apply to facts: admitting or
upgrading a skill may **never** silently regress a previously-learned skill. A
brand-new id has no dependents and so cannot regress anything; an *upgrade* (same
id, version+1) that breaks any transitive dependent is **rejected** and the prior
version is kept — the catastrophic-forgetting analogue, refused.

Discipline (the "Sophia discipline"):
  * deterministic / offline — no wall-clock asserted, no unseeded randomness;
  * fail-closed — unsafe code, missing dep, precondition failure, or any
    ambiguity => reject/abstain, never a silent accept;
  * non-parametric — a skill is code in a library, not a learned adapter;
  * every emitted verdict carries ``"candidateOnly": True``.

The library sandboxes skill code via :mod:`agent.program_induction`
(``ast_program_is_safe`` / ``compile_program_source``) and reuses the
allowlisted AST node set; for *composition* it extends the name allowlist to the
declared dep ids only, so a composite cannot reach any name it did not declare.
The core admission decision is a **deterministic protected-regression tripwire**
computed directly (mirroring :mod:`agent.continual_retention`); an adapter
``to_update_candidate`` is additionally exposed to render the promotion through
:mod:`agent.continual_plasticity` for parity, but the conscience-coupled verdict
is never the gate.

    lib = SkillLibrary()
    lib.learn(Skill("double", source="def solve(x): return x * 2",
                    verifier_cases=[{"input": 3, "expected": 6}]))
    lib.invoke("double", 5)        # 10
    lib.version_tag("double")      # 'double@v1#<sha8>'
"""

from __future__ import annotations

import ast
import hashlib
from dataclasses import dataclass, field
from typing import Any, Callable

from agent.program_induction import _ALLOWED_NODES, _SAFE_BUILTINS, ast_program_is_safe

SCHEMA = "sophia.skill_library.v1"
RETENTION_SCHEMA = "sophia.skill_retention.v1"

# Documented pass-rate floor for ``ok``. Skills are tiny, deterministic, and
# hand-verified, so the floor is exact: a skill is verified iff it passes every
# one of its declared verifier cases.
PASS_FLOOR = 1.0

_MAX_SOURCE = 2200


@dataclass(frozen=True)
class Skill:
    """An executable skill plus its declarative verifier spec.

    ``source`` must define exactly one ``def solve(x): ...`` over the allowlisted
    AST node set; it may reference its declared ``deps`` by id (those callables
    are injected at execution time). ``version`` starts at 1; an upgrade keeps the
    id and increments the version.
    """

    id: str
    source: str
    version: int = 1
    preconditions: tuple[str, ...] = ()
    effects: tuple[str, ...] = ()
    verifier_cases: tuple[dict[str, Any], ...] = ()
    deps: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        # Normalise list inputs to tuples so the dataclass stays hashable and
        # callers may pass plain lists ergonomically.
        object.__setattr__(self, "preconditions", tuple(self.preconditions))
        object.__setattr__(self, "effects", tuple(self.effects))
        object.__setattr__(self, "verifier_cases", tuple(self.verifier_cases))
        object.__setattr__(self, "deps", tuple(self.deps))


# --------------------------------------------------------------------------- #
# Sandbox: reuse the program-induction allowlist, but permit declared dep names.
# --------------------------------------------------------------------------- #
def _skill_ast_is_safe(tree: ast.AST, allowed_names: set[str]) -> bool:
    """Allow exactly one ``def solve(x): return ...`` over the allowlisted nodes,
    permitting Load references to ``x``, safe builtins, and the *declared* deps.

    This mirrors :func:`agent.program_induction.ast_program_is_safe` (same node
    allowlist, same call/constant guards) and additionally admits the dep names so
    a composite skill can call its deps — and *only* its deps. Any undeclared name
    is rejected (fail-closed); imports/dunders/eval are rejected because their
    nodes are not in the allowlist.
    """
    body = getattr(tree, "body", [])
    funcs = [n for n in body if isinstance(n, ast.FunctionDef)]
    if len(body) != 1 or len(funcs) != 1:
        return False
    fn = funcs[0]
    if fn.name != "solve" or fn.decorator_list or len(fn.args.args) != 1:
        return False
    if fn.args.args[0].arg != "x":
        return False
    if not fn.body or not isinstance(fn.body[-1], ast.Return):
        return False
    callable_names = set(_SAFE_BUILTINS) | set(allowed_names)
    load_names = {"x", *_SAFE_BUILTINS, *allowed_names}
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            return False
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in callable_names:
                return False
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)) and abs(float(node.value)) > 1_000_000:
                return False
            if isinstance(node.value, str) and len(node.value) > 10_000:
                return False
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
            if isinstance(node.left, (ast.List, ast.Tuple, ast.Dict)) or isinstance(node.right, (ast.List, ast.Tuple, ast.Dict)):
                return False
            for side in (node.left, node.right):
                if isinstance(side, ast.Constant):
                    if isinstance(side.value, str):
                        return False
                    if isinstance(side.value, (int, float)) and abs(float(side.value)) > 1000:
                        return False
        if isinstance(node, ast.Name):
            if isinstance(node.ctx, ast.Load) and node.id not in load_names:
                return False
    return True


def _compile_skill(skill: Skill, dep_callables: dict[str, Callable[..., Any]]) -> Callable[..., Any] | None:
    """Compile a skill's ``solve`` after AST sandboxing, injecting dep callables.

    Returns ``None`` (=> unsafe / uncompilable => reject) on any failure. When the
    skill has no deps and references no extra names, this is exactly the
    program-induction sandbox; the dep-aware allowlist only *widens* by the
    declared dep ids.
    """
    source = skill.source
    if not isinstance(source, str) or "solve" not in source or len(source) > _MAX_SOURCE:
        return None
    try:
        tree = ast.parse(source, "<skill-candidate>", "exec")
    except SyntaxError:
        return None
    allowed = set(skill.deps)
    if not _skill_ast_is_safe(tree, allowed):
        return None
    # Belt-and-suspenders: a depless skill must also clear the base sandbox so we
    # never quietly diverge from program_induction's guarantee for simple skills.
    if not allowed and not ast_program_is_safe(tree):
        return None
    # Inject dep callables into the *globals* of the compiled function so that
    # ``solve`` can resolve dep names at call time (a function's free names are
    # looked up in its ``__globals__``, not the exec locals). Builtins are pinned
    # to the safe allowlist so the body can reach nothing else.
    glb: dict[str, Any] = {"__builtins__": _SAFE_BUILTINS}
    glb.update(dep_callables)
    try:
        exec(compile(tree, "<skill-candidate>", "exec"), glb)  # noqa: S102
    except Exception:
        return None
    fn = glb.get("solve")
    if not callable(fn):
        return None
    return fn


def _eq(a: Any, b: Any) -> bool:
    if isinstance(a, float) or isinstance(b, float):
        try:
            return abs(float(a) - float(b)) <= 1e-9
        except Exception:
            return False
    return a == b


# --------------------------------------------------------------------------- #
# Verification
# --------------------------------------------------------------------------- #
def verify_skill(skill: Skill, library: "SkillLibrary") -> dict[str, Any]:
    """Run every verifier case through the composed, compiled skill.

    Fail-closed: if any dep is missing or not itself verified, or the skill does
    not compile, ``ok`` is False with ``passed=0``. ``ok`` iff ``passRate >=
    PASS_FLOOR`` AND all deps are present and verified.
    """
    total = len(skill.verifier_cases)

    def _result(passed: int, ok: bool, reason: str) -> dict[str, Any]:
        rate = round(passed / total, 6) if total else 0.0
        return {
            "skillId": skill.id,
            "version": skill.version,
            "passed": passed,
            "total": total,
            "passRate": rate,
            "ok": bool(ok),
            "reason": reason,
            "candidateOnly": True,
        }

    # Resolve dep callables: each dep must be admitted AND verified.
    dep_callables: dict[str, Callable[..., Any]] = {}
    for dep_id in skill.deps:
        admitted = library.get(dep_id)
        if admitted is None:
            return _result(0, False, f"missing dep: {dep_id}")
        if not library.is_verified(dep_id):
            return _result(0, False, f"unverified dep: {dep_id}")
        dep_fn = library.compiled(dep_id)
        if dep_fn is None:
            return _result(0, False, f"uncompilable dep: {dep_id}")
        dep_callables[dep_id] = dep_fn

    fn = _compile_skill(skill, dep_callables)
    if fn is None:
        return _result(0, False, "unsafe or uncompilable source")

    if total == 0:
        # No cases to prove the skill; cannot certify => fail-closed.
        return _result(0, False, "no verifier cases")

    passed = 0
    for case in skill.verifier_cases:
        args = _case_args(case.get("input"))
        try:
            got = fn(*args)
        except Exception:
            continue
        if _eq(got, case.get("expected")):
            passed += 1
    rate = passed / total
    ok = rate >= PASS_FLOOR
    return _result(passed, ok, "verified" if ok else "below pass floor")


def _case_args(raw: Any) -> tuple[Any, ...]:
    """A verifier-case ``input`` is the single argument ``x``. (solve takes one
    arg by construction.) We pass it through verbatim."""
    return (raw,)


# --------------------------------------------------------------------------- #
# The library + the anti-forgetting gate
# --------------------------------------------------------------------------- #
@dataclass
class SkillLibrary:
    """Versioned, gated, compositional skill store with a retention tripwire."""

    _skills: dict[str, Skill] = field(default_factory=dict)
    _verified: dict[str, bool] = field(default_factory=dict)

    # -- read-only accessors ------------------------------------------------- #
    def get(self, skill_id: str) -> Skill | None:
        return self._skills.get(skill_id)

    def is_verified(self, skill_id: str) -> bool:
        return bool(self._verified.get(skill_id, False))

    def ids(self) -> list[str]:
        return sorted(self._skills)

    def compiled(self, skill_id: str) -> Callable[..., Any] | None:
        """Compile an admitted skill with its (already-admitted) deps injected."""
        skill = self._skills.get(skill_id)
        if skill is None:
            return None
        dep_callables: dict[str, Callable[..., Any]] = {}
        for dep_id in skill.deps:
            dep_fn = self.compiled(dep_id)
            if dep_fn is None:
                return None
            dep_callables[dep_id] = dep_fn
        return _compile_skill(skill, dep_callables)

    # -- dependency graph ---------------------------------------------------- #
    def _direct_dependents(self, skill_id: str) -> list[str]:
        return sorted(sid for sid, s in self._skills.items() if skill_id in s.deps)

    def transitive_dependents(self, skill_id: str) -> list[str]:
        """All admitted skills that (transitively) depend on ``skill_id``."""
        seen: set[str] = set()
        frontier = [skill_id]
        while frontier:
            cur = frontier.pop()
            for dep_user in self._direct_dependents(cur):
                if dep_user not in seen:
                    seen.add(dep_user)
                    frontier.append(dep_user)
        return sorted(seen)

    def _protected_rates(self, skill_ids: list[str]) -> dict[str, float]:
        """Current passRate of each given admitted skill (deterministic)."""
        rates: dict[str, float] = {}
        for sid in skill_ids:
            s = self._skills.get(sid)
            if s is None:
                continue
            rates[sid] = verify_skill(s, self)["passRate"]
        return rates

    # -- the GATE ------------------------------------------------------------ #
    def learn(self, skill: Skill) -> dict[str, Any]:
        """Admit a skill IFF it self-verifies AND the anti-forgetting tripwire holds.

        Rule:
          (a) the skill compiles safe and passes its own verifier (passRate >=
              PASS_FLOOR with all deps present+verified); else reject; AND
          (b) the protected-regression tripwire: re-verify EVERY previously-admitted
              skill that transitively depends on this id *after* tentatively applying
              the change; if ANY dependent's passRate drops vs before, REJECT and keep
              the prior version. A brand-new id has no dependents => admit. An upgrade
              (same id, version+1) that breaks a dependent => REJECT.
        """
        skill_id = skill.id
        prior = self._skills.get(skill_id)
        is_upgrade = prior is not None

        # Snapshot the protected set (transitive dependents) and their before-rates
        # against the CURRENT library state.
        dependents = self.transitive_dependents(skill_id)
        before = self._protected_rates(dependents)

        def _decision(decision: str, reason: str, after: dict[str, float], regressed: list[str], version: int) -> dict[str, Any]:
            return {
                "skillId": skill_id,
                "version": version,
                "decision": decision,
                "reason": reason,
                "regressedDependents": regressed,
                "protectedBefore": before,
                "protectedAfter": after,
                "candidateOnly": True,
                "schema": SCHEMA,
            }

        # (a) self-verify the candidate against the CURRENT library (its deps must
        # already be admitted+verified). We verify the candidate as proposed.
        candidate = skill
        if is_upgrade:
            # An upgrade must carry a strictly greater version (forgetting-resistant
            # versioning: you cannot silently replace a skill at the same version).
            if candidate.version <= prior.version:
                return _decision("reject", f"upgrade must bump version (> {prior.version})", {}, [], prior.version)

        self_report = verify_skill(candidate, self)
        if not self_report["ok"]:
            keep_version = prior.version if is_upgrade else candidate.version
            return _decision("reject", f"candidate failed own verifier: {self_report['reason']}", {}, [], keep_version)

        # (b) tentatively apply, re-verify dependents, then decide. We mutate, then
        # roll back on regression so the library is never left in a worse state.
        saved_skill = prior
        saved_verified = self._verified.get(skill_id, False)
        self._skills[skill_id] = candidate
        self._verified[skill_id] = True
        try:
            after = self._protected_rates(dependents)
        finally:
            pass

        regressed = sorted(d for d in dependents if after.get(d, 0.0) < before.get(d, 0.0) - 1e-9)
        if regressed:
            # ROLL BACK: keep the prior version (or remove a never-admitted new id).
            if saved_skill is None:
                self._skills.pop(skill_id, None)
                self._verified.pop(skill_id, None)
            else:
                self._skills[skill_id] = saved_skill
                self._verified[skill_id] = saved_verified
            keep_version = prior.version if is_upgrade else candidate.version
            return _decision(
                "reject",
                "anti-forgetting tripwire: dependent skill(s) regressed",
                after,
                regressed,
                keep_version,
            )

        # Admit: the candidate is now the live version.
        return _decision("admit", "self-verified and no dependent regression", after, [], candidate.version)

    # -- invocation ---------------------------------------------------------- #
    def invoke(self, skill_id: str, *args: Any) -> Any:
        """Run an admitted, verified skill (composing its deps).

        Fail-closed: invoking an unadmitted or unverified skill raises KeyError /
        RuntimeError rather than guessing a result.
        """
        skill = self._skills.get(skill_id)
        if skill is None:
            raise KeyError(f"unknown skill: {skill_id}")
        if not self.is_verified(skill_id):
            raise RuntimeError(f"skill not verified: {skill_id}")
        fn = self.compiled(skill_id)
        if fn is None:
            raise RuntimeError(f"skill failed to compile: {skill_id}")
        return fn(*args)

    # -- deterministic version pin ------------------------------------------ #
    def version_tag(self, skill_id: str) -> str:
        """Deterministic version pin ``id@vN#<sha8>`` over the skill source.

        Process-independent: the sha is a SHA-256 over a canonical encoding of the
        skill id, version, and source text (no wall-clock, no object ids).
        """
        skill = self._skills.get(skill_id)
        if skill is None:
            raise KeyError(f"unknown skill: {skill_id}")
        return version_tag(skill)

    # -- plasticity-gate parity adapter ------------------------------------- #
    def to_update_candidate(self, skill_id: str, before_after_rows: dict[str, tuple[float, float]]):
        """Render a skill promotion as an :class:`continual_plasticity.UpdateCandidate`.

        Target = the skill's own verifier delta (keyed by suite ``skill:<id>``);
        protected = each dependent skill's passRate before/after. This lets a
        promotion be run through ``continual_plasticity.evaluate_update`` for
        PARITY. NOTE: the library's authoritative decision is the deterministic
        tripwire in :meth:`learn`, NOT the conscience-coupled plasticity verdict.

        ``before_after_rows`` maps suite name -> (before, after). The skill's own
        suite (``skill:<id>``) is treated as the target; all others are protected
        dependent suites.
        """
        from agent.continual_plasticity import EvalMetric, UpdateCandidate

        target_suite = f"skill:{skill_id}"
        metrics: list[EvalMetric] = []
        for suite, (before, after) in sorted(before_after_rows.items()):
            metrics.append(EvalMetric(suite, float(before), float(after), protected=(suite != target_suite)))
        skill = self._skills.get(skill_id)
        version = skill.version if skill is not None else 0
        return UpdateCandidate(
            id=f"{skill_id}@v{version}",
            kind="skill",
            metrics=tuple(metrics),
            verifier_artifacts=(f"verifier-cases:{skill_id}", f"dependent-retention:{skill_id}"),
            contaminated=False,
            notes=f"skill promotion {skill_id} v{version}",
        )


def version_tag(skill: Skill) -> str:
    """Module-level deterministic pin so a pin can be computed without a library."""
    payload = f"{skill.id}\x1f{skill.version}\x1f{skill.source}".encode("utf-8")
    sha = hashlib.sha256(payload).hexdigest()[:8]
    return f"{skill.id}@v{skill.version}#{sha}"


# --------------------------------------------------------------------------- #
# Skill-retention benchmark (mirrors agent.continual_retention philosophy)
# --------------------------------------------------------------------------- #
def _retention_stream() -> list[Skill]:
    """A deterministic stream of skills, later ones composing earlier ones.

    inc -> double -> quadruple(uses double) -> octuple(uses quadruple).
    Hand-built tiny verifier cases; no randomness.
    """
    return [
        Skill(
            id="inc",
            source="def solve(x): return x + 1",
            verifier_cases=({"input": 0, "expected": 1}, {"input": 4, "expected": 5}),
            preconditions=("x is a number",),
            effects=("returns x + 1",),
        ),
        Skill(
            id="double",
            source="def solve(x): return x * 2",
            verifier_cases=({"input": 3, "expected": 6}, {"input": 0, "expected": 0}),
            preconditions=("x is a number",),
            effects=("returns 2*x",),
        ),
        Skill(
            id="quadruple",
            source="def solve(x): return double(double(x))",
            deps=("double",),
            verifier_cases=({"input": 3, "expected": 12}, {"input": 1, "expected": 4}),
            preconditions=("double is admitted",),
            effects=("returns 4*x via double(double(x))",),
        ),
        Skill(
            id="octuple",
            source="def solve(x): return double(quadruple(x))",
            deps=("double", "quadruple"),
            verifier_cases=({"input": 1, "expected": 8}, {"input": 2, "expected": 16}),
            preconditions=("double and quadruple admitted",),
            effects=("returns 8*x",),
        ),
    ]


def skill_retention_benchmark() -> dict[str, Any]:
    """Learn a stream of (composing) skills; prove 0 regression of prior skills.

    Builds a retention matrix: after each ``learn``, re-verify ALL prior skills and
    record their passRates (mirroring ``agent.continual_retention``). Then attempt a
    BREAKING upgrade of an early depended-on skill (changes ``double`` so the
    composite ``quadruple`` fails its verifier) and assert it is REJECTED with all
    prior skills still at passRate 1.0. Then a NON-breaking improvement upgrade ->
    admitted (version bumps). Deterministic; ``candidateOnly``.
    """
    lib = SkillLibrary()
    stream = _retention_stream()
    retention_matrix: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []

    for skill in stream:
        decision = lib.learn(skill)
        decisions.append(decision)
        # After each learn, re-verify ALL admitted skills (prior + just-learned).
        row: dict[str, Any] = {"afterLearning": skill.id, "decision": decision["decision"], "rates": {}}
        for sid in lib.ids():
            s = lib.get(sid)
            row["rates"][sid] = verify_skill(s, lib)["passRate"]
        retention_matrix.append(row)

    all_admitted = all(d["decision"] == "admit" for d in decisions)

    # Reuse-across-episodes: an early skill (double, episode 2) is composed by a
    # later composite (quadruple, episode 3 / octuple, episode 4) and invocation
    # of the composite uses the early skill's code.
    reused = False
    try:
        reused = lib.invoke("octuple", 3) == 24 and lib.invoke("quadruple", 5) == 20
    except Exception:
        reused = False

    # BREAKING upgrade of `double`: change behavior to x*3 so quadruple (expects
    # 4*x) regresses. Must be REJECTED by the tripwire; quadruple stays at 1.0.
    quad_before = verify_skill(lib.get("quadruple"), lib)["passRate"]
    breaking = lib.learn(Skill(
        id="double",
        version=2,
        source="def solve(x): return x * 3",
        verifier_cases=({"input": 3, "expected": 9}, {"input": 0, "expected": 0}),
    ))
    quad_after = verify_skill(lib.get("quadruple"), lib)["passRate"]
    breaking_rejected = (
        breaking["decision"] == "reject"
        and "quadruple" in breaking["regressedDependents"]
        and lib.get("double").version == 1
        and quad_after >= quad_before
        and quad_after == 1.0
    )

    # NON-breaking improvement upgrade of `double`: same x*2 behavior, richer spec /
    # version bump. Dependents keep passing => admitted, version -> 2.
    nonbreaking = lib.learn(Skill(
        id="double",
        version=2,
        source="def solve(x): return x + x",
        verifier_cases=({"input": 3, "expected": 6}, {"input": 7, "expected": 14}),
        effects=("returns 2*x via x + x (improved)",),
    ))
    nonbreaking_admitted = nonbreaking["decision"] == "admit" and lib.get("double").version == 2

    # Forgotten skills: any admitted skill whose final passRate < 1.0.
    forgotten = 0
    for sid in lib.ids():
        if verify_skill(lib.get(sid), lib)["passRate"] < 1.0:
            forgotten += 1

    ok = (
        all_admitted
        and forgotten == 0
        and breaking_rejected
        and nonbreaking_admitted
        and reused
    )
    return {
        "ok": bool(ok),
        "retentionMatrix": retention_matrix,
        "forgottenSkills": forgotten,
        "breakingUpgradeRejected": bool(breaking_rejected),
        "nonBreakingUpgradeAdmitted": bool(nonbreaking_admitted),
        "reusedAcrossEpisodes": bool(reused),
        "decisions": decisions,
        "schema": RETENTION_SCHEMA,
        "candidateOnly": True,
    }


__all__ = [
    "Skill",
    "SkillLibrary",
    "verify_skill",
    "version_tag",
    "skill_retention_benchmark",
    "SCHEMA",
    "RETENTION_SCHEMA",
    "PASS_FLOOR",
]
