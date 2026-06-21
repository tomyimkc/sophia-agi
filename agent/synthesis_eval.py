"""Falsifiable evaluation of verifier synthesis — the ablation that earns the claim.

Builds a deterministic suite of *novel* tasks the synthesiser has never seen, split
into two groups by an honest criterion:

  - IN-LIBRARY  — the task's hidden rule is expressible by a template AND
    learnable from a sample (even, prime, divisible, positive, fixed-length
    digits, ISO date). Capability target: the synthesised gate generalises to
    held-out answers. (A *continuous* bound like "in [50,150]" is expressible but
    not exactly recoverable from a sample, so the strict precision floor makes the
    system abstain — correct conservative behaviour, hence omitted here.)
  - OUT-OF-LIBRARY — the rule is NOT expressible (numeric palindrome, perfect
    square, contains-a-7), and each decoy is **length-matched** to a correct
    answer (same digit count ⇒ same range/length), so no template can separate
    them on any split. Safety target: the system ABSTAINS on all of them.

Then it runs the same suite twice — WITH and WITHOUT meta-verification — so the
only variable is whether candidates are validated before they are trusted. The
contrast is the falsifiable result:
  * WITH:    in-library gates are precise and catch errors; out-of-library abstains.
  * WITHOUT: out-of-library tasks get a confident, wrong "verifier" (false
    admission), and in-library precision degrades — proving the meta-verification,
    not the template library, is what makes synthesis trustworthy.

Plus a calibration demo (competence where no verifier exists): label-free
self-consistency confidences whose selective risk beats answering everything.

All deterministic (seeded), no model, no network — so it gates CI.
"""

from __future__ import annotations

import math
import random

from agent import calibration as cal
from agent import horizon as hz
from agent import verifier_synthesis as vs


def _examples(corrects, incorrects) -> list:
    return ([{"answer": v, "label": True} for v in corrects]
            + [{"answer": v, "label": False} for v in incorrects])


def _is_pal(n: int) -> bool:
    s = str(n)
    return s == s[::-1]


def _ndigit(rng, d: int) -> int:
    lo = 10 ** (d - 1) if d > 1 else 1
    return rng.randint(lo, 10 ** d - 1)


def _matched_decoys(rng, corrects, predicate) -> list:
    """One decoy per correct, with the SAME digit length but NOT satisfying the
    rule. Matching length (hence range and digit-count) by construction means no
    in_range / length_range / regex template can separate correct from decoy — so
    abstention reflects the rule being inexpressible, not a sampling artefact."""
    out = []
    for c in corrects:
        d = len(str(int(c)))
        x = _ndigit(rng, d)
        while predicate(x):
            x = _ndigit(rng, d)
        out.append(x)
    return out


def _iso(rng) -> str:
    return f"{rng.randint(1900, 2099):04d}-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}"


def _not_iso(rng) -> str:
    y, m, d = rng.randint(1900, 2099), rng.randint(1, 12), rng.randint(1, 28)
    return rng.choice([
        f"{y}/{m:02d}/{d:02d}", f"{d:02d}-{m:02d}-{y}", f"{y}.{m:02d}.{d:02d}",
        f"{m:02d}/{d:02d}/{y}", f"{y}{m:02d}{d:02d}",
    ])


def build_suite(seed: int = 0) -> tuple:
    """Return ``(tasks, in_library_ids, out_of_library_ids)`` — deterministic."""
    rng = random.Random(seed * 7919 + 1)
    tasks: list = []

    def add(task_id, prompt, corrects, incorrects):
        tasks.append({"task_id": task_id, "prompt": prompt,
                      "examples": _examples(corrects, incorrects)})

    # --- IN-LIBRARY: the rule is expressible by a template ---------------------
    evens = [n for n in range(0, 201) if n % 2 == 0]
    odds = [n for n in range(0, 201) if n % 2 == 1]
    add("even", "Give an even number.", rng.sample(evens, 30), rng.sample(odds, 30))

    primes = [n for n in range(2, 301) if vs._is_prime(n)]
    comps = [n for n in range(2, 301) if not vs._is_prime(n)]
    add("prime", "Give a prime.", rng.sample(primes, 30), rng.sample(comps, 30))

    div3 = [n for n in range(0, 301) if n % 3 == 0]
    ndiv3 = [n for n in range(0, 301) if n % 3 != 0]
    add("divisible_by_3", "Give a multiple of 3.", rng.sample(div3, 30), rng.sample(ndiv3, 30))

    add("positive", "Give a positive number.",
        rng.sample(range(1, 101), 30), rng.sample(range(-100, 1), 30))

    four = list(range(1000, 10000))
    notfour = list(range(100, 1000)) + list(range(10000, 100000))
    add("four_digit", "Give a 4-digit number.", rng.sample(four, 30), rng.sample(notfour, 30))

    add("iso_date", "Give an ISO date.",
        [_iso(rng) for _ in range(30)], [_not_iso(rng) for _ in range(30)])

    in_ids = {t["task_id"] for t in tasks}

    # --- OUT-OF-LIBRARY: no template expresses the rule. Decoys are length-matched
    # to the correct answers (same digit count ⇒ same range/length), so no template
    # can separate them and the ONLY correct behaviour is to abstain. ---
    def _square(n):
        return math.isqrt(int(n)) ** 2 == int(n)

    def _has7(n):
        return "7" in str(int(n))

    pal = rng.sample([n for n in range(10, 2000) if _is_pal(n)], 30)
    add("palindrome", "Give a numeric palindrome.", pal,
        _matched_decoys(rng, pal, _is_pal))

    sq = rng.sample([n * n for n in range(4, 50)], 30)        # 2–4 digit squares
    add("perfect_square", "Give a perfect square.", sq,
        _matched_decoys(rng, sq, _square))

    seven = rng.sample([n for n in range(10, 2000) if _has7(n)], 30)
    add("contains_digit_7", "Give a number whose decimal contains a 7.", seven,
        _matched_decoys(rng, seven, _has7))

    out_ids = {t["task_id"] for t in tasks} - in_ids
    return tasks, in_ids, out_ids


def build_calibration_set(seed: int = 0, n: int = 400, k: int = 5) -> tuple:
    """Label-free confidence on a REAL (toy) solver — the correlation is emergent,
    not hand-set. For each chained-arithmetic task (varying length ⇒ varying
    difficulty) we draw ``k`` independent samples from a per-step-noisy solver and
    take the self-consistency agreement as the confidence; correctness is judged by
    the external oracle (the recomputed gold). Because a wrong run lands on a
    scattered value, agreement falls exactly when the solver is unreliable — so any
    selective-risk gain comes from the solver's behaviour, not the generator.

    Honest scope: a toy solver, not a frontier model — this demonstrates the
    metric machinery and that self-consistency tracks correctness here; it is not a
    capability claim. Returns ``(confidences, correct)``.
    """
    rng = random.Random(seed * 104729 + 3)
    confs: list = []
    correct: list = []
    for i in range(n):
        length = rng.choice([1, 2, 4, 8, 16, 32])
        task = hz.make_chain_task(length, seed=seed * 100003 + i * 31 + length)
        solver = hz.noisy_solver(0.2, seed=seed * 977 + i * 7 + 1)
        samples = [hz._final_int(solver(task)) for _ in range(k)]
        ans, conf = cal.self_consistency(samples)
        confs.append(conf)
        correct.append(str(ans) == str(task["gold"]))
    return confs, correct


def run_demo(seed: int = 0) -> dict:
    """Full demonstration + falsifiable invariants. ``ok`` is the CI gate."""
    tasks, in_ids, out_ids = build_suite(seed)
    with_mv = vs.evaluate_suite(tasks, in_library_ids=in_ids, out_of_library_ids=out_ids,
                                seed=seed, meta_verify=True)
    without_mv = vs.evaluate_suite(tasks, in_library_ids=in_ids, out_of_library_ids=out_ids,
                                   seed=seed, meta_verify=False)
    confs, correct = build_calibration_set(seed)
    calrep = cal.calibration_report(confs, correct, coverage=0.5)

    w_in, wo_in = with_mv["inLibrary"], without_mv["inLibrary"]
    w_out, wo_out = with_mv["outOfLibrary"], without_mv["outOfLibrary"]

    invariants = {
        # Capability: synthesised, validated verifiers generalise to held-out answers.
        "synthesized_verifiers_are_precise": w_in["meanTestPrecision"] >= 0.9,
        "synthesized_verifiers_catch_errors": w_in["meanTestRecall"] >= 0.7,
        "finds_checks_for_in_library_tasks": w_in["abstentionRate"] <= 0.2,
        # Safety: it abstains on EVERY task whose rule it cannot express (full
        # abstention — no slack), and even a hypothetical slip must not look good.
        "abstains_on_unverifiable_tasks": w_out["abstentionRate"] >= 1.0,
        "no_good_looking_wrong_gate": all((g["testPrecision"] or 0.0) < 0.9 for g in w_out["admittedGates"]),
        # The ablation — meta-verification is what makes synthesis trustworthy:
        "meta_verify_helps_in_library_precision": w_in["meanTestPrecision"] >= wo_in["meanTestPrecision"],
        "without_meta_falsely_admits_unverifiable": wo_out["falseAdmissionRate"] >= 0.5,
        "meta_verify_prevents_false_admission": w_out["falseAdmissionRate"] < wo_out["falseAdmissionRate"],
        # Competence where no verifier exists: confidence is usefully calibrated.
        "knows_what_it_doesnt_know": bool(calrep["selectiveBeatsBase"]),
    }
    return {
        "seed": seed,
        "withMetaVerify": {"inLibrary": w_in, "outOfLibrary": w_out},
        "withoutMetaVerify": {"inLibrary": wo_in, "outOfLibrary": wo_out},
        "calibration": calrep,
        "invariants": invariants,
        "ok": all(invariants.values()),
    }
