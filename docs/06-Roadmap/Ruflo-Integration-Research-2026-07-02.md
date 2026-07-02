# Ruflo → Sophia-AGI Integration Research (2026-07-02)

> **Status: research / brainstorm — candidateOnly, illustrative.** Nothing in this document
> is a measured result. Every ruflo performance figure quoted below is **vendor-reported and
> unverified** (their own benchmarks; treat with the same skepticism this repo applies to its
> own numbers). `canClaimAGI: false`. Any adoption below must clear the usual gates
> (IEC pre-registration, `claim_gate`, W2 promotion, protected suites) before a single
> public claim changes.

## 0. Executive summary

[ruvnet/ruflo](https://github.com/ruvnet/ruflo) (ex-claude-flow, MIT, TypeScript) is an
"agent meta-harness" for Claude Code: swarm orchestration, persistent vector memory
(AgentDB/HNSW), trajectory self-learning (SONA + ReasoningBank), a hook/worker automation
layer, a GOAP A\* planner, and signed-witness install verification.

The load-bearing finding of this research: **Sophia already has verifier-gated equivalents
of roughly half of ruflo's architecture** (subagent fan-out, councils, gated durable memory,
trajectory packs, a routing seam, signed-hash artifacts) — but three of ruflo's mechanisms
are genuinely missing here and map cleanly onto named open work items:

1. **Experience retrieval at plan time** (ReasoningBank-style) — Sophia *records* gated
   trajectories but never *retrieves* them into the next plan. Highest-leverage gap.
2. **Hook-driven learning-signal capture** — trajectory-pack building is manual; ruflo makes
   capture ambient. Direct efficiency win for the training-data flywheel.
3. **A learned router head** behind the existing `swarm_router.decide` seam — the seam was
   built for exactly this and is still hand-authored.

The integration thesis, in one line: **adopt ruflo's *mechanisms*, never its *claims*, and
run every one of them through the measurement contract** — producing something neither
project has: a self-improving agent swarm whose improvement is *measured, gated, and
honest-or-NO-GO*. Ruflo optimizes throughput; Sophia optimizes epistemic trust; the
combination is a trustworthy swarm.

---

## 1. What ruflo is (research notes)

### 1.1 Thesis

Ruflo's implicit thesis: *the model is no longer the bottleneck — coordination, memory, and
learned reuse are.* A single Claude Code session is stateless, single-threaded, and forgets
everything it learned about your project. Ruflo wraps it in an execution layer that
(a) fans work out to specialized agents, (b) persists what worked, and (c) routes future
work using what it learned. "After `init`, just use Claude Code normally — hooks route
tasks, learn from successful patterns, and coordinate agents in the background."

### 1.2 Architecture (as documented by the project)

| Layer | Mechanism | Vendor-reported numbers (unverified) |
|---|---|---|
| Orchestration | MCP server + ~17–27 lifecycle hooks + 12 auto-triggered background workers (audit, testgaps, memory-prune, pattern-extract, cost-track…) | — |
| Swarms | hierarchical (queen-led) / mesh / adaptive topologies; Raft / Byzantine / Gossip consensus | "~89% task-routing accuracy" |
| Agents | 100+ role agents (coder, tester, security, architect…) | — |
| Memory | AgentDB vector store, HNSW-indexed; hybrid search + graph hops + diversity ranking (`ruflo-rag-memory`); cross-session persistence (`ruflo-rvf`) | "1.9×–4.7× vs brute force above ~5k records, recall@10 ≈ 0.99; 2,345 writes/s; p99 4.9 ms k-hop" |
| Learning | SONA (trajectory → pattern extraction → learned routing, "<0.05 ms adaptation") + ReasoningBank (successful decision paths stored, HNSW-retrieved into future tasks) | — |
| Planning | GOAP: goals → preconditions/effects action models → A\* shortest viable path → replan-on-failure; every action node is an MCP tool call | — |
| Graph intelligence (v3.10) | agent state as a graph: personalized PageRank, dynamic mincut, temporal centrality, witness-chain divergence (failure blast-radius) | — |
| Security | zero-trust federation (mTLS + ed25519 identity, behavioral trust score `0.4·success+0.2·uptime+0.2·threat+0.2·integrity`), AIDefence (injection blocking, 14-type PII), signed witness manifests + `ruflo verify` | — |
| Meta | MetaHarness: 1–100 readiness score for an agent setup, config security scan, regression snapshots, `eject` | "1.3×–1953× vs LangGraph/AutoGen/CrewAI" |

### 1.3 Honest appraisal (Sophia lens)

- The headline benchmark matrix (up to 1953×) is the project's own, methodology largely
  undisclosed — exactly the class of number this repo's linter exists to prevent. Third-party
  coverage says the same ("recommend independent verification").
- SONA/ReasoningBank learning is **ungated**: patterns extracted from trajectories are
  written to memory and reused with no independent verifier in the loop. In Sophia terms
  that is unlicensed self-training on unlabelled self-generated data — the exact failure
  mode `poison_resistant_ingestion.py` / the W2 gate / decontamination exist to prevent.
- The 210-tool MCP surface and 100-agent roster are breadth-first; Sophia's ~80 `sophia_*`
  tools are already at the edge of discoverability. Tool count is a cost, not a feature.
- What *is* genuinely strong: the ergonomics (ambient hooks, zero-config capture), the
  memory data structures (HNSW at scale, hybrid+graph retrieval), replanning as a
  first-class operation, and witness-signed supply-chain verification.

---

## 2. Side-by-side: ruflo mechanism ↔ Sophia counterpart

| Ruflo | Sophia today | Verdict |
|---|---|---|
| Queen-led swarm, agent fan-out | `agent/subagent.py` (least-privilege tool scoping, per-child budgets, fail-closed reduce) + `agent/long_horizon.py` (durable resumable task tree) | **Have it, better-governed.** Keep ours. |
| Consensus (Raft/Byzantine/Gossip) | ≥2-judge-family gate (κ/AC1), councils (`council_deliberate`, N_eff, divergence-aware abstention), `gateway/consensus.py` | **Have the epistemic version.** Their consensus is infra-level; ours is evidence-level. Minor borrow only (§3.6). |
| AgentDB persistent memory | `agent/gated_memory.py` (SQLite accepted/quarantine), `layered_memory.py`, OKF belief graph, committed RAG index | **Have the write path; missing the read-at-plan-time path** (§3.1). |
| HNSW vector search | brute-force npz in `agent/vector_store.py` | **Gap, but not yet binding** — their own data says crossover ≈ 5k records; our index is smaller. Adopt when trace memory grows (§3.3). |
| SONA / ReasoningBank trajectory learning | `build_trajectory_pack.py` (A1), `trace_distill.py` (fail→fix DPO), recovery-memory in `long_horizon` (within-run only) | **Half-built: we mine trajectories for *training*, never for *inference-time reuse*** (§3.1, §3.2). |
| Learned task routing ("89%") | `swarm_router.decide` — hand-authored v1, trained head documented as drop-in; `claim_router`, `continual_qa_controller` | **The seam exists, empty.** Highest-confidence adoption (§3.4). |
| GOAP A\* + replan-on-failure | static dependency tree in `long_horizon`; MCTS shims (`planner_mcts`, `verification_mcts`) | **Gap** (§3.5). |
| Hooks + 12 background workers | 3 Claude-Code hooks (session_start, git-write guard, skill-capture nudge); CI drift gates | **Gap in capture/automation breadth** (§3.2, §3.7). |
| Witness verification (ed25519) | sha256 everywhere (ledger rule: no claim without artifact + sha256), git-crypt | **Cheap upgrade** (§3.8). |
| Zero-trust federation, trust scoring | `swarm_trust_boundary.py`, BLP/Biba labels, Biba taint on external output; cross-*session* coordination via git + handover docs | **Have the intra-run version; inter-session is manual** (§3.9). |
| MetaHarness readiness score | `sophia-security-audit` skill (3 local gates) | **Cheap upgrade** (§3.10). |
| 210 MCP tools / 100 agents | ~80 `sophia_*` tools, 11 skills | **Do not adopt.** Consolidate instead (§4). |

---

## 3. Integration ideas, ranked

Each idea names the concrete seam, the Sophia-idiom constraint that makes it safe, and the
open item it advances. Effort: S < 1 day, M ≈ 1–3 days, L ≈ 1–2 weeks (illustrative).

### Tier 1 — do these first (high payoff, seams already exist)

#### 3.1 Gated ReasoningBank: retrieve past *accepted* trajectories at plan time — **M**

The single biggest ruflo idea Sophia lacks. Today a `long_horizon` run starts blind: the
recovery memory is within-run only, and the A1 trajectory packs flow exclusively toward
SFT/DPO. Add the *inference-time* read path:

- **Write** (mostly exists): every run-case / long-horizon trajectory that clears the A1
  acceptance gates gets embedded (existing `local-hash-v1` backend — deterministic, offline)
  and stored in `gated_memory` with task-signature, tool-scope, outcome, and provenance.
  Only `accepted`-table entries are ever retrievable; `quarantine` is invisible.
- **Read**: at plan time, `subagent`/`long_horizon` query top-k similar past trajectories
  and inject them as *hints* into the planning prompt — the same mechanism as the existing
  failure-signature recovery memory, generalized across runs. Retrieved patterns are
  suggestions, never authority: every step still passes its verifier, so a stale/wrong
  pattern costs one failed attempt, not a corrupted belief.
- **Measure it like everything else**: pre-register a spec (primary metric: verified-steps
  per solved task, or steps-to-done on a fixed offline task pack; ≥3 seeds; memory-on vs
  memory-off arms). If it doesn't clear the gate, it ships disabled.

Why this is the top pick: it is the "self-learning agent" payoff without weight updates —
cheaper than any GPU run, fully offline-testable, reversible, and it *compounds*: the same
store improves both inference (hints) and training (richer packs). It is also the natural
Phase-2 payload for the in-flight **codebase-memory-mcp** work (Phase-0 controls are the
security prerequisite this repo already insisted on building first).

#### 3.2 Ambient trajectory capture via hooks (ruflo's real ergonomic win) — **S/M**

Ruflo's hooks make learning-signal capture *ambient*; here it is manual
(`build_trajectory_pack.py` runs when someone remembers). Extend `.claude/hooks/`:

- **Stop / PostToolUse hook**: append structured events (task, tool calls, verdicts, final
  state) to an append-only session-trace JSONL under `agent/memory/` — same idiom as the
  existing harness run traces, now covering Claude-Code sessions themselves.
- **Nightly worker** (see §3.7) runs the A1 gates over new traces and folds survivors into
  the trajectory pack + the §3.1 memory store.
- Also close the loop the bootstrap already nags about: a **PreToolUse commit-guard hook**
  that runs `make claim-check` advisory-style before `git commit`, like the existing
  git-write guard. (The guardrail exists as prose; make it a hook.)

Zero model risk, pure plumbing, and it multiplies the data the flywheel sees — the
sessions doing the most interesting work in this repo are Claude-Code sessions, and today
their trajectories evaporate at handover.

#### 3.3 Memory scale-path: HNSW + graph-hop hybrid retrieval — **S now, M later**

- **Now (S)**: nothing — ruflo's own benchmark says ANN loses below ~5k records, and the
  curated index is below that. Record the crossover as a tripwire (assert in CI on index
  size) so the upgrade triggers itself.
- **When traces + codebase-memory grow past it (M)**: add an `hnswlib` backend behind
  `vector_store` with a deterministic build (fixed seed/M/ef) and the committed-manifest
  verify pattern `build_rag_index.py` already uses. Ruflo's `ruflo-rag-memory` "hybrid
  search + graph hops + diversity ranking" is architecturally what `ai_search` (RRF) + OKF
  edges already are — the borrow is to make **belief-graph hops a retrieval stage**: dense
  hit → follow OKF supports/contradicts edges 1 hop → rerank with min-over-chain confidence.
  That is retrieval that *inherits provenance*, which ruflo cannot do.

#### 3.4 Train the router head that `swarm_router` was built for — **M (no GPU needed to start)**

Ruflo's "89% routing accuracy" is unverifiable, but the mechanism is right and this repo
pre-built the exact seam: `SwarmRouter.decide(task) → SwarmPlan` with the trained head
documented as a drop-in that cannot change the plan contract. Concretely:

1. Mine `(task-features → chosen plan → outcome)` tuples from run traces (the §3.2 capture
   makes this dataset grow on its own).
2. Train a small classifier/regressor head (CPU-trainable; logistic head or tiny GBM over
   `query_understanding` features) predicting solo-vs-fanout, team, k.
3. Gate it like an adapter: `evaluate_update()` with protected suites + answerable-coverage
   protected, ablation vs the hand-authored v1 (the `mac-mlx-bench.yml suite=claim-router-ablation`
   lane is the template — same vehicle, different router).

This is "SONA-style learned routing" rebuilt as a *promotable, reversible, measured*
component. It is also the cheapest "model to train" on the docket — worth doing before the
next big adapter run because it improves every future run's efficiency.

### Tier 2 — architecture upgrades (adopt selectively)

#### 3.5 GOAP-style replanning in `long_horizon` — **M/L**

Give `SubtaskNode`s optional `preconditions`/`effects` over a small typed state vocabulary
(artifact-exists, gate-passed, budget-remaining, resource-claimed). On node failure, instead
of only retry-with-hints: re-run a bounded A\*/best-first search (the `planner_mcts` shim is
the natural home) from *current* state to the goal and splice the new subtree — replanning as
a first-class, auditable event in the ledger. Two Sophia-native bonuses ruflo doesn't have:
each replan event is (a) a scoreable trajectory branch point → **automatic DPO pairs**
(abandoned plan = rejected, successful replan = chosen), and (b) ENFORCED `resourceManifest`
(A6) becomes a *precondition*, so plans that would violate resource claims are unreachable
in search rather than failed at runtime.

#### 3.6 Name the judge-farm consensus policy explicitly — **S**

Sophia's ≥2-family judge gate *is* a consensus mechanism; make it a declared, versioned
policy object (quorum size, family-diversity requirement, tie/abstain rule, κ/AC1 health
threshold) in `gateway/consensus.py` rather than convention spread across tools. Borrowing
the *vocabulary* (quorum, fault assumption) makes the m3-3family judge-farm run easier to
pre-register and makes "judge collusion/correlation" a first-class failure mode with a
ledger row. No behavior change required to start.

#### 3.7 Background workers as scheduled, budget-capped Actions — **S each**

Ruflo's 12 workers, translated to this repo's infra (cron workflows on the self-hosted
runners, hard token/time budgets, advisory-only output):

| Worker (borrowed) | Sophia implementation |
|---|---|
| pattern-extract | nightly A1 gates over new traces → pack + §3.1 store |
| memory-prune | scheduled `decay_okf` / `forgetting_audit` run (exists — schedule it) |
| testgaps | coverage-diff report on PRs, advisory comment |
| audit | `sophia-security-audit` skill body as a cron lane |
| cost-track | RunPod spend snapshot vs budget → ledger append (extends wisdom-gpu-prebaked discipline) |
| docs-sync | already exists as drift gates — the one worker category where Sophia is ahead |

#### 3.8 Witness-sign the evidence chain — **S**

The ledger rule is already "no claim without artifact + sha256"; upgrade the highest-value
artifacts (`published-results.json`, promotion reports, adapter checkpoints) to **ed25519
signatures with a committed public key** (`ruflo verify` pattern). Cost: a ~50-line signer +
verifier in `tools/`, a CI check, one key in the same secret channel as the git-crypt key.
Payoff: third parties can verify the evidence chain without trusting the repo history — a
strict strengthening of the thing this repo stakes its credibility on.

#### 3.9 Machine-enforced session coordination (mini-federation) — **M**

The 5+ concurrent sessions coordinate via `SESSION-COORDINATION.md` + handover docs — prose
locks. Ruflo's federation is overkill (mTLS between your own sessions buys little), but the
*shape* is right: make GPU/branch claims machine-readable and enforced. The MCP contract
substrate already has `sophia_enqueue_task` / `sophia_next_task`; add
`sophia_claim_resource(resource, ttl)` backed by a committed claims file with heartbeat
timestamps. The Spark one-GPU-job invariant stops being a skill-prose rule and becomes a
fail-closed check the launcher tools consult. (The trust-*score* part of ruflo federation:
skip — sessions here are all root-of-trust-equal.)

#### 3.10 MetaHarness-style readiness audit — **S**

Extend `sophia-security-audit` with a deterministic **agent-readiness report**: MCP tool
count/risk-table coverage, gateway env-flags state, skills-index freshness, hook health,
unlock state, claims-file liveness (§3.9). Output a scorecard, not a 1–100 vanity number —
each line pass/fail with the fix. Cheap, and it operationalizes what the session bootstrap
currently prints as prose.

### Anti-adoptions (explicit, so nobody re-litigates)

- **The 210-tool / 100-agent surface.** Breadth is a discoverability tax. See §4.
- **Ungated SONA-style self-learning.** Pattern reuse without a verifier in the loop is the
  anti-thesis. Everything in §3.1/§3.4 exists precisely to keep the loop gated.
- **Vendor benchmark framing.** No "N× faster" language anywhere near this repo's public
  copy; if a mechanism lands, its number comes from a pre-registered spec or it has no number.
- **Mesh/gossip runtime topologies.** Sophia's hierarchy-with-fail-closed-reduce is a
  deliberate auditability choice; peer-to-peer emergent coordination is unauditable by
  construction. Revisit only with a concrete task that hierarchy demonstrably cannot do.

---

## 4. Advice: MCP structure

1. **Hold the line on tool count.** ~80 tools already strains discoverability (ruflo "solves"
   this with hooks so users never see the tools — telling). Group the `sophia_*` surface into
   declared families (gate, virtue, belief, calibration, council, contract, external) and
   expose a `sophia_capabilities` index tool; deprecate near-duplicates (`*_check` vs
   `*_assess` vs `*_benchmark` variants could take a `mode` arg).
2. **The gateway is the moat — finish wiring it.** `boundary/audit/approval/gateway_wiring`
   is genuinely ahead of ruflo's security story (env-only identity, approval holds with
   digest-only storage, Biba taint on external output). Two additions from this research:
   route **memory writes** (§3.1) through `governed()` with taint labels — memory poisoning
   via retrieved trajectories becomes the top new attack surface once the read path exists —
   and add the §3.9 resource-claim tools to the risk table from day one.
3. **New tools to add (small, gated):** `sophia_memory_search` / `sophia_memory_store`
   (accepted-table only; store is medium-risk, approval-gated), `sophia_trajectory_record`
   (§3.2 capture endpoint), `sophia_route_task` (read-only `SwarmPlan` preview — lets any
   client benefit from §3.4), `sophia_claim_resource` (§3.9).
4. **Land codebase-memory-mcp Phase 0 → 1 as planned** (it is the open next-step in the
   2026-07-02 handover); design its index API so §3.1's trajectory store is a second backend
   of the same interface rather than a parallel system.

## 5. Advice: skills

1. **The plaintext/encrypted split and auto-trigger prose are working** — keep the
   convention. The gap is *feedback*: nothing measures whether a skill firing actually
   prevented waste. Add lightweight **skill efficacy tracking**: the §3.2 session traces
   record skill invocations; a cron worker correlates them with outcomes (green CI, no
   ledger "trap rediscovered" entries). `gateway/skill_flywheel.py` is the natural home —
   the flywheel concept exists in-code but has no data source yet.
2. **Fold ruflo-shaped playbooks into existing skills, not new ones**: replan-on-failure
   guidance → `rlvr-harness-traps`/long-horizon docs; resource claims → `spark-cluster-ops`;
   readiness audit → `sophia-security-audit`. Eleven skills is near the right number;
   twenty is not.
3. **Portable-skill opportunity**: package the measurement contract itself
   (`skills/portable/`, like `sophia-source-discipline`) as a skill other projects can
   install — "no-overclaim for agent benchmarks" is exactly what the ruflo ecosystem lacks,
   and it is this repo's most differentiated export.

## 6. Advice: the model you are about to train

Applies to the near-term docket (v7 QAT, the next RLVR arm, Sophia-Wisdom v-next):

1. **Train the router head first (§3.4).** It is CPU-cheap, offline-gateable, and improves
   the *efficiency of every subsequent run* — data generation, judging, and long-horizon
   evals all get cheaper before the expensive training starts.
2. **Use swarm parallelism for data diversity, not speed.** Ruflo's real gift to a training
   pipeline is many *differently-prompted* agents generating trajectories in parallel. The
   generators already exist (`selfplay_task_forge` A5, `train_council_teacher` A3 two-stage
   distillation, A1 packs, fail→fix DPO from `trace_distill`, §3.1's store as a curriculum
   source ranked by verified-success). Keep the invariants: decontamination fails closed,
   `lint_training_rows`, provenance-weighted curriculum (R6), holdout enforcement.
3. **Prefer memory to weights where memory suffices.** §3.1 gives procedural improvement
   with zero catastrophic-forgetting risk and zero GPU cost. Pre-register the comparison:
   *retrieval-hints-only vs fine-tuned vs both* on the same task pack. If hints capture most
   of the uplift, the training budget moves to what only weights can learn (calibration,
   source-discipline style) — and the "both" arm tests whether trained models use retrieved
   hints better.
4. **Carry the RLVR forensics lessons forward mechanically**: passAt1/VSC load-bearing
   (never meanReward), seed-stamped report paths, seed reaching the trainer config,
   ≥3 seeds, pre-registered thresholds (`PREREGISTRATION-NEXT-ARM.md` is the template).
   Any §3.1/§3.4 uplift claim uses the same machinery — memory-on vs memory-off is just
   another adapter-vs-baseline comparison to the gate.
5. **Trajectory packs are about to get much bigger (§3.2) — cap by quality, not recency.**
   Rank by A1 gate margins + §3.1 retrieval-usefulness so the pack composition is itself
   an audited, deterministic artifact.

## 7. Suggested adoption order

| # | Item | Effort | Unblocks / advances |
|---|---|---|---|
| 1 | §3.2 hook capture + commit-guard hook | S/M | flywheel data; every later item |
| 2 | codebase-memory-mcp Phase 0→1 (already planned) | M | §3.1 vehicle |
| 3 | §3.1 gated ReasoningBank (read path + pre-registered eval) | M | headline efficiency win |
| 4 | §3.4 router head + ablation lane | M | cheaper runs; first "trained model" win |
| 5 | §3.8 ed25519 witness signing | S | evidence-chain credibility |
| 6 | §3.10 readiness audit + §3.6 consensus policy object | S | ops hygiene; judge-farm prereg |
| 7 | §3.9 machine-enforced resource claims | M | cluster safety |
| 8 | §3.5 GOAP replanning | M/L | long-horizon capability + free DPO pairs |
| 9 | §3.3 HNSW backend | M | only when the tripwire fires |

**Risks to watch:** memory poisoning via the new read path (mitigate: accepted-table only,
taint labels, hints-not-authority); retrieval staleness after repo drift (mitigate: TTL +
re-verify on retrieval); router head overfitting to historical task mix (mitigate: protected
suites + periodic hand-authored-v1 shadow comparison); scope creep toward ruflo's breadth
(mitigate: the anti-adoption list above is part of this doc on purpose).

## 8. Sources

- https://github.com/ruvnet/ruflo (README, wiki: Hooks & Workers, Intelligence Pipeline,
  Witness Verification, Benchmarks)
- https://www.augmentcode.com/learn/ruflo-v3-10-graph-intelligence-claude-code (v3.10 graph
  intelligence; independent caveats on vendor benchmarks)
- https://www.augmentcode.com/learn/ruflo-claude-code-multi-agent-orchestration
- https://medium.com/data-science-in-your-pocket/ruflo-multi-agent-ai-orchestration-for-claude-code-e5343c33f062
- Internal: `SESSION-HANDOVER-2026-07-02.md`, `agent/` survey (subagent, long_horizon,
  swarm_router, gated_memory, layered_memory), `sophia_mcp/` gateway,
  `tools/build_trajectory_pack.py` (A1), `agi-proof/failure-ledger.md`.
