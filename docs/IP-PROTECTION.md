# Protecting Sophia — IP, provenance & priority strategy

> **Not legal advice.** This is an engineering/strategy memo. For patent and trademark
> decisions, consult a qualified IP attorney in your jurisdiction. The patent timing
> points below are **time-sensitive** — act on them early.

## TL;DR

The Sophia **code is already public and Apache-2.0 licensed**, which *grants everyone*
(including large companies) the right to use, modify, and commercialize it for free, and
includes an explicit patent license on what the code covers. You cannot claw that back
for already-published versions. So "protection" here means three realistic things:

1. **Credit / priority** — be the provable originator (this doc's main focus).
2. **Brand** — trademark the names you've reserved.
3. **Optional exclusivity going forward** — only via *future* relicensing and/or
   patents, and only if you decide that matters more than openness.

## The one fact that drives everything

| What you might protect | Tool | Reality for Sophia |
|---|---|---|
| The code | Copyright (automatic) | Already licensed away via Apache-2.0; irreversible for shipped versions |
| The method/algorithm | Patent **or** trade secret | Trade secret is gone (published). Patent window is **open but ticking** — see below |
| The name / brand | Trademark | Reserved in `TRADEMARK-POLICY.md`; can be *registered* |
| Being the originator | Defensive publication + timestamp | Cheap, strong, and aligned with the open project — **do this** |

## Patent timing (time-sensitive)

- **First public disclosure:** 2026-06-22 (first public commit; public repo + HF
  dataset + thesis site).
- **United States:** a **1-year grace period** after your *own* public disclosure to
  file (35 U.S.C. §102(b)(1)). A US patent is still on the table — but the clock runs
  from the disclosure date above.
- **Most other jurisdictions (EU, China, …):** **absolute novelty** — public disclosure
  *before* filing generally destroys patentability. Foreign rights may already be
  compromised. Ask an attorney about any exceptions before disclosing anything further.
- **Software-patent reality (US):** algorithms/abstract ideas are hard to patent
  (*Alice/Mayo*). A "verify-claim-then-abstain" method is borderline — patentable only
  if framed as a concrete technical process — and costs ~$10k–25k+ to prosecute and far
  more to enforce. For a solo author, weigh this honestly.
- **Existing patents are a hard obstacle.** A prior-art search (see
  [prior-art-survey.md](prior-art-survey.md)) found **granted US patents already covering
  the core idea**, with priority dates in **early 2023** — well before any Sophia filing:
  - **US 12,468,899** — "Hallucination prevention for natural language insights"
    (priority 2023-05-08).
  - **US 12,505,311** — "Hallucination detection and handling for an LLM-based
    domain-specific conversation system."
  Together with NeMo Guardrails (Oct 2023), RARR/ALCE/Self-RAG (2023), and Conformal
  Abstention/R-Tuning (2024), these make a **broad** method claim ("verify LLM claims
  against sources and abstain/block when unsupported") almost certainly
  anticipated/obvious. Only a **narrow** claim on a specific non-obvious mechanism could
  survive — and threading between these references is hard. **Realistic conclusion: do
  not bank on a patent over the pipeline. The defensive-publication path below secures
  your priority and freedom to operate at a fraction of the cost.**

**If exclusivity matters, talk to a patent attorney within the grace window and stop
disclosing new core IP until you have.** Otherwise, lean into defensive publication
(below), which *prevents others from patenting it over you* and cements your priority.

## Recommended plan (defensive-publication path)

This is the path that fits an open, already-public project. Checklist:

- [x] **Citable authorship + date record** — `CITATION.cff` (this repo).
- [x] **Archive metadata** — `.zenodo.json` (this repo) for a permanent DOI.
- [x] **Whitepaper** — `paper/sophia-whitepaper.md`, an arXiv-ready defensive
      publication of the method.
- [ ] **Mint a DOI** — connect the GitHub repo to **Zenodo** (zenodo.org → log in with
      GitHub → flip the `tomyimkc/sophia-agi` switch **on**), then cut a GitHub
      **Release**. Zenodo archives that release and mints a DOI automatically. Add the
      DOI badge to the README and the `doi:` field to `CITATION.cff`.
- [ ] **Post the whitepaper to arXiv** (cs.CL / cs.AI) — establishes a timestamped,
      citable prior-art record. (arXiv needs endorsement for a first submission in a
      category; alternatively post to **Zenodo** or **OSF**, which need no endorsement.)
- [ ] **Sign your commits** — see below; cryptographically binds the authorship record.
- [ ] **Register the trademark** — for "Sophia AGI" / "Wisdom Gate" / "Moral Gate" /
      "Conscience Kernel" with the **Hong Kong Intellectual Property Department (IPD)**
      (https://www.ipd.gov.hk) if the brand has commercial value; consider also the
      **USPTO** if you target the US market, and **WIPO's Madrid System** to extend one
      filing to multiple countries. You already assert common-law rights in
      `TRADEMARK-POLICY.md`; registration makes them enforceable.
- [x] **Fill the author identity** — author of record is **Yim Kin Cheong** (Hong Kong),
      set in `CITATION.cff`, `.zenodo.json`, and the whitepaper. Remaining: register a
      free **ORCID** (https://orcid.org) and uncomment the `orcid:` line in `CITATION.cff`.

## Optional: exclusivity going forward

You cannot un-license shipped Apache-2.0 versions, but you control the *direction*:

- **Relicense future versions** to a copyleft or source-available license to limit
  commercial free-riding on new work — e.g. **AGPL-3.0** (network-copyleft: competitors
  must open-source their modifications) or a **source-available** license (BSL/Elastic:
  blocks commercial competition for a set period, then converts to open). This is a
  **strategic decision with real community trade-offs** — it is intentionally left to you
  and not applied here.
- Keep genuinely novel *unpublished* work as a **trade secret** until a patent decision.

## Honest take on "is this unique?"

The *general* idea — ground LLM outputs in sources, verify attributions, abstain instead
of fabricate — is a **crowded, active research area** (selective prediction, attributed
QA, Self-RAG/RARR/ALCE, SelfCheckGPT/FacTool/FActScore, guardrail/verifier systems).
That crowdedness **weakens a patent** (examiners will find references) but does **not**
diminish your real contribution: the specific humanities-attribution framing, the
bilingual corpus, the pre-registered no-overclaim measurement protocol with an
independent judge audit, and the fail-closed governance contract. Your durable moat is
**execution + dataset + brand + being the recognized originator**, which the
defensive-publication path protects directly. The full prior-art survey backing this
section — named systems, benchmarks, commercial products, and the granted patents — is in
[prior-art-survey.md](prior-art-survey.md).

## Signing your commits (provenance)

Signed commits cryptographically tie each change to your key, hardening the
authorship/date record. SSH signing is the simplest:

```bash
# Use an existing SSH key (or generate one: ssh-keygen -t ed25519 -C "you@example.com")
git config --global gpg.format ssh
git config --global user.signingkey ~/.ssh/id_ed25519.pub
git config --global commit.gpgsign true      # sign every commit
git config --global tag.gpgsign true         # sign every tag (important for releases)
```

Then add the **same** public key to GitHub under *Settings → SSH and GPG keys → New SSH
key → key type: **Signing Key***, so commits show as **Verified**. For release tags:

```bash
git tag -s v0.9.0 -m "Sophia v0.9.0"
git push origin v0.9.0
```

(Prefer GPG? `gpg --full-generate-key`, then `git config --global user.signingkey <KEYID>`
and `git config --global gpg.format openpgp`.)
