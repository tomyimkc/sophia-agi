# Monetization Ops & Legal Checklist — "before you charge a dollar"

The Tier-1 honest monetization streams (Sponsors, Services, Education) are safe to *prepare*
in the repo, but **do not take payment from anyone until the boxes below are checked.** This
is operational guidance, not legal advice — consult a qualified professional for your
jurisdiction.

## A. Business & tax substrate
- [ ] Legal entity formed (sole-proprietor or LLC/Ltd as appropriate).
- [ ] Business bank account separate from personal.
- [ ] Tax registration + how you'll handle income tax and **VAT/GST/sales tax** on digital
      goods (Gumroad/Paddle act as merchant-of-record and handle this; raw Stripe does not).
- [ ] Invoicing process for services.

## B. Public legal docs (publish before checkout goes live)
- [ ] **Terms of Service** for any paid product/API.
- [ ] **Privacy Policy** (covers sponsor names, emails, analytics; GDPR/CCPA rights incl.
      deletion).
- [ ] **Refund Policy** (digital-goods + services).
- [ ] License clarity: Apache-2.0 core stays Apache-2.0; what (if anything) paid assets add.

## C. Claims compliance (the project's whole point)
- [ ] `python tools/lint_claims.py` passes on all sales copy.
- [ ] No sales/marketing copy says "0% fabrication", "100%", "safe to ship", "first/only/AGI",
      or "guarantee." Use the validated, scoped numbers only.
- [ ] Every result cited in sales copy maps to a `agi-proof/failure-ledger.md` entry.

## D. Services (consulting) guardrails
- [ ] Written **SOW** template: deliverables, *measured* target, residual error, timeline, price.
- [ ] **Limitation-of-liability** clause (cap at fees paid) + **indemnification** + "no
      guarantee of specific accuracy outcomes."
- [ ] Never promise "0% fabrication in your stack" — promise a *measured reduction* on a
      baseline you run first.
- [ ] Client data handling / NDA where needed.

## E. Sponsors guardrails (already encoded in SPONSORS.md)
- [ ] Recognition only in `SPONSORS.md` / release notes — **never** in `training/` data.
- [ ] Sponsors influence **direction**, never **content/verdicts** — gate stays neutral.
- [ ] No benchmark result, "validated" label, or AGI claim is purchasable.

## F. API / model resale (currently OUT OF SCOPE — do not ship without these)
- [ ] **Do NOT resell Anthropic/Monica/OpenAI API access** under your own billing — it
      violates their terms and risks account bans. An inference API is only viable on a model
      **you host and own**, with honest performance claims.
- [ ] If/when self-hosting: rate limiting, usage metering, auth, abuse handling, status page.

## G. Payments & security
- [ ] Use a PCI-compliant processor (Stripe/Gumroad/Paddle) — never handle raw card data.
- [ ] **Do not let an autonomous agent execute billing/payment/deploy commands** (`--execute
      --approve` on payment or infra is out of scope for automation).
- [ ] Test the full purchase + refund flow with a real (small) transaction before launch.

## H. Secrets hygiene (standing)
- [ ] No keys in git (`.env`, `private/` stay gitignored). Rotate the OpenRouter, DeepSeek,
      and llmhub keys used in development.

## Recommended sequencing
1. **Services + Education first** — highest margin, honest (you sell expertise/method), lowest
   infra. Needs A, B, C, D.
2. **Sponsors next** — recurring base; needs A, B, C, E + the GitHub Sponsors profile.
3. **Everything else (marketplace, API, sponsored domains) — only after** the P2 credibility
   unlock (live backend + one independent third-party validation) lands, so you're selling a
   proven result, not a promise.
