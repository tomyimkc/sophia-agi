# Sector Councils — Law, Finance, Economy

> Decision-support scaffolding and product-design reference only. These councils
> are **not** a substitute for a licensed lawyer, financial advisor, auditor,
> regulator, judge, or public authority. Every council carries an explicit
> human-authority boundary and a "not advice" stance.

Sophia models hard, contested domains as a **council of constrained seats** with
different incentives, rather than one monolithic expert — the same pattern as the
[Coding Council](./Coding-Council.md) and [Religion Figure Council](./Religion-Figure-Council.md).
Each seat is a *source-inspired perspective*, never an impersonation.

## How it works

- Data: `data/{law,financial,economy}_council_figures.json`
- Router: `agent/sector_council.py` (`load_council`, `route_council`, `format_council`)
- CLI: `python tools/sector_council.py <law|financial|economy> "<query>"`
- Tests: `tests/test_sector_councils.py`

A query is matched against each seat's `triggerTerms`. Non-core groups contribute
their top-N matched seats; **core "guardian" groups are always seated**. The
`workflow.defaultSeats` are force-included, and a generalist `fallbackSeat` is
added when no specialist matches. Each council ends with a `decisionContract` and
a `humanBoundary`.

```bash
python tools/sector_council.py law "gacha odds + refund policy for a HK + EU Steam launch"
python tools/sector_council.py financial "model 18-month runway and flag AML for Stripe payouts"
python tools/sector_council.py economy "simulate a minimum-wage rise: who gains, who loses?"
python tools/sector_council.py --list
```

## Law & Governance Council

Built from the AGI-law-agent role archetypes and the multi-agent council pattern.

| Group | Seats |
|---|---|
| Jurisprudence source seats | natural-law, legal-positivism, legal-realism, common-law, civil-code, Chinese legal-tradition |
| Legal role seats | client advocate, opposing counsel, judge clerk, policy analyst, compliance officer, contract negotiator, legal-aid navigator, legal-ops analyst |
| Practice-area seats | IP, privacy/data, tax, employment, corporate/M&A, litigation, criminal defense, immigration, consumer, platform/ToS, fintech/crypto, gaming/monetization, content/media |
| Jurisdiction seats | Hong Kong, Mainland China, EU, UK, US, Japan/Korea/Taiwan, cross-border/trade |
| Guardian seats (always) | ethics officer, citation auditor, jurisdiction & freshness detector, plain-language explainer, human-review gatekeeper |

Human boundary: never the final decision-maker on guilt, sentencing, custody,
immigration/asylum, detention, constitutional rights, or denial of public
services. High-stakes seats (criminal, immigration) force human escalation.

## Finance & Treasury Council

| Group | Seats |
|---|---|
| Finance source seats | value-investing, portfolio-theory, efficient-markets/factors, behavioral-finance, tail-risk, corporate-finance, accounting-integrity |
| Finance role seats | CFO, controller, treasurer, FP&A, internal auditor, forensic accountant (adversarial), tax strategist, investor advocate, short-seller skeptic (adversarial), credit-risk officer, valuation analyst |
| Market & asset seats | equities, fixed-income, FX, derivatives/hedging, crypto, private markets/VC, real-estate/commodities |
| Compliance seats | AML/KYC, sanctions, securities regulation, payments/card-scheme, consumer credit, GAAP/IFRS reporting, insurance/actuarial |
| Guardian seats (always) | numbers auditor, assumptions auditor, conflict-of-interest/fiduciary officer, plain-language explainer, human-review gatekeeper |

Human boundary: no fully automated denial of banking, insurance, credit, or
payment access; management/fiduciaries own risk acceptance and sign-off. Two
deliberate adversarial seats (forensic accountant, short-seller skeptic) attack
the rosy case.

## Economics & Policy Council

| Group | Seats |
|---|---|
| Economic-school source seats | classical, Keynesian, Austrian, monetarist, institutional/commons, behavioral, development/welfare, political-economy (critique), game-theory |
| Field-economist seats | macro, micro, labor, trade, monetary-policy, fiscal/public-finance, industrial/competition, environmental, regional/urban & digital |
| Method seats | econometrician (causal inference), market designer, forecaster/modeler, economic historian/data |
| Stakeholder-impact seats | consumer, worker, small-business/startup, vulnerable-groups/equity, future-generations/sustainability, jurisdiction-comparison |
| Guardian seats (always) | assumptions auditor, data & source auditor, uncertainty/humility, distributional-equity officer, value-judgment flagger, plain-language explainer |

Human boundary: economics informs but does not decide value tradeoffs; elected
officials and accountable institutions own normative choices. The value-judgment
flagger separates positive analysis ("what would happen") from normative choices
("what ought"), and forecasts are reported as ranges, not false precision.

## Design principles (shared)

- **Adversarial seats by design** — opposing counsel, short-seller skeptic,
  forensic accountant, political-economy critique — so the council attacks its
  own conclusions.
- **Guardians always seated** — citation/numbers auditing, ethics/equity,
  plain-language, and a human-review gatekeeper run on every query.
- **Source-inspired, never impersonation** — figure/school seats name a lineage
  and carry a `speakerBoundary`.
- **Human authority preserved** — high-stakes outputs (liberty, money access,
  rights, normative policy) are routed to accountable humans.
