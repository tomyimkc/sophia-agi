# Domain expansion overview

Sophia AGI uses one **source discipline** method across multiple knowledge domains.

| Domain | Status | Focus |
|--------|--------|-------|
| Philosophy | **Active** | Text authorship, tradition boundaries |
| Psychology | Planned | Concept coinage, pop vs clinical misuse |
| History | Planned | Dated events, primary sources, mythologized past |
| Religion | Planned | Scripture attribution, sect boundaries |

## Shared schema

See `data/schema.json` and `data/domains.json`.

Every record should specify:

- `domain` + `recordType`
- `confidence` (attributed, compiled, legendary, disputed, …)
- `doNotAttributeTo` / `doNotMergeWith` where lineage errors are common

## Next step

Answer the questions in [Expansion-Questionnaire.md](Expansion-Questionnaire.md) so we can populate psychology, history, and religion with the right scope and tone.