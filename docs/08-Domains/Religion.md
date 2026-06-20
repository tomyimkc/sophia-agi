# Religion domain (active)

## Source discipline + council panel

Philosophy-style provenance plus **full council panel** debate mode:

- Who **authored** or **compiled** scripture?
- **Tradition boundary** — Christianity, Buddhism, Islam, Daoist, Confucian ritual
- **Theological vs historical** — do not collapse
- **Pop spirituality myths** — label misconception / myth
- **Figure source seats** — Jesus tradition witness and Buddhist dharma witness,
  grounded in documented textual traditions rather than impersonation

## Data center

- **Records:** `data/religion_concepts.json`
- **Figure seats:** `data/religion_council_figures.json`
- **Hub training:** `training/examples/019-religion-source-discipline-hub.json`
- **Council spec:** [Religion-Council-Debate-Mode.md](Religion-Council-Debate-Mode.md)
- **Figure council:** [Religion-Figure-Council.md](Religion-Figure-Council.md)
- **Benchmark:** `tests/benchmark-religion.json` (5 cases)

## Example traps

| Trap | Record | Requirement |
|------|--------|-------------|
| Gospel of Matthew | `gospel_matthew` | Council + Christianity tradition |
| Ancestor veneration | `confucian_ancestor_veneration` | Split philosophy vs ritual |
| Dao De Jing register | `dao_de_jing_daoist_scripture` | Philosophy vs religion |
| Early Islam sensitive | `islam_early_history` | Council + careful scholarly tone |
| Nirvana = heaven | `nirvana_pop_heaven` | Council + Buddhism + myth label |

## Status

**Active** — see [Source-Discipline-Methodology.md](Source-Discipline-Methodology.md).
