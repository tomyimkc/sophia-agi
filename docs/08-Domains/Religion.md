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
- **Benchmark:** `tests/benchmark-religion.json` (6 cases)

## Example traps

| Trap | Record | Requirement |
|------|--------|-------------|
| Gospel of Matthew | `gospel_matthew` | Council + Christianity tradition |
| Ancestor veneration | `confucian_ancestor_veneration` | Split philosophy vs ritual |
| Dao De Jing register | `dao_de_jing_daoist_scripture` | Philosophy vs religion |
| Early Islam sensitive | `islam_early_history` | Council + careful scholarly tone |
| Nirvana = heaven | `nirvana_pop_heaven` | Council + Buddhism + myth label |
| Hadith sect boundary | `hadith_canonical_collections` | Council + Islam + sect boundaries |

## Sensitive-topic handling (scripture attribution with sect boundaries)

Scripture records often carry **within-tradition sect boundaries** that must not be
flattened. `hadith_canonical_collections` (GF-30) is the worked example: the Sunni
canonical six books (Kutub al-Sittah, e.g. *Sahih al-Bukhari*, *Sahih Muslim*) and
the Shia four books (e.g. *al-Kafi*) are **distinct collections**, and hadith are
**compiled reports** graded by chains of transmission (isnad) — not the Quran.

Handling rules for this and similar sensitive records:

- **Seat the council, name the sects.** Answer in `council_debate` mode with named
  sect voices (Sunni / Shia hadith-science seats) rather than one flattened voice.
- **Do not merge sect canons** with each other, and **do not collapse** a graded
  report corpus (hadith) into its scripture (the Quran). These are encoded in the
  record's `doNotMergeWith` and `sectBoundaries` fields.
- **Separate registers.** Keep scholarly grading (sahih / da'if) distinct from
  theological authority claims; keep historical-critical reconstruction distinct
  from the devotional/theological voice.
- **Tone.** Use tradition-respecting, scholarly framing; avoid a partisan sectarian
  position. The benchmark case `hadith_sect_boundary_council` enforces council
  format + Islam tradition context + sensitive handling.

## Status

**Active** — see [Source-Discipline-Methodology.md](Source-Discipline-Methodology.md).
