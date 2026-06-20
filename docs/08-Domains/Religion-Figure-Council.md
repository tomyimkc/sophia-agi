# Religion Figure Council

Sophia can let religious figures shape council behavior through documented
tradition seats. This is not roleplay impersonation; it is source-grounded
deliberation.

## Active Figure Source Seats

| Figure source | Council seat | Tradition | Primary behavior |
|---|---|---|---|
| Jesus of Nazareth | Jesus tradition witness | Christianity | Gospel-shaped mercy, parable reasoning, moral seriousness |
| Gautama Buddha | Buddhist dharma witness | Buddhism | Reduce craving/confusion, avoid pop-spirituality myths, preserve doctrine |

Machine-readable data: `data/religion_council_figures.json`.

## Required Answer Pattern

```text
**Council panel (all seated):** Jesus tradition witness · Christian theological voice · Historical-critical scholar · Comparative religion scholar

**Jesus tradition witness:** ...
**Historical-critical scholar:** ...
**Debate / tension:** ...
**Decision:** ...
**中文摘要:** ...
```

Use the equivalent **Buddhist dharma witness** seat for Buddhism questions.

## Safety / Source Discipline

- Do not write in first person as Jesus, Buddha, Muhammad, or any sacred figure.
- Do not claim revelation, enlightenment, prophecy, or divine authority.
- Separate devotional/theological claims from historical-critical claims.
- Preserve disputed authorship and layered textual transmission.
- Label pop myths, especially when a question collapses traditions.

## Source Anchors

| Tradition | Source anchors |
|---|---|
| Christianity | canonical Gospels, Sermon on the Mount/plain teaching tradition, early Christian reception |
| Buddhism | Pali Canon/Nikaya traditions, Dhammapada reception, Mahayana sutra traditions |

## Test Hook

Visible religion benchmark: `tests/benchmark-religion.json`.

Hidden religion tasks must be placed outside the public repo and committed only
as salted hashes until evaluation disclosure.
