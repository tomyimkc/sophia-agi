# Confucian philosophy vs ritual religion — split when appropriate

## Rule (owner decision 2026-06-18)

**Split when appropriate** — not every mention of 儒家 requires two labels, but do split when:

- The question mixes **ethics / classics** with **祭祖 / temple rite / 禮教**
- The model would otherwise collapse 儒家哲學 and 儒家禮教 into one voice
- A religion-domain answer touches ritual obligation vs Analects moral teaching

## Do not force-split when

- The question is clearly philosophy-only (e.g. Ren 仁 in Analects)
- The question is clearly ritual-only (e.g. offering procedure in a family rite)
- A brief mention does not confuse lineages

## Data mapping

| Layer | Domain | Tradition ID |
|-------|--------|--------------|
| Moral philosophy, Analects | philosophy | `confucian` |
| Ancestor veneration, 禮教 | religion | `confucian_ritual` |

## Training metadata

Use `"splitWhenAppropriate": true` on religion examples that cross both layers.