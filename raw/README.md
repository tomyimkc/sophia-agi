# raw/ — librarian ingestion folder

Drop raw sources here (articles, notes, transcripts, primary-source excerpts). The
OKF wiki librarian reads them as **untrusted** text, extracts one provenance-stamped
page each, and writes a gated draft under `wiki/drafts/` (only if it passes the
source-discipline gate — schema-valid, no forbidden attribution, no lineage merge).

```bash
python tools/wiki_ingest.py raw/sample-oikeiosis.txt --provider mock   # offline dry-run
python tools/wiki_ingest.py raw/my-source.txt --provider glm           # real model
```

The librarian never edits files in `raw/`; it only synthesizes wiki pages from them.
Source drops are gitignored by default (may be large or copyrighted); this README and
the sample are kept. See `agent/wiki_librarian.py` and `skills/registry/wiki-maintenance.json`.
