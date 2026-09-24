# Data directory

This V1 repository contains **no copied V0 Google Sheet data** by default.

V0 remains an independent text-first research record. V1 may consult it read-only, and reproducible experiments may later use frozen V0 exports with provenance.

Future data layout:

```text
data/
  raw/         source snapshots, never edited
  processed/   normalized research tables
  manifests/   source/timestamp/hash metadata
  v0_exports/  optional frozen, one-way research exports
```

Raw and processed files are git-ignored by default. Commit only schemas, small examples, and reproducible transformation code.
