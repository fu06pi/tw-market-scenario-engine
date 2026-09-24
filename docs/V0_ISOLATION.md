# V0 Boundary and Reference Policy

V0 is the existing **text-first ChatGPT + Google Sheets morning-hypothesis / post-close-review workflow**.

V1 Quant is a separate implementation that adds structured quantitative features, models, immutable snapshots, and statistical evaluation.

The separation exists to protect V0 as an independent historical research record — **not because V0 lacks value**.

## What V0 contributes

V0 contains qualitative information that may remain useful to V1 research:

- original written pre-open hypotheses;
- causal reasoning and catalyst narratives;
- Bull / Base / Bear style scenario descriptions;
- assumptions about heavyweight stocks and market breadth;
- invalidation conditions;
- post-close error analysis;
- model-rule evolution over time.

V1 should preserve this text-first research philosophy instead of replacing it with numeric outputs only.

## Hard write boundary

V1 must never:

- update the V0 Google Sheet;
- rewrite historical V0 hypotheses;
- change V0 review fields;
- delete or normalize V0 rows in place;
- use V0 as a mutable source of truth;
- silently import V0 post-close conclusions as morning features or training labels.

## Read-only reference is allowed

V1 may consult V0 in two ways.

### 1. Qualitative reference mode

For exploratory research, architecture work, hypothesis design, or model interpretation, V0 may be read directly in **read-only mode**.

Any V1 artifact derived from V0 should identify the relevant trading date / item so a human can trace the source.

### 2. Frozen research export mode

For reproducible quantitative analysis, V0 data should be copied one-way into a frozen research dataset:

```text
V0 Google Sheet (unchanged)
        |
        | read-only export
        v
Frozen V0 export
        |
        +-- source spreadsheet identifier
        +-- source row / prediction item
        +-- original timestamp
        +-- export timestamp
        +-- content hash
        v
Cleaning / schema mapping
        v
V1 research dataset
```

The export is a research snapshot. It does not synchronize changes back to V0.

## Time-causality rule

A V0 field may only be used for a V1 morning prediction if the information existed before the V1 pre-open cutoff.

Examples:

- morning V0 hypothesis written before cutoff: potentially valid reference;
- prior-day post-close model lesson: potentially valid reference for the next day;
- same-day post-close explanation: **not valid** as a same-day morning feature;
- text rewritten after the fact: not valid unless the original version is preserved.

## Narrative preservation

V1 predictions remain text-first in presentation even though evaluation becomes quantitative.

Each V1 prediction snapshot includes written hypotheses for:

- open;
- intraday path;
- close;
- breadth / heavyweight structure;
- Bull / Base / Bear scenarios;
- drivers and invalidation conditions.

The numerical fields exist to make those hypotheses testable — not to replace them.
