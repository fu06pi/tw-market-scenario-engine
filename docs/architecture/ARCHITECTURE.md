# System Architecture

## Design principle

V1 is a quantitative research system with a **mandatory narrative layer**. It is separate from V0 operationally, while V0 remains available as a read-only historical reference.

## Layers

### 0. V0 reference lane

The original Google Sheets workflow is not modified by V1.

V1 may:

- read V0 for qualitative research;
- export selected V0 records into frozen, provenance-preserving research snapshots;
- cite V0 dates / items when they informed a V1 design decision.

V1 may not write back to V0.

### 1. Source layer

Official Taiwan market data should have highest priority for realized outcomes. Global overnight markets and corporate/macro events enrich the morning feature set.

### 2. Ingestion and validation

Every record must carry source, timestamp, timezone, and freshness metadata. Conflicting sources are resolved by source priority rather than convenience.

### 3. Feature layer

Features are computed only from information available by the pre-open cutoff.

### 4. Forecast layer

Run at least three quantitative systems in parallel:

- naive rule baseline;
- statistical baseline;
- richer quantitative model.

Alongside them, a narrative engine turns the same pre-open evidence into written:

- market thesis;
- open hypothesis;
- intraday hypothesis;
- close hypothesis;
- breadth / heavyweight hypothesis;
- Bull / Base / Bear scenario scripts;
- invalidation conditions.

### 5. Snapshot layer

The complete output — narrative + quantitative fields — is immutable and hashed.

### 6. Realization layer

Stores official open/high/low/close plus breadth, OTC, sector, and heavyweight contribution information.

### 7. Evaluator

Produces target-specific quantitative metrics and an error-attribution record. Narrative claims are evaluated through their pre-coded target labels rather than post-hoc semantic interpretation.

### 8. Model registry

A new model version is created only when the model-change gate is satisfied.

### 9. Presentation layer

User-facing output remains primarily written market scenarios, supported by probability, evidence, and track-record metrics. Quantification strengthens the text product; it does not replace it.
