# TW Market OHLC Forecast Engine — Codex handoff

**Current direction (2026-10-07):** quantitative TAIEX and contract-specific TX daytime OHLC forecasts first; frozen numerical output then feeds narrative reports with news/world/financial context. TEJ access is deferred.

Start with [CODEX_HANDOFF.md](CODEX_HANDOFF.md), [AGENTS.md](AGENTS.md), and the [OHLC architecture](docs/architecture/TW_OHLC_QUANT_ARCHITECTURE.md). First milestone: official-source/CSV ingestion, naive and ARIMA forecasters, reproducible walk-forward evaluation.

**Status:** architecture/handoff prepared; new OHLC models and measured accuracy are not yet established. Existing live tasks and historical records continue during development.

---

## Legacy scenario-engine documentation

The text below records the previous text-first system. New OHLC implementation follows the current direction above.

# TW Market Scenario Engine (V1 Quant)

A quantitative, auditable research system for generating **pre-open Taiwan equity market scenarios**, freezing them as immutable prediction snapshots, and validating them against **actual open / intraday / close** outcomes after the market closes.

> **V1 is separate from V0, not a replacement for it.** V0 remains the text-first ChatGPT + Google Sheets research log and may be used as a **read-only qualitative reference corpus**. V1 must never write back to or rewrite V0.

## Why V0 is preserved

V0 captures something V1 should not lose: the original project was built around **written market hypotheses and daily scenario narratives**, not only numeric forecasts. Those narrative records contain reasoning, assumptions, catalysts, invalidation logic, and post-close reflection that remain valuable for research.

V1 therefore uses a dual representation:

1. **Narrative layer** — mandatory pre-open written hypotheses and Bull / Base / Bear scripts.
2. **Quant layer** — explicit numeric targets, probabilities, features, baselines, and evaluation metrics.

The goal is not to turn the project into a purely numerical black box. The goal is to make the original text-first workflow measurable, reproducible, and testable.

## Research question

Can a structured scenario engine using overnight markets, Taiwan heavyweight structure, market breadth, event features, probabilistic calibration, and explicit narrative hypotheses produce better and better-calibrated daily Taiwan market scenarios than simple baselines?

## Core principle

The system never edits a prediction after the cutoff. Each morning prediction is frozen with a timestamp, model version, feature version, data cutoff, and SHA-256 hash. Evaluation later compares the **same pre-open narrative hypothesis and quantitative targets** independently with:

1. Actual open
2. Actual intraday path
3. Actual close

This prevents hindsight rewriting.

## V1 architecture

```text
V0 text-first history (READ ONLY)
          |
          | reference / frozen export with provenance
          v
External market + event data
          |
          v
Ingestion + validation
          |
          v
Feature store / feature frame
          |
          +----------------------+----------------------+
          |                      |                      |
          v                      v                      v
Simple baselines          Quant models          Narrative engine
          |                      |                      |
          +----------------------+----------+-----------+
                                            v
                              Immutable prediction snapshot
                              - written hypotheses
                              - scenario probabilities
                              - numeric targets
                              - source/version metadata
                                            |
                                            v
                              Open / intraday / close realization
                                            |
                                            v
                                    Quant evaluator
                                            |
                              +-------------+-------------+
                              |                           |
                              v                           v
                       KPI / calibration           Error attribution
                              |                           |
                              +-------------+-------------+
                                            v
                                  Model registry / research log
```

## Prediction targets

V1 explicitly separates targets that were previously mixed together:

- `open_direction`: gap/open direction
- `close_direction`: close-to-previous-close direction
- `intraday_regime`: trend-up / trend-down / gap-fade / range
- `breadth_direction`: broad participation vs narrow heavyweight-led move
- `magnitude`: open gap / close return / MFE / MAE in basis points
- `scenario_probability`: Bull / Base / Bear probabilities

The central research hypothesis is that signals useful for the **open** are not necessarily useful for the **close**.

## Narrative contract

Every frozen daily prediction must include text for:

- overall market view;
- expected open;
- expected intraday path;
- expected close;
- expected breadth / heavyweight structure;
- Bull / Base / Bear scripts;
- key drivers;
- invalidation conditions.

This preserves the original V0 philosophy while allowing V1 to measure whether the written thesis was consistent with the numerical forecast and the realized market.

## Quantitative evaluation

Primary metrics:

- Open Direction Accuracy
- Close Direction Accuracy
- Regime Accuracy
- Magnitude MAE
- Brier Score
- Calibration Error
- Breadth Accuracy
- Baseline Lift

Every model must be compared against simple baselines before being called useful.

## Repository layout

```text
src/tw_scenario/        core Python package
config/                 target / feature definitions
schemas/                JSON contracts for predictions and realizations
predictions/            immutable daily V1 snapshots
realizations/           realized market outcomes
reports/                generated research outputs
docs/architecture/      LaTeX system diagrams and architecture notes
docs/methodology/       research method, validation, anti-leakage design
notebooks/               walk-forward and exploratory research
examples/                sample prediction and realization payloads
data/                    V1 data contracts; optional frozen V0 exports only
```

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
pytest -q
```

Run the sample evaluation:

```bash
python -m tw_scenario.cli evaluate \
  examples/sample_prediction.json \
  examples/sample_realization.json
```

## V0 boundary / reference policy

- V1 **must not write to** the V0 Google Sheet.
- V1 **may read or consult** V0 as qualitative reference material.
- Reproducible quantitative experiments should use a **frozen V0 export** with source date, row identity, extraction time, and provenance preserved.
- V0 post-close commentary must never be backfilled into a pre-open V1 feature set.
- Historical V0 text is reference evidence, not automatically a training label.

See [`docs/V0_ISOLATION.md`](docs/V0_ISOLATION.md).

## Status

**Stage: research scaffold / V1.0.0-alpha**

Next milestones:

1. Official-source market ingestion
2. Read-only V0 reference/export adapter
3. Reproducible feature pipeline
4. Naive baseline benchmark
5. 60+ trading-day frozen sample
6. Walk-forward evaluation
7. Calibration and error-attribution dashboard
8. Subscription-product feasibility review only after statistical and legal validation

## Important

This repository is a research system, not a promise of investment returns. Commercialization, advertising, and paid distribution require a separate legal/compliance review in Taiwan.


## V0 operational continuity

V0 is not an archived prototype. It remains a **live operating system** and continues to run through the existing ChatGPT scheduled tasks.

The V0 operating loop includes:

- the scheduled **財經事件晨報**, which creates the text-first pre-open hypotheses and writes them to the existing Google Sheet;
- the scheduled **台股收盤預測檢討**, which reads the morning hypotheses, compares them with actual open / intraday / close outcomes, and writes the review and reusable model lessons back to the existing Google Sheet.

V1 must not pause, replace, reschedule, redirect, or otherwise interfere with those V0 automations unless an explicit migration is separately approved.

V1 may consume V0 outputs in read-only mode for reference, research comparison, and frozen provenance-preserving exports.
