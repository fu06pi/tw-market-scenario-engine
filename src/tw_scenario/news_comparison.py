"""Separate, pre-open news-direction research; never edits quantitative prices.

This module accepts independently produced decisions, not an LLM price generator.
Local timestamps alone do not establish genuine prospective eligibility.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, time

import numpy as np
from pydantic import AwareDatetime, Field, model_validator

from .metrics import realized_direction
from .models import Direction
from .morning_adapter import validate_forecast
from .policy import ScoringPolicy
from .shadow import seal
from .snapshot import document_hash, verify_document, write_immutable_json
from .split_forecast import TAIPEI, ContractModel, ObservedBar


class NewsEvidence(ContractModel):
    news_id: str = Field(min_length=1)
    source_url: str = Field(pattern=r"^https://")
    published_at: AwareDatetime
    fetched_at: AwareDatetime
    evidence_text: str = Field(min_length=1)
    availability_basis: str = Field(min_length=1)

    @model_validator(mode="after")
    def chronology(self):
        if self.published_at > self.fetched_at:
            raise ValueError("news cannot be fetched before publication")
        return self


class NewsDecision(ContractModel):
    schema_version: str = "news-direction-v1"
    forecast_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    market: str
    contract: str | None = None
    quantitative_method: str
    direction: Direction
    generated_at: AwareDatetime
    model_version: str = Field(min_length=1)
    prompt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    rationale: str = Field(min_length=1)
    news: tuple[NewsEvidence, ...] = Field(min_length=1)


def validate_decision(forecast, decision):
    validate_forecast(forecast)
    if decision.schema_version != "news-direction-v1" or decision.forecast_hash != forecast["snapshot_hash"]:
        raise ValueError("decision must reference the unchanged quantitative snapshot")
    data = forecast["markets"][decision.market]
    if decision.contract != data["inputs"]["contract"]:
        raise ValueError("news decision contract differs from quantitative contract")
    if decision.quantitative_method != data["default_baseline"]:
        raise ValueError("quantitative comparator must be the locked default baseline")
    generated = decision.generated_at.astimezone(TAIPEI)
    if (generated.date().isoformat() != forecast["trading_date"]
            or generated.time() >= time(8, 40)
            or decision.generated_at < datetime.fromisoformat(forecast["generated_at"])):
        raise ValueError("news decision must follow quantitative freeze and precede 08:40")
    ids = [n.news_id for n in decision.news]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate news identity")
    for news in decision.news:
        fetched = news.fetched_at.astimezone(TAIPEI)
        cutoff = datetime.combine(generated.date(), time(8, 30), TAIPEI)
        if news.fetched_at > decision.generated_at or fetched > cutoff:
            raise ValueError("news was unavailable before the decision/cutoff")
    return decision


def freeze_decision(forecast, payload, at, path):
    if at.tzinfo is None:
        raise ValueError("aware actual clock required")
    if "generated_at" in payload:
        raise ValueError("generated_at is assigned from the execution clock, not input")
    decision = validate_decision(forecast, NewsDecision(**payload, generated_at=at))
    frozen = seal({**decision.model_dump(mode="json"), "remote_persistence_verified": False,
                   "formal_accuracy_eligible": False, "target": "same_day_close_direction"})
    write_immutable_json(frozen, path)
    return frozen


def evaluate_pairs(forecasts, decisions, actual_records):
    """Expected population is every supplied forecast market, including missing B."""
    forecast_map = {}
    for forecast in forecasts:
        validate_forecast(forecast)
        if forecast["snapshot_hash"] in forecast_map:
            raise ValueError("duplicate quantitative snapshot")
        forecast_map[forecast["snapshot_hash"]] = forecast
    actuals = {}
    for raw in actual_records:
        actual = ObservedBar.model_validate(raw)
        if actual.available_at > datetime.now(TAIPEI):
            raise ValueError("actual availability is in the future")
        key = (actual.trading_date.isoformat(), actual.market, actual.contract)
        if key in actuals:
            raise ValueError("duplicate actual identity")
        actuals[key] = actual
    accepted, failures, duplicate = {}, [], set()
    for raw in decisions:
        try:
            if not verify_document(raw).valid:
                raise ValueError("news decision hash failed")
            keys = NewsDecision.model_fields
            metadata = {"hash_version", "snapshot_hash", "remote_persistence_verified",
                        "formal_accuracy_eligible", "target"}
            if set(raw) - set(keys) - metadata or raw.get("target") != "same_day_close_direction":
                raise ValueError("unexpected price fields or inconsistent prediction target")
            decision = NewsDecision(**{k: v for k, v in raw.items() if k in keys})
            forecast = forecast_map[decision.forecast_hash]
            validate_decision(forecast, decision)
            key = (decision.forecast_hash, decision.market)
            if key in accepted or key in duplicate:
                accepted.pop(key, None); duplicate.add(key)
                raise ValueError("duplicate B decisions; reject the entire pair, no cherry-picking")
            accepted[key] = decision
        except (ValueError, KeyError, TypeError) as exc:
            failures.append({"forecast_hash": raw.get("forecast_hash"), "market": raw.get("market"),
                             "reason": str(exc)})
    grouped, missing, expected = defaultdict(list), [], 0
    for digest, forecast in forecast_map.items():
        policy = ScoringPolicy.model_validate(forecast["scoring_policy"])
        for market, data in forecast["markets"].items():
            expected += 1
            key = (digest, market)
            b = accepted.get(key)
            inputs = data["inputs"]
            actual = actuals.get((forecast["trading_date"], "TX" if market == "TX_selected_monthly"
                                 else market, inputs["contract"]))
            if b is None or actual is None:
                missing.append({"forecast_hash": digest, "market": market,
                                "reason": "missing/invalid B" if b is None else "missing actual"})
                continue
            if b.generated_at >= actual.available_at:
                failures.append({"forecast_hash": digest, "market": market,
                                 "reason": "decision is not earlier than actual availability"})
                continue
            reference = inputs["previous_close"]
            method = data["default_baseline"]
            quant_close = data["models"][method]["prices"]["close"]
            a_direction = realized_direction((quant_close / reference - 1) * 10000,
                                             policy.flat_threshold_bps)
            truth = realized_direction((actual.close / reference - 1) * 10000,
                                       policy.flat_threshold_bps)
            group = (market, forecast["checkpoint_hash"], method, b.model_version,
                     b.prompt_sha256, document_hash(policy.model_dump()))
            grouped[group].append({"date": forecast["trading_date"], "contract": inputs["contract"],
                                   "a_hit": int(a_direction == truth), "b_hit": int(b.direction == truth),
                                   "actual_direction": truth.value})
    results = []
    for group, rows in grouped.items():
        rows.sort(key=lambda r: r["date"])
        if len({r["date"] for r in rows}) != len(rows):
            raise ValueError("distinct snapshots on the same date cannot be pooled")
        diff = np.array([r["b_hit"] - r["a_hit"] for r in rows], dtype=float)
        interval = None
        if len(rows) >= 30:
            rng = np.random.default_rng(20261116)
            blocks = rng.integers(0, len(rows) - 4, size=(2000, (len(rows) + 4) // 5))
            indices = (blocks[:, :, None] + np.arange(5)).reshape(2000, -1)[:, :len(rows)]
            interval = np.quantile(diff[indices].mean(axis=1), [.025, .975]).tolist()
        results.append({"market": group[0], "checkpoint": group[1], "quantitative_method": group[2],
                        "ai_model_version": group[3], "prompt_sha256": group[4], "policy_hash": group[5],
                        "paired_n": len(rows), "a_direction_accuracy": float(np.mean([r["a_hit"] for r in rows])),
                        "b_direction_accuracy": float(np.mean([r["b_hit"] for r in rows])),
                        "b_minus_a_accuracy": float(diff.mean()),
                        "a_only_correct": int(sum(d < 0 for d in diff)),
                        "b_only_correct": int(sum(d > 0 for d in diff)),
                        "block5_interval_95": interval, "paired_rows": rows,
                        "price_mae_comparison": "not_applicable: B emits direction, no new prices",
                        "conclusion": "insufficient observations" if len(rows) < 30 else "research comparison only"})
    return {"schema_version": "news-direction-comparison-v1", "expected_pairs": expected,
            "paired_n": sum(g["paired_n"] for g in results), "missing": missing, "failures": failures,
            "groups": results, "formal_accuracy_eligible": False,
            "remote_timing_verification": "required separately; local hashes are not time evidence",
            "price_predictions_modified": False}
