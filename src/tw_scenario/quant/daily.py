from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from datetime import date, datetime, time
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .contracts import TZ, write_once

HASH_PATTERN = r"^[0-9a-f]{64}$"
FORECAST_ID_PATTERN = r"^ohlc-v1-\d{4}-\d{2}-\d{2}-[0-9a-f]{64}$"
SNAPSHOT_VERSION = "ohlc-forecast-v1"
REALIZATION_VERSION = "ohlc-realization-v1"
EVALUATION_VERSION = "ohlc-evaluation-v1"

V0_SHEET_NAME = "每日預測與覆盤"
V0_HEADERS = (
    "日期",
    "盤前預測項目",
    "盤前原始假設",
    "預測方向",
    "信心度",
    "盤前證據/原因",
    "實際開盤",
    "實際盤中",
    "實際收盤",
    "驗證結果",
    "錯誤類型",
    "實際主因",
    "需要修正的假設",
    "隔日改進規則",
    "晨報來源/備註",
    "覆盤時間",
)


class StrictFrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


def _aware(value: datetime, field: str) -> None:
    if value.utcoffset() is None:
        raise ValueError(f"{field} must be timezone-aware")


def _canonical_hash(payload: Mapping) -> str:
    raw = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


class OHLCValues(StrictFrozenModel):
    open: float = Field(gt=0)
    high: float = Field(gt=0)
    low: float = Field(gt=0)
    close: float = Field(gt=0)

    @model_validator(mode="after")
    def coherent(self):
        if self.high < max(self.open, self.close):
            raise ValueError("high must be >= max(open, close)")
        if self.low > min(self.open, self.close):
            raise ValueError("low must be <= min(open, close)")
        return self


class MarginalInterval(StrictFrozenModel):
    lower: float = Field(gt=0)
    upper: float = Field(gt=0)

    @model_validator(mode="after")
    def ordered(self):
        if self.lower > self.upper:
            raise ValueError("interval lower must be <= upper")
        return self


class OHLCMarginalIntervals80(StrictFrozenModel):
    level: float = 0.8
    open: MarginalInterval
    high: MarginalInterval
    low: MarginalInterval
    close: MarginalInterval
    method: str = Field(min_length=1)

    @model_validator(mode="after")
    def exactly_eighty_percent(self):
        if abs(self.level - 0.8) > 1e-12:
            raise ValueError("daily contract requires nominal 80% marginal intervals")
        return self


class FallbackInfo(StrictFrozenModel):
    model_name: str = Field(min_length=1)
    reason: str = Field(min_length=1)


class ModelIdentity(StrictFrozenModel):
    model_name: str = Field(min_length=1)
    model_version: str = Field(min_length=1)
    checkpoint_id: str = Field(min_length=1)
    representation: Literal["relative", "gbud"]
    fallback: FallbackInfo | None = None


class InstrumentOHLCForecast(StrictFrozenModel):
    instrument: Literal["TAIEX", "TX"]
    contract: str | None
    reference_contract: str | None
    session: Literal["day"] = "day"
    reference_date: date
    previous_close: float = Field(gt=0)
    point: OHLCValues
    intervals: OHLCMarginalIntervals80
    model: ModelIdentity

    @model_validator(mode="after")
    def instrument_contract_and_intervals(self):
        if self.instrument == "TAIEX":
            if self.contract is not None or self.reference_contract is not None:
                raise ValueError("TAIEX must not carry a futures contract")
        else:
            if self.contract is None or len(self.contract) != 6 or not self.contract.isdigit():
                raise ValueError("TX requires an explicit YYYYMM contract")
            if self.reference_contract != self.contract:
                raise ValueError("TX previous close must come from the same contract")

        for field in ("open", "high", "low", "close"):
            point = getattr(self.point, field)
            interval = getattr(self.intervals, field)
            if not interval.lower <= point <= interval.upper:
                raise ValueError(f"{field} point forecast must lie inside its 80% interval")
        return self


class DataQualityEvidence(StrictFrozenModel):
    status: Literal["PASS", "WARN", "FAIL"]
    report_hash: str = Field(pattern=HASH_PATTERN)
    issues: tuple[str, ...] = ()

    @model_validator(mode="after")
    def failure_has_reason(self):
        if self.status == "FAIL" and not self.issues:
            raise ValueError("FAIL data quality requires at least one issue")
        return self


class SourceEvidence(StrictFrozenModel):
    source_id: str = Field(min_length=1)
    content_hash: str = Field(pattern=HASH_PATTERN)
    available_at: datetime

    @model_validator(mode="after")
    def timestamp_is_aware(self):
        _aware(self.available_at, "source.available_at")
        return self


def deterministic_forecast_id(
    *,
    trading_date: date,
    cutoff_at: datetime,
    forecasts: Sequence[InstrumentOHLCForecast],
    data_quality: DataQualityEvidence,
    sources: Sequence[SourceEvidence],
    data_manifest_hash: str,
    config_hash: str,
    code_version: str,
) -> str:
    """Identify forecast content independently of run and persistence timestamps."""
    identity = {
        "schema_version": SNAPSHOT_VERSION,
        "trading_date": trading_date.isoformat(),
        "cutoff_at": cutoff_at.isoformat(),
        "forecasts": [forecast.model_dump(mode="json") for forecast in forecasts],
        "data_quality": data_quality.model_dump(mode="json"),
        "sources": [source.model_dump(mode="json") for source in sources],
        "data_manifest_hash": data_manifest_hash,
        "config_hash": config_hash,
        "code_version": code_version,
    }
    return f"ohlc-v1-{trading_date.isoformat()}-{_canonical_hash(identity)}"


class OHLCForecastSnapshotBodyV1(StrictFrozenModel):
    schema_version: Literal["ohlc-forecast-v1"] = SNAPSHOT_VERSION
    forecast_id: str = Field(pattern=FORECAST_ID_PATTERN)
    trading_date: date
    cutoff_at: datetime
    generated_at: datetime
    persisted_at: datetime
    timeliness: Literal["ON_TIME", "LATE"]
    forecasts: tuple[InstrumentOHLCForecast, ...] = Field(min_length=2, max_length=2)
    data_quality: DataQualityEvidence
    sources: tuple[SourceEvidence, ...] = Field(min_length=1)
    data_manifest_hash: str = Field(pattern=HASH_PATTERN)
    config_hash: str = Field(pattern=HASH_PATTERN)
    code_version: str = Field(min_length=1)

    @model_validator(mode="after")
    def daily_contract(self):
        for field in ("cutoff_at", "generated_at", "persisted_at"):
            _aware(getattr(self, field), field)

        cutoff_local = self.cutoff_at.astimezone(TZ)
        if cutoff_local.date() != self.trading_date:
            raise ValueError("cutoff date must equal trading_date in Asia/Taipei")
        if (cutoff_local.hour, cutoff_local.minute, cutoff_local.second, cutoff_local.microsecond) != (
            8,
            30,
            0,
            0,
        ):
            raise ValueError("cutoff_at must be exactly 08:30 Asia/Taipei")
        if not self.cutoff_at <= self.generated_at <= self.persisted_at:
            raise ValueError("timestamps must satisfy cutoff <= generated <= persisted")

        deadline = datetime.combine(self.trading_date, time(8, 40), TZ)
        expected_timeliness = "ON_TIME" if self.persisted_at <= deadline else "LATE"
        if self.timeliness != expected_timeliness:
            raise ValueError(f"timeliness must be {expected_timeliness}")
        if self.trading_date.isoformat() not in self.forecast_id:
            raise ValueError("forecast_id must contain trading_date")

        instruments = [forecast.instrument for forecast in self.forecasts]
        if instruments != ["TAIEX", "TX"]:
            raise ValueError("forecasts must contain exactly TAIEX then TX")
        for forecast in self.forecasts:
            if forecast.reference_date >= self.trading_date:
                raise ValueError("reference_date must precede trading_date")

        source_ids = [source.source_id for source in self.sources]
        if len(source_ids) != len(set(source_ids)):
            raise ValueError("source_id values must be unique")
        if any(source.available_at > self.cutoff_at for source in self.sources):
            raise ValueError("forecast sources must be available by cutoff_at")
        expected_forecast_id = deterministic_forecast_id(
            trading_date=self.trading_date,
            cutoff_at=self.cutoff_at,
            forecasts=self.forecasts,
            data_quality=self.data_quality,
            sources=self.sources,
            data_manifest_hash=self.data_manifest_hash,
            config_hash=self.config_hash,
            code_version=self.code_version,
        )
        if self.forecast_id != expected_forecast_id:
            raise ValueError("forecast_id does not match deterministic forecast content")
        return self


class OHLCForecastSnapshotV1(OHLCForecastSnapshotBodyV1):
    snapshot_hash: str = Field(pattern=HASH_PATTERN)

    @model_validator(mode="after")
    def hash_matches_payload(self):
        body = OHLCForecastSnapshotBodyV1.model_validate(
            self.model_dump(mode="json", exclude={"snapshot_hash"})
        )
        if self.snapshot_hash != snapshot_body_hash(body):
            raise ValueError("snapshot_hash does not match canonical snapshot payload")
        return self


def snapshot_body_hash(body: OHLCForecastSnapshotBodyV1) -> str:
    return _canonical_hash(body.model_dump(mode="json"))


def seal_snapshot(
    body: OHLCForecastSnapshotBodyV1 | Mapping,
) -> OHLCForecastSnapshotV1:
    """Validate a daily body, add its canonical hash, and return a sealed snapshot."""
    validated = (
        body
        if isinstance(body, OHLCForecastSnapshotBodyV1)
        else OHLCForecastSnapshotBodyV1.model_validate(body)
    )
    payload = validated.model_dump(mode="json")
    payload["snapshot_hash"] = snapshot_body_hash(validated)
    return OHLCForecastSnapshotV1.model_validate(payload)


def freeze_snapshot(snapshot: OHLCForecastSnapshotV1, path: str | Path) -> Path:
    """Durably create one immutable snapshot. Existing paths are never replaced."""
    path = Path(path)
    write_once(path, snapshot.model_dump(mode="json"))
    persisted = OHLCForecastSnapshotV1.model_validate_json(path.read_text(encoding="utf-8"))
    if persisted.snapshot_hash != snapshot.snapshot_hash:
        raise RuntimeError("frozen snapshot readback mismatch")
    return path


def load_snapshot(path: str | Path) -> OHLCForecastSnapshotV1:
    return OHLCForecastSnapshotV1.model_validate_json(Path(path).read_text(encoding="utf-8"))


class OHLCRealizationV1(StrictFrozenModel):
    schema_version: Literal["ohlc-realization-v1"] = REALIZATION_VERSION
    realization_id: str = Field(min_length=1)
    forecast_id: str = Field(pattern=FORECAST_ID_PATTERN)
    snapshot_hash: str = Field(pattern=HASH_PATTERN)
    trading_date: date
    instrument: Literal["TAIEX", "TX"]
    contract: str | None
    session: Literal["day"] = "day"
    actual: OHLCValues
    source_id: str = Field(min_length=1)
    source_hash: str = Field(pattern=HASH_PATTERN)
    observed_at: datetime
    recorded_at: datetime

    @model_validator(mode="after")
    def valid_realization(self):
        _aware(self.observed_at, "observed_at")
        _aware(self.recorded_at, "recorded_at")
        if self.observed_at > self.recorded_at:
            raise ValueError("observed_at must not follow recorded_at")
        if self.instrument == "TAIEX" and self.contract is not None:
            raise ValueError("TAIEX realization must not carry a contract")
        if self.instrument == "TX" and (
            self.contract is None or len(self.contract) != 6 or not self.contract.isdigit()
        ):
            raise ValueError("TX realization requires YYYYMM contract")
        return self


class OHLCMetricVector(StrictFrozenModel):
    open: float
    high: float
    low: float
    close: float


class OHLCEvaluationV1(StrictFrozenModel):
    schema_version: Literal["ohlc-evaluation-v1"] = EVALUATION_VERSION
    evaluation_id: str = Field(min_length=1)
    realization_id: str = Field(min_length=1)
    forecast_id: str = Field(pattern=FORECAST_ID_PATTERN)
    snapshot_hash: str = Field(pattern=HASH_PATTERN)
    trading_date: date
    instrument: Literal["TAIEX", "TX"]
    contract: str | None
    signed_error_points: OHLCMetricVector
    absolute_error_points: OHLCMetricVector
    absolute_error_bps: OHLCMetricVector
    gap_error_bps: float
    close_return_error_bps: float
    range_error_points: float
    evaluated_at: datetime

    @model_validator(mode="after")
    def valid_evaluation(self):
        _aware(self.evaluated_at, "evaluated_at")
        if self.instrument == "TAIEX" and self.contract is not None:
            raise ValueError("TAIEX evaluation must not carry a contract")
        if self.instrument == "TX" and (
            self.contract is None or len(self.contract) != 6 or not self.contract.isdigit()
        ):
            raise ValueError("TX evaluation requires YYYYMM contract")
        return self


def _forecast_for(
    snapshot: OHLCForecastSnapshotV1, instrument: str
) -> InstrumentOHLCForecast:
    return next(forecast for forecast in snapshot.forecasts if forecast.instrument == instrument)


def evaluate_realization(
    snapshot: OHLCForecastSnapshotV1,
    realization: OHLCRealizationV1,
    evaluated_at: datetime,
) -> OHLCEvaluationV1:
    """Create a separate evaluation record without changing the forecast snapshot."""
    _aware(evaluated_at, "evaluated_at")
    if realization.forecast_id != snapshot.forecast_id:
        raise ValueError("realization forecast_id does not match snapshot")
    if realization.snapshot_hash != snapshot.snapshot_hash:
        raise ValueError("realization snapshot_hash does not match snapshot")
    if realization.trading_date != snapshot.trading_date:
        raise ValueError("realization trading_date does not match snapshot")
    if evaluated_at < realization.recorded_at:
        raise ValueError("evaluation cannot precede realization recording")

    forecast = _forecast_for(snapshot, realization.instrument)
    if realization.contract != forecast.contract:
        raise ValueError("realization contract does not match frozen forecast")

    fields = ("open", "high", "low", "close")
    signed = {
        field: getattr(forecast.point, field) - getattr(realization.actual, field)
        for field in fields
    }
    absolute = {field: abs(value) for field, value in signed.items()}
    absolute_bps = {
        field: value / forecast.previous_close * 10_000 for field, value in absolute.items()
    }
    return OHLCEvaluationV1(
        evaluation_id=f"evaluation-{realization.realization_id}",
        realization_id=realization.realization_id,
        forecast_id=snapshot.forecast_id,
        snapshot_hash=snapshot.snapshot_hash,
        trading_date=snapshot.trading_date,
        instrument=forecast.instrument,
        contract=forecast.contract,
        signed_error_points=OHLCMetricVector(**signed),
        absolute_error_points=OHLCMetricVector(**absolute),
        absolute_error_bps=OHLCMetricVector(**absolute_bps),
        gap_error_bps=signed["open"] / forecast.previous_close * 10_000,
        close_return_error_bps=signed["close"] / forecast.previous_close * 10_000,
        range_error_points=(forecast.point.high - forecast.point.low)
        - (realization.actual.high - realization.actual.low),
        evaluated_at=evaluated_at,
    )


def _append_path(root: Path, kind: str, forecast_id: str, instrument: str, contract: str | None):
    label = contract if contract is not None else "cash"
    return root / kind / forecast_id / f"{instrument}-{label}.json"


def append_realization(record: OHLCRealizationV1, root: str | Path) -> Path:
    """Append by exclusive record creation; a duplicate can never replace prior evidence."""
    path = _append_path(
        Path(root), "realizations-v1", record.forecast_id, record.instrument, record.contract
    )
    write_once(path, record.model_dump(mode="json"))
    return path


def append_evaluation(record: OHLCEvaluationV1, root: str | Path) -> Path:
    path = _append_path(
        Path(root), "evaluations-v1", record.forecast_id, record.instrument, record.contract
    )
    write_once(path, record.model_dump(mode="json"))
    return path


class NewsBackgroundItem(StrictFrozenModel):
    news_id: str = Field(min_length=1)
    headline: str = Field(min_length=1)
    summary: str = Field(min_length=1)
    source_url: str = Field(min_length=1)
    published_at: datetime

    @model_validator(mode="after")
    def timestamp_is_aware(self):
        _aware(self.published_at, "news.published_at")
        return self


class V0NarrativeItem(StrictFrozenModel):
    instrument: Literal["TAIEX", "TX"]
    item_name: str
    forecast_quote: InstrumentOHLCForecast
    open_hypothesis: str
    intraday_hypothesis: str
    close_hypothesis: str
    confidence_text: Literal["未建立方向機率；80%邊際預測區間僅作區間呈現。"]


class V0NarrativePayload(StrictFrozenModel):
    schema_version: Literal["v0-narrative-from-ohlc-v1"] = "v0-narrative-from-ohlc-v1"
    forecast_id: str
    snapshot_hash: str = Field(pattern=HASH_PATTERN)
    trading_date: date
    cutoff_at: datetime
    items: tuple[V0NarrativeItem, ...] = Field(min_length=2, max_length=2)
    news_background: tuple[NewsBackgroundItem, ...]

    @model_validator(mode="after")
    def daily_narrative_contract(self):
        _aware(self.cutoff_at, "cutoff_at")
        if [item.instrument for item in self.items] != ["TAIEX", "TX"]:
            raise ValueError("narrative items must contain exactly TAIEX then TX")
        news_ids = [item.news_id for item in self.news_background]
        if len(news_ids) != len(set(news_ids)):
            raise ValueError("news_id values must be unique")
        if any(item.published_at > self.cutoff_at for item in self.news_background):
            raise ValueError("news published after cutoff cannot enter the morning narrative")
        if any("待驗證情境假說" not in item.intraday_hypothesis for item in self.items):
            raise ValueError("every intraday hypothesis must be labelled as unvalidated")
        return self


def _number(value: float) -> str:
    """Use the same JSON numeric representation as the frozen snapshot."""
    return json.dumps(value, allow_nan=False, separators=(",", ":"))


def build_v0_narrative(
    snapshot: OHLCForecastSnapshotV1,
    news: Sequence[NewsBackgroundItem | Mapping] = (),
) -> V0NarrativePayload:
    """Render deterministic hypotheses; news is background and never changes any number."""
    if snapshot.data_quality.status == "FAIL":
        raise ValueError("cannot render a publishable narrative from DQ FAIL")
    validated_news = tuple(
        item if isinstance(item, NewsBackgroundItem) else NewsBackgroundItem.model_validate(item)
        for item in news
    )
    ordered_news = tuple(
        sorted(validated_news, key=lambda item: (item.published_at, item.news_id))
    )
    if len({item.news_id for item in ordered_news}) != len(ordered_news):
        raise ValueError("news_id values must be unique")
    if any(item.published_at > snapshot.cutoff_at for item in ordered_news):
        raise ValueError("news published after cutoff cannot enter the morning narrative")

    items = []
    for forecast in snapshot.forecasts:
        label = "加權指數日盤 OHLC" if forecast.instrument == "TAIEX" else (
            f"台指期 {forecast.contract} 日盤 OHLC"
        )
        point, interval = forecast.point, forecast.intervals
        open_hypothesis = (
            f"凍結盤前數字：前收 {_number(forecast.previous_close)}；"
            f"預測開盤 {_number(point.open)}；80%邊際區間 "
            f"[{_number(interval.open.lower)}, {_number(interval.open.upper)}]。"
        )
        intraday_hypothesis = (
            "待驗證情境假說：日 OHLC 僅提供"
            f"預測最高 {_number(point.high)}（80%邊際區間 "
            f"[{_number(interval.high.lower)}, {_number(interval.high.upper)}]）與"
            f"預測最低 {_number(point.low)}（80%邊際區間 "
            f"[{_number(interval.low.lower)}, {_number(interval.low.upper)}]）；"
            "不能判定高低點先後或盤中路徑。"
        )
        close_hypothesis = (
            f"凍結盤前數字：預測收盤 {_number(point.close)}；80%邊際區間 "
            f"[{_number(interval.close.lower)}, {_number(interval.close.upper)}]。"
        )
        items.append(
            V0NarrativeItem(
                instrument=forecast.instrument,
                item_name=label,
                forecast_quote=forecast,
                open_hypothesis=open_hypothesis,
                intraday_hypothesis=intraday_hypothesis,
                close_hypothesis=close_hypothesis,
                confidence_text="未建立方向機率；80%邊際預測區間僅作區間呈現。",
            )
        )
    return V0NarrativePayload(
        forecast_id=snapshot.forecast_id,
        snapshot_hash=snapshot.snapshot_hash,
        trading_date=snapshot.trading_date,
        cutoff_at=snapshot.cutoff_at,
        items=tuple(items),
        news_background=ordered_news,
    )


def verify_narrative_snapshot_link(
    snapshot: OHLCForecastSnapshotV1, narrative: V0NarrativePayload
) -> bool:
    try:
        expected = build_v0_narrative(snapshot, narrative.news_background)
    except ValueError:
        return False
    return narrative == expected


def header_fingerprint(headers: Sequence[str]) -> str:
    return hashlib.sha256("\x1f".join(headers).encode("utf-8")).hexdigest()


V0_HEADER_FINGERPRINT = header_fingerprint(V0_HEADERS)


class V0MorningSheetRow(StrictFrozenModel):
    values: tuple[str, ...] = Field(min_length=16, max_length=16)

    @model_validator(mode="after")
    def only_preopen_columns_are_populated(self):
        if any(self.values[index] for index in range(6, 14)) or self.values[15]:
            raise ValueError("dry-run morning rows must leave G:N and P blank")
        if not all(self.values[index] for index in range(6)) or not self.values[14]:
            raise ValueError("dry-run morning rows require A:F and O")
        return self


class V0SheetDryRunBatch(StrictFrozenModel):
    mode: Literal["DRY_RUN_ONLY"] = "DRY_RUN_ONLY"
    sheet_name: Literal["每日預測與覆盤"] = V0_SHEET_NAME
    header_range: Literal["'每日預測與覆盤'!A1:P1"] = "'每日預測與覆盤'!A1:P1"
    expected_headers: tuple[str, ...] = V0_HEADERS
    expected_header_fingerprint: str = V0_HEADER_FINGERPRINT
    forecast_id: str
    snapshot_hash: str = Field(pattern=HASH_PATTERN)
    write_range: str
    morning_write_columns: Literal["A:F,O"] = "A:F,O"
    postclose_reserved_columns: Literal["G:N,P"] = "G:N,P"
    rows: tuple[V0MorningSheetRow, ...] = Field(min_length=2, max_length=2)

    @model_validator(mode="after")
    def fixed_sheet_contract(self):
        if self.expected_headers != V0_HEADERS:
            raise ValueError("dry-run headers must preserve the existing A:P mapping")
        if self.expected_header_fingerprint != header_fingerprint(self.expected_headers):
            raise ValueError("dry-run header fingerprint does not match expected headers")
        return self


def _direction(value: float, reference: float) -> str:
    if value > reference:
        return "偏多"
    if value < reference:
        return "偏空"
    return "持平"


def build_v0_sheet_dry_run(
    snapshot: OHLCForecastSnapshotV1,
    narrative: V0NarrativePayload,
    observed_headers: Sequence[str],
    start_row: int,
) -> V0SheetDryRunBatch:
    """Build an A:P row batch. This module intentionally has no Sheet write operation."""
    if tuple(observed_headers) != V0_HEADERS:
        raise ValueError(
            "V0 header fingerprint mismatch: "
            f"expected={V0_HEADER_FINGERPRINT} observed={header_fingerprint(observed_headers)}"
        )
    if start_row < 2:
        raise ValueError("start_row must be below the header")
    if not verify_narrative_snapshot_link(snapshot, narrative):
        raise ValueError("narrative does not quote the supplied frozen snapshot")

    news_text = "；".join(
        f"{item.headline}（{item.published_at.isoformat()}；{item.source_url}）"
        for item in narrative.news_background
    ) or "無截止前新聞背景"
    rows = []
    for item in narrative.items:
        forecast = item.forecast_quote
        model = forecast.model
        fallback = (
            f"；fallback={model.fallback.model_name}：{model.fallback.reason}"
            if model.fallback is not None
            else "；fallback=none"
        )
        evidence = (
            f"數值模型={model.model_name}/{model.model_version}；"
            f"checkpoint={model.checkpoint_id}；representation={model.representation}{fallback}。"
            f"新聞僅作背景，不修改凍結數字：{news_text}"
        )
        note = (
            f"forecast_id={snapshot.forecast_id}; snapshot_hash={snapshot.snapshot_hash}; "
            f"cutoff_at={snapshot.cutoff_at.isoformat()}; "
            f"persisted_at={snapshot.persisted_at.isoformat()}; "
            f"timeliness={snapshot.timeliness}; data_quality={snapshot.data_quality.status}"
        )
        values = (
            snapshot.trading_date.strftime("%Y/%m/%d"),
            item.item_name,
            f"{item.open_hypothesis}\n{item.intraday_hypothesis}\n{item.close_hypothesis}",
            (
                f"開盤{_direction(forecast.point.open, forecast.previous_close)}／"
                f"收盤{_direction(forecast.point.close, forecast.previous_close)}"
            ),
            item.confidence_text,
            evidence,
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            note,
            "",
        )
        rows.append(V0MorningSheetRow(values=values))
    end_row = start_row + len(rows) - 1
    return V0SheetDryRunBatch(
        forecast_id=snapshot.forecast_id,
        snapshot_hash=snapshot.snapshot_hash,
        write_range=f"'{V0_SHEET_NAME}'!A{start_row}:P{end_row}",
        rows=tuple(rows),
    )


def save_sheet_dry_run(batch: V0SheetDryRunBatch, path: str | Path) -> Path:
    path = Path(path)
    write_once(path, batch.model_dump(mode="json"))
    return path
