"""Causal TAIFEX market features with immutable source provenance."""

from __future__ import annotations

import base64
import csv
import hashlib
import io
import json
import os
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np

from .contracts import Calendar, at, digest, write_once

TAIFEX_PC_RATIO_URL = "https://www.taifex.com.tw/cht/3/pcRatioDown"
PC_RATIO_HEADER = (
    "日期",
    "賣權成交量",
    "買權成交量",
    "買賣權成交量比率%",
    "賣權未平倉量",
    "買權未平倉量",
    "買賣權未平倉量比率%",
)
PC_RATIO_FEATURE_NAMES = (
    "pcr_volume_log_ratio",
    "pcr_open_interest_log_ratio",
    "pcr_log_total_volume",
    "pcr_log_total_open_interest",
    "pcr_volume_log_ratio_change_1",
    "pcr_open_interest_log_ratio_change_1",
    "pcr_log_total_volume_change_1",
    "pcr_log_total_open_interest_change_1",
)


@dataclass(frozen=True)
class PutCallRatioRow:
    day: date
    put_volume: int
    call_volume: int
    volume_ratio_pct: float
    put_open_interest: int
    call_open_interest: int
    open_interest_ratio_pct: float
    available_at: datetime
    availability_basis: str
    source: str
    source_hash: str


def _parse_request(item: dict) -> tuple[dict, date, date]:
    if not isinstance(item, dict):
        raise TypeError("TAIFEX browser bundle item must be an object")
    request = item.get("request")
    if not isinstance(request, dict) or request.get("url") != TAIFEX_PC_RATIO_URL:
        raise ValueError("TAIFEX Put/Call request URL changed")
    form = request.get("form")
    if not isinstance(form, dict) or set(form) != {
        "queryStartDate", "queryEndDate", "down_type"
    } or form["down_type"] != "":
        raise ValueError("TAIFEX Put/Call request form changed")
    try:
        start = datetime.strptime(form["queryStartDate"], "%Y/%m/%d").date()
        end = datetime.strptime(form["queryEndDate"], "%Y/%m/%d").date()
    except (TypeError, ValueError):
        raise ValueError("TAIFEX Put/Call request dates invalid") from None
    if start > end or (end - start).days > 30 or (start.year, start.month) != (
        end.year, end.month
    ):
        raise ValueError("TAIFEX Put/Call request must be one bounded calendar month")
    return request, start, end


def _write_bytes_once(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != raw:
            raise FileExistsError(f"refusing to replace different immutable cache: {path}")
        return
    with path.open("xb") as target:
        target.write(raw)
        target.flush()
        os.fsync(target.fileno())


def import_pc_ratio_browser_bundle(bundle_path: Path, cache: Path) -> list[Path]:
    """Split a browser-exported bundle into the normal immutable official cache."""
    try:
        bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        raise ValueError("invalid TAIFEX browser bundle JSON") from None
    if not isinstance(bundle, dict) or bundle.get("version") != (
        "taifex-pcratio-browser-bundle-v1"
    ):
        raise ValueError("unsupported TAIFEX browser bundle version")
    items = bundle.get("items")
    if not isinstance(items, list) or not items:
        raise ValueError("TAIFEX browser bundle has no items")

    output = []
    seen = set()
    for item in items:
        request, _, _ = _parse_request(item)
        key = digest(request)
        if key in seen:
            raise ValueError("duplicate TAIFEX Put/Call request in browser bundle")
        seen.add(key)
        content_type = item.get("content_type")
        if not isinstance(content_type, str) or "MS950" not in content_type.upper():
            raise ValueError("TAIFEX Put/Call content type changed")
        try:
            fetched_at = datetime.fromisoformat(item["fetched_at"])
            raw = base64.b64decode(item["bytes_base64"], validate=True)
        except (KeyError, TypeError, ValueError):
            raise ValueError("TAIFEX browser bundle payload invalid") from None
        if fetched_at.utcoffset() is None or not raw:
            raise ValueError("TAIFEX browser bundle lacks timezone or bytes")

        csv_path = cache / f"{key}.csv"
        manifest_path = cache / f"{key}.manifest.json"
        sha256 = hashlib.sha256(raw).hexdigest()
        manifest = {
            "request": request,
            "fetched_at": fetched_at.isoformat(),
            "sha256": sha256,
            "bytes": len(raw),
            "content_type": content_type,
            "transport": "browser_cdp_same_origin",
            "availability_basis": (
                "ASSUMPTION: same-day 18:00 after the complete TAIFEX session; "
                "the historical page has no archived publication timestamps"
            ),
        }
        _write_bytes_once(csv_path, raw)
        if manifest_path.exists():
            try:
                existing = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                raise ValueError("existing TAIFEX Put/Call manifest is invalid") from None
            if existing != manifest:
                raise FileExistsError(
                    f"refusing to replace different immutable manifest: {manifest_path}"
                )
        else:
            write_once(manifest_path, manifest)
        # Parsing during import prevents a byte-valid but schema-invalid cache.
        parse_pc_ratio_cache(csv_path, manifest_path)
        output.append(csv_path)
    return output


def parse_pc_ratio_cache(csv_path: Path, manifest_path: Path) -> list[PutCallRatioRow]:
    try:
        raw = csv_path.read_bytes()
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        raise ValueError("invalid TAIFEX Put/Call cache") from None
    request, start, end = _parse_request({"request": manifest.get("request")})
    if (
        manifest.get("sha256") != hashlib.sha256(raw).hexdigest()
        or manifest.get("bytes") != len(raw)
        or manifest.get("content_type") is None
    ):
        raise ValueError("TAIFEX Put/Call cache integrity mismatch")
    expected_key = digest(request)
    if csv_path.stem != expected_key or manifest_path.name != f"{expected_key}.manifest.json":
        raise ValueError("TAIFEX Put/Call cache path does not match request")
    try:
        fetched_at = datetime.fromisoformat(manifest["fetched_at"])
        text = raw.decode("cp950")
    except (KeyError, TypeError, ValueError, UnicodeDecodeError):
        raise ValueError("TAIFEX Put/Call cache encoding or fetched_at invalid") from None
    if fetched_at.utcoffset() is None:
        raise ValueError("TAIFEX Put/Call fetched_at must be timezone aware")

    reader = csv.reader(io.StringIO(text))
    try:
        header = tuple(next(reader))
    except StopIteration:
        raise ValueError("TAIFEX Put/Call CSV is empty") from None
    if header != PC_RATIO_HEADER:
        raise ValueError("TAIFEX Put/Call CSV schema changed")
    rows = []
    seen = set()
    for values in reader:
        if not values or all(not value.strip() for value in values):
            continue
        if len(values) == 8 and values[-1] == "":
            values = values[:-1]
        if len(values) != 7:
            raise ValueError("TAIFEX Put/Call CSV row width changed")
        try:
            day = datetime.strptime(values[0], "%Y/%m/%d").date()
            counts = tuple(int(value.replace(",", "")) for value in (
                values[1], values[2], values[4], values[5]
            ))
            volume_ratio = float(values[3])
            oi_ratio = float(values[6])
        except (TypeError, ValueError):
            raise ValueError("TAIFEX Put/Call CSV value invalid") from None
        if not start <= day <= end or day in seen or min(counts) <= 0:
            raise ValueError("TAIFEX Put/Call date/count is outside contract")
        seen.add(day)
        put_volume, call_volume, put_oi, call_oi = counts
        if (
            abs(volume_ratio - put_volume / call_volume * 100) > 0.011
            or abs(oi_ratio - put_oi / call_oi * 100) > 0.011
        ):
            raise ValueError("TAIFEX Put/Call published ratio fails recomputation")
        rows.append(PutCallRatioRow(
            day=day,
            put_volume=put_volume,
            call_volume=call_volume,
            volume_ratio_pct=volume_ratio,
            put_open_interest=put_oi,
            call_open_interest=call_oi,
            open_interest_ratio_pct=oi_ratio,
            available_at=at(day, "18:00"),
            availability_basis=manifest["availability_basis"],
            source=TAIFEX_PC_RATIO_URL,
            source_hash=manifest["sha256"],
        ))
    if [row.day for row in rows] != sorted(seen, reverse=True):
        raise ValueError("TAIFEX Put/Call CSV must be reverse chronological")
    return list(reversed(rows))


def load_pc_ratio_cache(cache: Path, start: date, end: date) -> tuple[list[PutCallRatioRow], dict]:
    rows = []
    manifests = []
    month = start.replace(day=1)
    while month <= end:
        next_month = (month.replace(day=28) + timedelta(days=4)).replace(day=1)
        request_end = min(end, next_month - timedelta(days=1))
        request = {
            "url": TAIFEX_PC_RATIO_URL,
            "form": {
                "queryStartDate": month.strftime("%Y/%m/%d"),
                "queryEndDate": request_end.strftime("%Y/%m/%d"),
                "down_type": "",
            },
        }
        key = digest(request)
        csv_path = cache / f"{key}.csv"
        manifest_path = cache / f"{key}.manifest.json"
        if not csv_path.exists() or not manifest_path.exists():
            candidates = []
            for candidate in cache.glob("*.manifest.json"):
                try:
                    candidate_manifest = json.loads(candidate.read_text(encoding="utf-8"))
                    _, candidate_start, candidate_end = _parse_request(
                        {"request": candidate_manifest.get("request")}
                    )
                except (OSError, UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError):
                    continue
                if candidate_start == month and candidate_end >= request_end:
                    candidates.append(candidate)
            if len(candidates) != 1:
                raise ValueError(f"missing or ambiguous TAIFEX Put/Call month: {month:%Y-%m}")
            manifest_path = candidates[0]
            csv_path = cache / f"{manifest_path.name.removesuffix('.manifest.json')}.csv"
        parsed = parse_pc_ratio_cache(csv_path, manifest_path)
        rows.extend(row for row in parsed if start <= row.day <= end)
        manifests.append(json.loads(manifest_path.read_text(encoding="utf-8")))
        month = next_month
    days = [row.day for row in rows]
    if days != sorted(set(days)):
        raise ValueError("TAIFEX Put/Call combined cache contains duplicates")
    identity = {
        "version": "taifex-pcratio-feature-data-v1",
        "requested_start": str(start),
        "requested_end": str(end),
        "rows": len(rows),
        "row_hash": digest([asdict(row) for row in rows]),
        "source_hashes": sorted({manifest["sha256"] for manifest in manifests}),
    }
    identity["feature_dataset_id"] = digest(identity)
    return rows, identity


def pc_ratio_quality(rows: list[PutCallRatioRow], calendar: Calendar, start: date, end: date) -> dict:
    days = [row.day for row in rows]
    counts = {day: days.count(day) for day in set(days)}
    duplicates = sorted(str(day) for day, count in counts.items() if count > 1)
    expected = [day for day in calendar.days if start <= day <= end]
    observed = set(days)
    missing = [str(day) for day in expected if day not in observed]
    extra = [str(day) for day in sorted(observed) if day not in set(expected)]
    coverage = (len(expected) - len(missing)) / len(expected) if expected else 0.0
    return {
        "status": "PASS" if not duplicates and not extra and coverage >= 0.99 else "FAIL",
        "expected_days": len(expected),
        "observed_days": len(observed),
        "coverage": coverage,
        "missing": missing,
        "duplicates": duplicates,
        "extra": extra,
        "availability_basis": (
            "Only the previous complete exchange day is eligible for a D 08:30 forecast; "
            "same-day Put/Call values are excluded."
        ),
    }


class PutCallFeatureSet:
    def __init__(self, rows: list[PutCallRatioRow], identity: dict):
        self.rows = {row.day: row for row in rows}
        self.identity = identity

    @staticmethod
    def feature_names() -> tuple[str, ...]:
        return PC_RATIO_FEATURE_NAMES

    @staticmethod
    def _levels(row: PutCallRatioRow) -> np.ndarray:
        return np.asarray([
            np.log(row.put_volume / row.call_volume),
            np.log(row.put_open_interest / row.call_open_interest),
            np.log(row.put_volume + row.call_volume),
            np.log(row.put_open_interest + row.call_open_interest),
        ], dtype=float)

    def features(self, forecast_day: date, calendar: Calendar) -> np.ndarray:
        previous = calendar.previous(forecast_day)
        previous_previous = calendar.previous(previous)
        latest = self.rows.get(previous)
        prior = self.rows.get(previous_previous)
        if latest is None or prior is None:
            raise ValueError("missing previous complete TAIFEX Put/Call feature")
        cutoff = at(forecast_day, "08:30")
        if latest.day >= forecast_day or latest.available_at > cutoff:
            raise ValueError("TAIFEX Put/Call feature not available by cutoff")
        latest_levels = self._levels(latest)
        prior_levels = self._levels(prior)
        result = np.concatenate((latest_levels, latest_levels - prior_levels))
        if result.shape != (len(PC_RATIO_FEATURE_NAMES),) or not np.isfinite(result).all():
            raise ValueError("invalid TAIFEX Put/Call feature vector")
        return result


class FeatureAugmentedDataset:
    """Delegate OHLC access while extending only the Phase-4 research features."""

    def __init__(self, base, put_call: PutCallFeatureSet):
        self.base = base
        self.put_call = put_call
        self.feature_dataset_id = put_call.identity["feature_dataset_id"]

    def __getattr__(self, name):
        return getattr(self.base, name)

    def research_feature_names(self):
        return self.base.research_feature_names() + self.put_call.feature_names()

    def research_features(self, instrument, day):
        return np.concatenate((
            self.base.research_features(instrument, day),
            self.put_call.features(day, self.base.calendar),
        ))
