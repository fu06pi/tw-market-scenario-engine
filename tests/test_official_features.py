from __future__ import annotations

import base64
import csv
import hashlib
import io
import json
from datetime import date, datetime

import pytest

from tw_scenario.quant.contracts import TZ, Calendar, at, digest
from tw_scenario.quant.official_features import (
    PC_RATIO_HEADER,
    TAIFEX_PC_RATIO_URL,
    PutCallFeatureSet,
    import_pc_ratio_browser_bundle,
    parse_pc_ratio_cache,
    pc_ratio_quality,
)


def _csv_bytes(rows):
    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\r\n")
    writer.writerow(PC_RATIO_HEADER)
    for row in rows:
        writer.writerow([*row, ""])
    return output.getvalue().encode("cp950")


def _bundle(raw):
    return {
        "version": "taifex-pcratio-browser-bundle-v1",
        "year": 2025,
        "items": [{
            "request": {
                "url": TAIFEX_PC_RATIO_URL,
                "form": {
                    "queryStartDate": "2025/12/01",
                    "queryEndDate": "2025/12/31",
                    "down_type": "",
                },
            },
            "fetched_at": "2026-01-02T04:00:00.000Z",
            "content_type": "text/html;charset=MS950",
            "bytes_base64": base64.b64encode(raw).decode(),
        }],
    }


def test_browser_bundle_import_is_immutable_and_recomputes_ratios(tmp_path):
    raw = _csv_bytes([
        ("2025/12/02", 120, 100, "120.00", 240, 200, "120.00"),
        ("2025/12/01", 100, 100, "100.00", 200, 200, "100.00"),
    ])
    bundle_path = tmp_path / "bundle.json"
    bundle_path.write_text(json.dumps(_bundle(raw)), encoding="utf-8")

    paths = import_pc_ratio_browser_bundle(bundle_path, tmp_path / "raw")
    repeated = import_pc_ratio_browser_bundle(bundle_path, tmp_path / "raw")
    assert repeated == paths
    request = _bundle(raw)["items"][0]["request"]
    key = digest(request)
    assert paths == [tmp_path / "raw" / f"{key}.csv"]
    manifest = json.loads((tmp_path / "raw" / f"{key}.manifest.json").read_text())
    assert manifest["sha256"] == hashlib.sha256(raw).hexdigest()
    assert manifest["transport"] == "browser_cdp_same_origin"

    rows = parse_pc_ratio_cache(paths[0], tmp_path / "raw" / f"{key}.manifest.json")
    assert [row.day for row in rows] == [date(2025, 12, 1), date(2025, 12, 2)]
    assert rows[0].available_at == at(date(2025, 12, 1), "18:00")
    assert rows[0].source_hash == hashlib.sha256(raw).hexdigest()


def test_ratio_tampering_and_cross_month_request_fail_closed(tmp_path):
    raw = _csv_bytes([
        ("2025/12/01", 120, 100, "99.00", 240, 200, "120.00"),
    ])
    bundle = _bundle(raw)
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(bundle), encoding="utf-8")
    with pytest.raises(ValueError, match="recomputation"):
        import_pc_ratio_browser_bundle(path, tmp_path / "raw")

    bundle["items"][0]["request"]["form"]["queryEndDate"] = "2026/01/01"
    path.write_text(json.dumps(bundle), encoding="utf-8")
    with pytest.raises(ValueError, match="bounded calendar month"):
        import_pc_ratio_browser_bundle(path, tmp_path / "other")


def test_previous_complete_day_features_and_quality_are_causal():
    rows = []
    for index, day in enumerate((date(2025, 12, 1), date(2025, 12, 2)), start=1):
        from tw_scenario.quant.official_features import PutCallRatioRow
        rows.append(PutCallRatioRow(
            day=day,
            put_volume=100 + index,
            call_volume=100,
            volume_ratio_pct=(100 + index),
            put_open_interest=200 + index,
            call_open_interest=200,
            open_interest_ratio_pct=(200 + index) / 2,
            available_at=at(day, "18:00"),
            availability_basis="synthetic test",
            source=TAIFEX_PC_RATIO_URL,
            source_hash="0" * 64,
        ))
    calendar = Calendar(
        days=[date(2025, 12, 1), date(2025, 12, 2), date(2025, 12, 3)],
        expiries={"202512": date(2025, 12, 17)},
        source="synthetic test",
        available_at=datetime(2025, 12, 1, tzinfo=TZ),
    )
    feature_set = PutCallFeatureSet(rows, {"feature_dataset_id": "x"})
    values = feature_set.features(date(2025, 12, 3), calendar)
    assert values.shape == (8,)
    assert values[4] > 0

    report = pc_ratio_quality(rows, calendar, date(2025, 12, 1), date(2025, 12, 2))
    assert report["status"] == "PASS"
    assert report["coverage"] == 1

    late = rows[-1].__class__(**{
        **rows[-1].__dict__,
        "available_at": at(date(2025, 12, 3), "09:00"),
    })
    with pytest.raises(ValueError, match="not available by cutoff"):
        PutCallFeatureSet([rows[0], late], {"feature_dataset_id": "x"}).features(
            date(2025, 12, 3), calendar
        )
