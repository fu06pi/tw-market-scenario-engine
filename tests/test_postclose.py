"""Synthetic outcomes and clocks. Live download evidence is reported separately."""

import csv
import hashlib
import io
import json
import zipfile
from datetime import date

import pytest

from scripts.finish_shadow_day import materialize
from tests.test_shadow import stamp
from tw_scenario import postclose
from tw_scenario.snapshot import verify_document


def official_response(market, actual_date="115/11/05", contract="202611"):
    if market == "TAIEX":
        return json.dumps({"stat": "OK", "fields": ["日期", "開盤指數", "最高指數", "最低指數", "收盤指數"],
                           "data": [[actual_date, "100", "103", "98", "101"]]}).encode()
    stream = io.StringIO()
    writer = csv.writer(stream)
    writer.writerow(['交易日期', '契約', '到期月份(週別)', '開盤價', '最高價', '最低價', '收盤價',
                     '成交量', '結算價', '交易時段'])
    writer.writerow(['2026/11/05', 'TX', contract, 100, 103, 98, 101, 200, 999, '一般'])
    return stream.getvalue().encode()


def install_fetch(monkeypatch, cash_date="115/11/05", contract="202611"):
    at = stamp(date(2026, 11, 5), 14, 15)
    monkeypatch.setattr(postclose, "now", lambda: at)

    def fetch(url, directory, form=None):
        market = "TAIEX" if "twse" in url else "TX"
        return official_response(market, cash_date, contract), {
            "url": url, "available_at": at.isoformat(), "raw_sha256": "synthetic",
            "availability_basis": "synthetic test receipt",
        }
    monkeypatch.setattr(postclose, "fetch", fetch)


def test_exact_day_contract_last_transaction_and_receipt_are_preserved(monkeypatch, tmp_path):
    install_fetch(monkeypatch)
    result = postclose.collect_day(date(2026, 11, 5), "202611", tmp_path / "day")
    assert result["status"] == "complete" and verify_document(result).valid
    assert len(result["actuals"]) == 2
    assert all(r["close"] == 101 and r["session"] == "day" for r in result["actuals"])
    assert result["actuals"][1]["contract"] == "202611"
    assert all(r["available_at"].startswith("2026-11-05T14:15") for r in result["actuals"])
    with pytest.raises(FileExistsError):
        postclose.collect_day(date(2026, 11, 5), "202611", tmp_path / "day")


def test_missing_cash_or_contract_never_substitutes_nearest_available(monkeypatch, tmp_path):
    install_fetch(monkeypatch, cash_date="115/11/04", contract="202612")
    result = postclose.collect_day(date(2026, 11, 5), "202611", tmp_path / "missing")
    assert result["actuals"] == [] and len(result["failures"]) == 2
    assert result["status"] == "partial_or_failed"


def test_unfinished_daytime_outcome_collection_is_rejected(monkeypatch, tmp_path):
    monkeypatch.setattr(postclose, "now", lambda: stamp(date(2026, 11, 5), 13, 50))
    with pytest.raises(ValueError, match="14:00"):
        postclose.collect_day(date(2026, 11, 5), "202611", tmp_path / "early")
    assert not (tmp_path / "early").exists()


def test_archive_extraction_rejects_digest_mismatch_and_traversal(tmp_path):
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as zipped:
        zipped.writestr("../outside.json", "{}");
    archive = stream.getvalue()
    with pytest.raises(ValueError, match="digest"):
        materialize({"digest": "sha256:wrong"}, archive, tmp_path / "bad-digest")
    with pytest.raises(ValueError, match="unsafe"):
        materialize({"digest": "sha256:" + hashlib.sha256(archive).hexdigest()}, archive,
                    tmp_path / "bad-path")
    assert not (tmp_path / "outside.json").exists()
