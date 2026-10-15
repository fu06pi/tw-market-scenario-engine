from __future__ import annotations

from urllib.parse import unquote

import pytest

from tw_scenario.quant.daily import (
    V0_HEADERS,
    V0MorningSheetRow,
    V0PostcloseSheetBatch,
    V0PostcloseSheetRow,
    V0SheetDryRunBatch,
    preopen_row_fingerprint,
)
from tw_scenario.quant.sheets import StagingSheetWriter

FORECAST_ID = "ohlc-v1-2026-10-16-" + "1" * 64
SNAPSHOT_HASH = "2" * 64


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


class FakeSession:
    def __init__(self):
        self.rows = [[""] * 16 for _ in range(2)]
        self.posts = 0

    def get(self, url, **_kwargs):
        encoded = url.split("/values/", 1)[1]
        range_name = unquote(encoded)
        if range_name.endswith("A1:P1"):
            return FakeResponse({"values": [list(V0_HEADERS)]})
        return FakeResponse({"values": self.rows})

    def post(self, _url, json, **_kwargs):
        self.posts += 1
        for update in json["data"]:
            range_name = update["range"]
            values = update["values"]
            if "!A" in range_name:
                for index, row in enumerate(values):
                    self.rows[index][:6] = row
            elif "!O" in range_name:
                for index, row in enumerate(values):
                    self.rows[index][14] = row[0]
            elif "!G" in range_name:
                for index, row in enumerate(values):
                    self.rows[index][6:14] = row
            elif "!P" in range_name:
                for index, row in enumerate(values):
                    self.rows[index][15] = row[0]
            else:
                raise AssertionError(f"unexpected range {range_name}")
        return FakeResponse({"totalUpdatedRows": 2})


def _morning_batch() -> V0SheetDryRunBatch:
    rows = []
    for instrument in ("TAIEX", "TX"):
        values = [
            "2026/10/16",
            instrument,
            "hypothesis",
            "偏多",
            "no probability",
            "evidence",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            f"forecast_id={FORECAST_ID}",
            "",
        ]
        rows.append(V0MorningSheetRow(values=tuple(values)))
    return V0SheetDryRunBatch(
        forecast_id=FORECAST_ID,
        snapshot_hash=SNAPSHOT_HASH,
        write_range="'每日預測與覆盤'!A42:P43",
        rows=tuple(rows),
    )


def test_staging_writer_morning_and_postclose_readback_is_idempotent():
    session = FakeSession()
    writer = StagingSheetWriter(
        sheet_id="staging",
        staging_sheet_id="staging",
        production_sheet_id="production",
        session=session,
    )
    morning = _morning_batch()
    receipt = writer.write_morning(morning)
    assert receipt["operation"] == "MORNING_STAGING_WRITE"
    assert receipt["reused"] is False
    assert writer.write_morning(morning)["reused"] is True
    assert writer.next_start_row("每日預測與覆盤", max_row=10) == 4

    postclose = V0PostcloseSheetBatch(
        forecast_id=FORECAST_ID,
        snapshot_hash=SNAPSHOT_HASH,
        rows=tuple(
            V0PostcloseSheetRow(
                row_number=42 + index,
                instrument=instrument,
                contract=None if instrument == "TAIEX" else "202610",
                preopen_fingerprint=preopen_row_fingerprint(morning.rows[index].values),
                values=tuple(f"{instrument}-{column}" for column in range(9)),
            )
            for index, instrument in enumerate(("TAIEX", "TX"))
        ),
    )
    receipt = writer.write_postclose(postclose)
    assert receipt["operation"] == "POSTCLOSE_STAGING_WRITE"
    assert receipt["reused"] is False
    assert writer.write_postclose(postclose)["reused"] is True
    assert session.posts == 2


def test_staging_writer_rejects_production_or_unconfigured_sheet():
    with pytest.raises(ValueError, match="configured staging"):
        StagingSheetWriter(
            sheet_id="production",
            staging_sheet_id="staging",
            production_sheet_id="production",
            session=FakeSession(),
        )
