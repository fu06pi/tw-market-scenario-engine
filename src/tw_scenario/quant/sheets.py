from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import quote

import requests

from .contracts import digest, write_once
from .daily import (
    V0_HEADER_FINGERPRINT,
    V0_HEADERS,
    V0PostcloseSheetBatch,
    V0SheetDryRunBatch,
    preopen_row_fingerprint,
)

SHEETS_SCOPE = "https://www.googleapis.com/auth/spreadsheets"
SHEETS_API = "https://sheets.googleapis.com/v4/spreadsheets"


def authorized_session():
    """Use Application Default Credentials without reading or logging credential values."""
    try:
        import google.auth
        from google.auth.exceptions import DefaultCredentialsError
        from google.auth.transport.requests import AuthorizedSession
    except ImportError:
        raise RuntimeError("Google Sheets support requires the project sheets extra") from None
    try:
        credentials, _ = google.auth.default(scopes=[SHEETS_SCOPE])
    except DefaultCredentialsError:
        raise RuntimeError(
            "Google ADC is unavailable; configure staging-only Application Default "
            "Credentials outside the repository"
        ) from None
    return AuthorizedSession(credentials)


def _response_json(response) -> dict:
    try:
        response.raise_for_status()
        payload = response.json()
    except (requests.RequestException, TypeError, ValueError):
        raise RuntimeError("Google Sheets request failed; provider details suppressed") from None
    if not isinstance(payload, dict):
        raise TypeError("Google Sheets returned a non-object response")
    return payload


def _normalize_rows(values, rows: int, columns: int) -> list[list[str]]:
    if not isinstance(values, list):
        raise TypeError("Google Sheets values response is invalid")
    normalized = []
    for index in range(rows):
        raw = values[index] if index < len(values) else []
        if not isinstance(raw, list) or len(raw) > columns:
            raise ValueError("Google Sheets row shape is invalid")
        normalized.append([str(value) for value in raw] + [""] * (columns - len(raw)))
    return normalized


class StagingSheetWriter:
    def __init__(
        self,
        *,
        sheet_id: str,
        staging_sheet_id: str,
        production_sheet_id: str,
        session=None,
    ):
        if not sheet_id or sheet_id != staging_sheet_id:
            raise ValueError("writer is restricted to the configured staging Sheet")
        if sheet_id == production_sheet_id:
            raise ValueError("production Sheet writes are disabled")
        self.sheet_id = sheet_id
        self.session = session

    def _session(self):
        if self.session is None:
            self.session = authorized_session()
        return self.session

    def _url(self, suffix: str) -> str:
        return f"{SHEETS_API}/{self.sheet_id}/{suffix}"

    def read(self, range_name: str, rows: int, columns: int) -> list[list[str]]:
        encoded = quote(range_name, safe="")
        response = self._session().get(
            self._url(f"values/{encoded}"),
            params={"majorDimension": "ROWS", "valueRenderOption": "UNFORMATTED_VALUE"},
            timeout=30,
        )
        payload = _response_json(response)
        return _normalize_rows(payload.get("values", []), rows, columns)

    def update(self, ranges: list[dict]) -> dict:
        response = self._session().post(
            self._url("values:batchUpdate"),
            json={"valueInputOption": "RAW", "includeValuesInResponse": False, "data": ranges},
            timeout=30,
        )
        return _response_json(response)

    def verify_headers(self, header_range: str) -> None:
        observed = tuple(self.read(header_range, 1, 16)[0])
        if observed != V0_HEADERS:
            raise ValueError("staging Sheet A:P headers do not match the frozen V0 contract")

    def next_start_row(self, sheet_name: str, max_row: int = 5000) -> int:
        if max_row < 3:
            raise ValueError("max_row must leave room for two data rows")
        rows = self.read(f"'{sheet_name}'!A2:P{max_row}", max_row - 1, 16)
        last_nonempty = max(
            (index for index, row in enumerate(rows, start=2) if any(row)),
            default=1,
        )
        start = last_nonempty + 1
        if start + 1 > max_row:
            raise ValueError("staging Sheet has no room for two forecast rows")
        if any(rows[start - 2]) or any(rows[start - 1]):
            raise RuntimeError("staging next-row discovery is inconsistent")
        return start

    def preflight(self, sheet_name: str) -> dict:
        """Read-only credential, access, header, and capacity check for staging."""
        header_range = f"'{sheet_name}'!A1:P1"
        self.verify_headers(header_range)
        next_row = self.next_start_row(sheet_name)
        return {
            "operation": "STAGING_READ_ONLY_PREFLIGHT",
            "sheet_id_hash": digest(self.sheet_id),
            "sheet_name": sheet_name,
            "header_fingerprint": V0_HEADER_FINGERPRINT,
            "next_start_row": next_row,
            "rows_reserved": [next_row, next_row + 1],
            "cells_written": 0,
            "production_sheet_written": False,
        }

    @staticmethod
    def _row_bounds(write_range: str) -> tuple[str, int, int]:
        match = re.fullmatch(r"'([^']+)'!A(\d+):P(\d+)", write_range)
        if match is None:
            raise ValueError("unsupported Sheet write range")
        start, end = int(match.group(2)), int(match.group(3))
        if start < 2 or end != start + 1:
            raise ValueError("morning batch must target exactly two consecutive data rows")
        return match.group(1), start, end

    def write_morning(self, batch: V0SheetDryRunBatch) -> dict:
        self.verify_headers(batch.header_range)
        sheet_name, start, end = self._row_bounds(batch.write_range)
        before = self.read(batch.write_range, 2, 16)
        intended = [list(row.values) for row in batch.rows]
        reused = before == intended
        if not reused and any(any(cell for cell in row) for row in before):
            raise ValueError("staging target rows are not empty and do not match this forecast")
        if not reused:
            self.update([
                {
                    "range": f"'{sheet_name}'!A{start}:F{end}",
                    "majorDimension": "ROWS",
                    "values": [row[:6] for row in intended],
                },
                {
                    "range": f"'{sheet_name}'!O{start}:O{end}",
                    "majorDimension": "ROWS",
                    "values": [[row[14]] for row in intended],
                },
            ])
        after = self.read(batch.write_range, 2, 16)
        if after != intended:
            raise RuntimeError("staging morning Sheet readback mismatch")
        return {
            "operation": "MORNING_STAGING_WRITE",
            "forecast_id": batch.forecast_id,
            "snapshot_hash": batch.snapshot_hash,
            "sheet_id_hash": digest(self.sheet_id),
            "write_range": batch.write_range,
            "reused": reused,
            "row_fingerprints": [preopen_row_fingerprint(row) for row in after],
            "production_sheet_written": False,
        }

    def write_postclose(self, batch: V0PostcloseSheetBatch) -> dict:
        start, end = batch.rows[0].row_number, batch.rows[-1].row_number
        full_range = f"'{batch.sheet_name}'!A{start}:P{end}"
        before = self.read(full_range, 2, 16)
        fingerprints = [preopen_row_fingerprint(row) for row in before]
        expected = [row.preopen_fingerprint for row in batch.rows]
        if fingerprints != expected:
            raise ValueError("staging pre-open cells changed before post-close update")
        intended_updates = [list(row.values) for row in batch.rows]
        existing_updates = [row[6:14] + [row[15]] for row in before]
        reused = existing_updates == intended_updates
        if not reused and any(any(cell for cell in row) for row in existing_updates):
            raise ValueError("staging post-close cells are already populated differently")
        if not reused:
            self.update([
                {
                    "range": f"'{batch.sheet_name}'!G{start}:N{end}",
                    "majorDimension": "ROWS",
                    "values": [row[:8] for row in intended_updates],
                },
                {
                    "range": f"'{batch.sheet_name}'!P{start}:P{end}",
                    "majorDimension": "ROWS",
                    "values": [[row[8]] for row in intended_updates],
                },
            ])
        after = self.read(full_range, 2, 16)
        if [preopen_row_fingerprint(row) for row in after] != expected:
            raise RuntimeError("staging post-close write altered immutable morning cells")
        if [row[6:14] + [row[15]] for row in after] != intended_updates:
            raise RuntimeError("staging post-close Sheet readback mismatch")
        return {
            "operation": "POSTCLOSE_STAGING_WRITE",
            "forecast_id": batch.forecast_id,
            "snapshot_hash": batch.snapshot_hash,
            "sheet_id_hash": digest(self.sheet_id),
            "write_range": full_range,
            "reused": reused,
            "preopen_fingerprints": expected,
            "production_sheet_written": False,
        }


def save_receipt(receipt: dict, path: Path) -> Path:
    receipt = {**receipt, "receipt_hash": digest(receipt)}
    if path.exists():
        import json

        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != receipt:
            raise FileExistsError(f"refusing to replace different Sheet receipt: {path}")
        return path
    write_once(path, receipt)
    return path
