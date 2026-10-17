"""Verified official night-session HTML, explicit missing quotes and session dates."""

from __future__ import annotations

import json
import re
import time
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlencode
from zoneinfo import ZoneInfo

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .data import write_new


class Tables(HTMLParser):
    def __init__(self):
        super().__init__()
        self.rows = []
        self.row = None
        self.cell = None

    def handle_starttag(self, tag, attrs):
        if tag == "tr":
            self.row = []
        elif tag in ("th", "td") and self.row is not None:
            self.cell = ""

    def handle_data(self, data):
        if self.cell is not None:
            self.cell += data

    def handle_endtag(self, tag):
        if tag in ("th", "td") and self.cell is not None:
            if self.row is not None:
                self.row.append(" ".join(self.cell.split()))
            self.cell = None
        elif tag == "tr" and self.row is not None:
            self.rows.append(self.row)
            self.row = None


def parse_night(html: str, day: str, month: str = "202603") -> dict:
    markers = re.findall(r'name="queryDate"[^>]*value="([^"]+)"', html)
    if day.replace("-", "/") not in markers:
        raise ValueError("official response date differs")
    parser = Tables()
    parser.feed(html)
    headers = next(
        (row for row in parser.rows if "最後成交價" in row and "到期月份(週別)" in row), None
    )
    if headers is None:
        return {"date": day, "close": None, "reason": "no_report_table"}
    if "*成交量" not in headers:
        raise ValueError("expected night session schema, not daytime combined volume")
    rows = [row for row in parser.rows if len(row) == len(headers) and row[:2] == ["TX", month]]
    if not rows:
        return {"date": day, "close": None, "reason": "no_contract_quote"}
    if len(rows) != 1:
        raise ValueError("ambiguous contract rows")
    value = rows[0][headers.index("最後成交價")]
    volume = rows[0][headers.index("*成交量")]
    if value == "-" or volume == "-" or float(volume.replace(",", "")) <= 0:
        return {"date": day, "close": None, "reason": "no_night_trade"}
    price = float(value.replace(",", ""))
    if price <= 0:
        raise ValueError("invalid night quote")
    return {
        "date": day,
        "contract": "TX" + month,
        "close": price,
        "volume": float(volume.replace(",", "")),
        "reason": "official_night_quote",
    }


def fetch_night(cache: Path, day: str) -> dict:
    path = cache / f"{day}-TX202603-night.json"
    if path.exists():
        return json.loads(path.read_text())
    url = "https://www.taifex.com.tw/cht/3/futDailyMarketReport?" + urlencode(
        {"queryType": 2, "marketCode": 1, "commodity_id": "TX", "queryDate": day.replace("-", "/")}
    )
    session = requests.Session()
    session.mount(
        "https://",
        HTTPAdapter(max_retries=Retry(total=2, backoff_factor=1, status_forcelist=[502, 503, 504])),
    )
    try:
        response = session.get(url, timeout=(10, 30))
        if response.status_code == 429:
            wait = response.headers.get("Retry-After", "15")
            if wait.isdigit():
                seconds = float(wait)
            else:
                try:
                    seconds = max(
                        0.0, (parsedate_to_datetime(wait) - datetime.now(UTC)).total_seconds()
                    )
                except (TypeError, ValueError):
                    seconds = 15.0
            if seconds > 30:
                raise RuntimeError("TAIFEX rate limit; resume after published Retry-After")
            time.sleep(max(seconds, 15.0))
            response = session.get(url, timeout=(10, 30))
        response.raise_for_status()
        response.encoding = "utf-8"
        result = parse_night(response.text, day)
    finally:
        session.close()
    result.update(
        source=url,
        fetched_at=datetime.now(ZoneInfo("Asia/Taipei")).isoformat(),
        availability_basis="Official note assigns afternoon D-1 to 05:00 D to exchange trading date D; historical publication delay not verified",
        historical_publication_verified=False,
    )
    write_new(path, result)
    return result
