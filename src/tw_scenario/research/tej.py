"""TEJ SDK transport: preserve proxy/authentication; bounded queries, sanitized errors."""

from __future__ import annotations

import os

from tejapi.api_config import ApiConfig
from tejapi.connection import Connection


class AccessError(RuntimeError):
    pass


def request(method: str, path: str, params: dict | None = None) -> dict:
    if not os.getenv("TEJ_API_KEY"):
        raise AccessError("TEJ_API_KEY missing")
    ApiConfig.api_key = os.environ["TEJ_API_KEY"]
    try:
        value = Connection.request(method, path, params=params or {}, timeout=(10, 30)).json()
    except Exception as error:  # noqa: BLE001 -- SDK messages can contain credential-bearing requests
        raise AccessError(type(error).__name__) from None
    if "error" in value:
        raise AccessError("TEJ " + str(value["error"].get("code", "unknown")))
    return value


def metadata(table: str) -> dict:
    return request("get", f"datatables/{table}/metadata")["datatable"]


def get_rows(table: str, identifier: str, columns: list[str], max_pages: int = 5) -> list[dict]:
    params = {"coid": identifier, "opts.columns[]": columns, "opts.per_page": 1000}
    rows = []
    for _ in range(max_pages):
        result = request("post", f"datatables/{table}", params)
        names = [c["name"] for c in result["datatable"]["columns"]]
        if set(names) != set(columns):
            raise ValueError("unexpected TEJ schema")
        rows.extend(dict(zip(names, row, strict=True)) for row in result["datatable"]["data"])
        cursor = result["meta"]["next_cursor_id"]
        if cursor is None:
            return rows
        params["opts.cursor_id"] = cursor
    raise AccessError("incomplete pagination: query budget exhausted")
