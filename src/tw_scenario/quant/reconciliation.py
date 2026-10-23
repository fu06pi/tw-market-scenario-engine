from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable

import numpy as np

from .contracts import OHLC, Bar


def _key(bar: Bar) -> tuple[str, str, str, str]:
    return bar.instrument, bar.contract, bar.session, str(bar.trading_date)


def _index(bars: Iterable[Bar], label: str) -> dict[tuple[str, str, str, str], Bar]:
    result = {}
    for bar in bars:
        key = _key(bar)
        if key in result:
            raise ValueError(f"duplicate {label} reconciliation key: {key}")
        result[key] = bar
    return result


def reconcile_bars(
    primary: Iterable[Bar],
    authority: Iterable[Bar],
    *,
    primary_label: str = "FinMind",
    authority_label: str = "official",
    taiex_tolerance: float = 0.011,
    tx_tolerance: float = 0.1,
) -> dict:
    """Compare source definitions without treating current fetch time as PIT evidence."""
    left, right = _index(primary, primary_label), _index(authority, authority_label)
    union = sorted(left.keys() | right.keys())
    missing = []
    mismatch_rows = []
    differences: dict[str, list[float]] = defaultdict(list)
    mismatch_counts: dict[str, int] = defaultdict(int)
    paired_counts: dict[str, int] = defaultdict(int)

    for key in union:
        if key not in left or key not in right:
            missing.append(
                {
                    "key": list(key),
                    "missing_from": primary_label if key not in left else authority_label,
                }
            )
            continue
        primary_bar, authority_bar = left[key], right[key]
        tolerance = taiex_tolerance if key[0] == "TAIEX" else tx_tolerance
        row_mismatches = {}
        for field in (*OHLC, "settlement"):
            primary_value = getattr(primary_bar, field)
            authority_value = getattr(authority_bar, field)
            if primary_value is None and authority_value is None:
                continue
            if primary_value is None or authority_value is None:
                mismatch_counts[field] += 1
                row_mismatches[field] = {
                    primary_label: primary_value,
                    authority_label: authority_value,
                    "reason": "availability_mismatch",
                }
                continue
            difference = float(primary_value - authority_value)
            paired_counts[field] += 1
            differences[field].append(difference)
            if abs(difference) > tolerance:
                mismatch_counts[field] += 1
                row_mismatches[field] = {
                    primary_label: primary_value,
                    authority_label: authority_value,
                    "difference": difference,
                    "tolerance": tolerance,
                }
        if row_mismatches:
            mismatch_rows.append({"key": list(key), "fields": row_mismatches})

    fields = {}
    for field in (*OHLC, "settlement"):
        values = np.asarray(differences[field], dtype=float)
        fields[field] = {
            "paired": paired_counts[field],
            "mismatches": mismatch_counts[field],
            "mean_signed_difference": float(values.mean()) if values.size else None,
            "mae": float(np.abs(values).mean()) if values.size else None,
            "max_abs_difference": float(np.abs(values).max()) if values.size else None,
        }

    if not union:
        status = "FAIL_EMPTY"
    elif missing or mismatch_rows:
        status = "FAIL"
    else:
        status = "PASS_RETROSPECTIVE_VALUE_AGREEMENT"
    return {
        "status": status,
        "primary_label": primary_label,
        "authority_label": authority_label,
        "primary_rows": len(left),
        "authority_rows": len(right),
        "union_keys": len(union),
        "paired_keys": len(union) - len(missing),
        "missing": missing,
        "fields": fields,
        "mismatch_rows": mismatch_rows,
        "point_in_time_verified": False,
        "note": (
            "This compares retrospective values and field availability. It does not prove "
            "what either source published by a historical 08:30 cutoff."
        ),
    }
