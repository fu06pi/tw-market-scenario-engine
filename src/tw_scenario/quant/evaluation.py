from __future__ import annotations

import numpy as np

from .contracts import OHLC


def direction(x, threshold=5.):
    return 'up' if x > threshold else 'down' if x < -threshold else 'flat'


def metrics(rows: list[dict], threshold=5.) -> dict:
    good = [r for r in rows if r['status'] == 'OK']
    out = {'expected_days': len(rows), 'success_days': len(good),
           'forecast_coverage': len(good) / len(rows) if rows else 0.0,
           'failed_days': [
               {'date': r['date'], 'reason': r.get('error')}
               for r in rows if r['status'] != 'OK'
           ]}
    if not good:
        return out
    a, p = [np.array([[r[k][v] for v in OHLC] for r in good]) for k in ('actual', 'predicted')]
    refs = np.array([r['previous_close'] for r in good])[:,None]
    err = p-a
    out['ohlc'] = {k: {'mae_points': float(np.mean(abs(err[:,j]))),
                       'mae_bps': float(np.mean(abs(err[:,j])/refs[:,0])*10000),
                       'rmse_points': float(np.sqrt(np.mean(err[:,j]**2)))}
                   for j,k in enumerate(OHLC)}
    out['gap_mae_bps'] = float(np.mean(abs(err[:,0])/refs[:,0])*10000)
    out['close_return_mae_bps'] = float(np.mean(abs(err[:,3])/refs[:,0])*10000)
    out['range_mae_bps'] = float(np.mean(abs((p[:,1]-p[:,2])-(a[:,1]-a[:,2]))/refs[:,0])*10000)
    labels = ['down','flat','up']
    cm = {a: {p: 0 for p in labels} for a in labels}
    base = {a: {p: 0 for p in labels} for a in labels}
    for actual, predicted in zip((a[:,3]/refs[:,0]-1)*10000, (p[:,3]/refs[:,0]-1)*10000):
        cm[direction(actual,threshold)][direction(predicted,threshold)] += 1
        base[direction(actual,threshold)]['flat'] += 1
    out['direction'] = {'neutral_bps': threshold, 'actual_rows_predicted_columns': cm, 'zero_return_baseline': base}
    ir = [r for r in good if r.get('interval')]
    out['interval_days'] = len(ir)
    out['intervals'] = {}
    for k in OHLC:
        if ir:
            lo, hi, actual = np.array([[r['interval'][k][0], r['interval'][k][1], r['actual'][k]] for r in ir]).T
            score = hi-lo+10*(lo-actual)*(actual<lo)+10*(actual-hi)*(actual>hi)
            out['intervals'][k] = {'coverage': float(np.mean((actual>=lo)&(actual<=hi))),
                                   'width': float(np.mean(hi-lo)), 'interval_score': float(np.mean(score))}
    return out


def block_improvement(candidate, baseline, block=20, draws=1000, seed=4655):
    c = {r['date']:r for r in candidate if r['status']=='OK'}
    b = {r['date']:r for r in baseline if r['status']=='OK'}
    dates = sorted(c.keys() & b.keys())
    if len(dates) < 2*block:
        return {'status': 'INSUFFICIENT_DATA', 'paired_days': len(dates)}
    ce = np.array([abs(c[d]['predicted']['close']-c[d]['actual']['close']) for d in dates])
    be = np.array([abs(b[d]['predicted']['close']-b[d]['actual']['close']) for d in dates])
    rng, estimates = np.random.default_rng(seed), []
    for _ in range(draws):
        starts = rng.integers(0,len(dates)-block+1,size=int(np.ceil(len(dates)/block)))
        ix = np.concatenate([np.arange(s,s+block) for s in starts])[:len(dates)]
        if np.mean(be[ix]) > 0:
            estimates.append(1-np.mean(ce[ix])/np.mean(be[ix]))
    return {'paired_days': len(dates), 'block_length': block,
            'close_relative_improvement_ci95': np.quantile(estimates,[.025,.975]).tolist() if estimates else None}


def _diagnostic_group(candidate: dict, baseline: dict, dates: list[str]) -> dict:
    candidate_metrics = metrics([candidate[day] for day in dates])
    baseline_metrics = metrics([baseline[day] for day in dates])
    candidate_close = candidate_metrics["ohlc"]["close"]["mae_points"]
    baseline_close = baseline_metrics["ohlc"]["close"]["mae_points"]
    return {
        "paired_days": len(dates),
        "candidate_ohlc_mae_points": {
            field: candidate_metrics["ohlc"][field]["mae_points"] for field in OHLC
        },
        "baseline_ohlc_mae_points": {
            field: baseline_metrics["ohlc"][field]["mae_points"] for field in OHLC
        },
        "close_relative_improvement": (
            1.0 - candidate_close / baseline_close if baseline_close > 0 else None
        ),
        "candidate_interval_coverage": {
            field: candidate_metrics["intervals"][field]["coverage"] for field in OHLC
        },
    }


def stability_breakdown(
    candidate_rows: list[dict],
    baseline_rows: list[dict],
    *,
    event_labels: dict[str, str] | None = None,
) -> dict:
    """Post-hoc final diagnostics only; never use these groups for model selection."""
    candidate_all = {row["date"]: row for row in candidate_rows}
    baseline_all = {row["date"]: row for row in baseline_rows}
    paired = sorted(
        day
        for day in candidate_all.keys() & baseline_all.keys()
        if candidate_all[day]["status"] == "OK" and baseline_all[day]["status"] == "OK"
    )
    for day in paired:
        if candidate_all[day]["actual"] != baseline_all[day]["actual"]:
            raise ValueError(f"candidate/baseline actual mismatch on {day}")

    labels: dict[str, dict[str, list[str]]] = {
        "calendar_year": {},
        "realized_range_regime": {
            "low_lt_100bp": [],
            "normal_100_200bp": [],
            "high_ge_200bp": [],
        },
    }
    for day in paired:
        row = candidate_all[day]
        labels["calendar_year"].setdefault(day[:4], []).append(day)
        range_bps = (
            (row["actual"]["high"] - row["actual"]["low"])
            / row["previous_close"]
            * 10_000
        )
        regime = (
            "low_lt_100bp"
            if range_bps < 100
            else "normal_100_200bp"
            if range_bps < 200
            else "high_ge_200bp"
        )
        labels["realized_range_regime"][regime].append(day)

    instrument = candidate_rows[0]["instrument"] if candidate_rows else None
    if instrument == "TX":
        rollover = {"rollover": [], "non_rollover": [], "boundary_unknown": []}
        prior_contract = None
        for day in paired:
            contract = candidate_all[day]["contract"]
            if prior_contract is None:
                rollover["boundary_unknown"].append(day)
            elif contract != prior_contract:
                rollover["rollover"].append(day)
            else:
                rollover["non_rollover"].append(day)
            prior_contract = contract
        labels["contract_rollover"] = rollover

    groups = {
        dimension: {
            label: _diagnostic_group(candidate_all, baseline_all, days)
            for label, days in members.items()
            if days
        }
        for dimension, members in labels.items()
    }
    if event_labels is None:
        event_result = {
            "status": "NOT_EVALUATED_MISSING_TIMESTAMPED_EVENT_CALENDAR",
            "reason": "No frozen point-in-time major-event labels were supplied.",
        }
    else:
        unknown = sorted(set(event_labels) - set(candidate_all))
        if unknown:
            raise ValueError("event calendar contains dates outside the final rows")
        event_groups: dict[str, list[str]] = {"non_event": []}
        for day in paired:
            label = event_labels.get(day, "non_event")
            if not isinstance(label, str) or not label.strip():
                raise ValueError("event labels must be non-empty strings")
            event_groups.setdefault(label, []).append(day)
        event_result = {
            "status": "EVALUATED_POST_HOC",
            "groups": {
                label: _diagnostic_group(candidate_all, baseline_all, days)
                for label, days in event_groups.items()
                if days
            },
        }

    return {
        "instrument": instrument,
        "paired_days": len(paired),
        "candidate_failed_days": [
            {"date": row["date"], "reason": row.get("error")}
            for row in candidate_rows
            if row["status"] != "OK"
        ],
        "baseline_failed_days": [
            {"date": row["date"], "reason": row.get("error")}
            for row in baseline_rows
            if row["status"] != "OK"
        ],
        "group_definitions": {
            "calendar_year": "calendar year of the final-test trading date",
            "realized_range_regime": (
                "post-hoc actual (H-L)/previous_close: low <100bp, "
                "normal 100-200bp, high >=200bp"
            ),
            "contract_rollover": (
                "TX contract differs from the preceding paired final-test row; "
                "first row is boundary_unknown"
                if instrument == "TX"
                else "not applicable to TAIEX"
            ),
        },
        "groups": groups,
        "major_event": event_result,
    }


def build_final_stability_audit(
    baseline_result: dict,
    candidate_result: dict,
    *,
    event_labels: dict[str, str] | None = None,
) -> dict:
    if baseline_result.get("dataset_id") != candidate_result.get("dataset_id"):
        raise ValueError("baseline and candidate dataset_id differ")
    baseline_split = baseline_result.get("splits", {}).get("final_test")
    if baseline_split != candidate_result.get("final_test_split"):
        raise ValueError("baseline and candidate final-test splits differ")
    if candidate_result.get("final_test_must_not_be_reused_for_tuning") is not True:
        raise ValueError("candidate result does not preserve the closed final-test contract")

    instruments = {}
    for instrument in ("TAIEX", "TX"):
        instruments[instrument] = stability_breakdown(
            candidate_result["final_test"][instrument]["rows"],
            baseline_result["final_test"][instrument]["rows"],
            event_labels=event_labels,
        )
    from .contracts import digest

    return {
        "schema_version": "final-stability-diagnostics-v1",
        "status": "POST_HOC_DIAGNOSTIC_ONLY",
        "dataset_id": candidate_result["dataset_id"],
        "final_test_split": candidate_result["final_test_split"],
        "final_test_closed_to_tuning": True,
        "candidate_acceptance_status": candidate_result["acceptance_gate"]["status"],
        "source_hashes": {
            "baseline_result": digest(baseline_result),
            "candidate_result": digest(candidate_result),
        },
        "interpretation": (
            "These groups describe the already-closed final test. They were not "
            "predeclared selection gates and must not be used to retune or relabel it."
        ),
        "instruments": instruments,
    }
