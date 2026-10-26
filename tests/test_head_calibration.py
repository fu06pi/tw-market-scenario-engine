import hashlib
import json
from copy import deepcopy
from datetime import date, datetime, time, timedelta

import pytest

from tw_scenario.quant.contracts import OHLC, TZ, digest
from tw_scenario.quant.head_calibration import (
    apply_head_calibration,
    build_head_calibration_artifact,
    calibrate_head_intervals,
    load_head_calibration_artifact,
    verify_head_calibration_availability,
)

CALIBRATION = {
    'version': 'causal-head-marginal-residual-v1',
    'candidate_context': 64,
    'window': 4,
    'minimum': 4,
    'lower_quantile': 0.1,
    'upper_quantile': 0.9,
    'normalization': 'prediction_day_same_contract_previous_close',
    'update_policy': 'previous_walk_forward_residuals_available_by_0830_only',
    'point_policy': 'unchanged',
    'interval_policy': 'marginal_only_not_joint_OHLC_quantiles',
}


def fixture_rows(count=8):
    rows, times = [], {}
    for i in range(count):
        day = date(2026, 1, 1) + timedelta(days=i)
        rows.append({'date': str(day), 'status': 'OK', 'previous_close': 100.,
                     'predicted': {f: 100. for f in OHLC},
                     'actual': {f: 100. + i for f in OHLC},
                     'interval': {f: [90., 110.] for f in OHLC}})
        times[str(day)] = datetime.combine(day, time(14), TZ)
    return rows, times


def test_calibration_uses_only_past_published_labels_and_keeps_warmup():
    rows, times = fixture_rows()
    before = deepcopy(rows)
    output = calibrate_head_intervals(rows, times, window=4, minimum=2)
    assert rows == before
    assert len(output) == len(rows)
    assert output[1]['interval'] == rows[1]['interval']
    assert output[2]['interval']['close'] == pytest.approx([100., 100.9])
    changed = deepcopy(rows)
    for row in changed[2:]:
        row['actual'] = {f: 900. for f in OHLC}
    counterfactual = calibrate_head_intervals(changed, times, window=4, minimum=2)
    assert output[2]['interval'] == counterfactual[2]['interval']
    assert all(r['predicted'] == original['predicted'] for r, original in zip(output, rows))
    times[rows[1]['date']] += timedelta(days=10)
    delayed = calibrate_head_intervals(rows, times, window=4, minimum=2)
    assert delayed[2]['calibration']['samples'] == 1
    assert delayed[2]['calibration']['status'] == 'WARMUP_ORIGINAL_INTERVAL'


def test_calibration_preserves_failures_and_rejects_bad_order_and_availability():
    rows, times = fixture_rows()
    rows[1] = {'date': rows[1]['date'], 'status': 'FAILED', 'error': 'missing features'}
    output = calibrate_head_intervals(rows, times, window=4, minimum=2)
    assert output[1]['error'] == 'missing features'
    assert output[1]['status'] == 'FAILED'
    assert output[2]['calibration']['samples'] == 1
    with pytest.raises(ValueError, match='chronological'):
        calibrate_head_intervals(rows[::-1], times)
    times[rows[0]['date']] -= timedelta(days=1)
    with pytest.raises(ValueError, match='after forecast cutoff'):
        calibrate_head_intervals(rows, times)


def test_daily_artifact_uses_only_residuals_available_by_cutoff_and_fails_warmup(tmp_path):
    rows, times = fixture_rows()
    for row in rows:
        row['instrument'] = 'TAIEX'
    identity = {
        'instrument': 'TAIEX',
        'spec': {'context': 64},
        'folds': [{'validation': ['2026-01-01', '2026-01-08']}],
    }
    source_path = tmp_path / 'result.json'
    source_path.write_text(json.dumps({'identity': identity}))
    source_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()
    fold_path = tmp_path / 'fold-1.json'
    fold_path.write_text(json.dumps({
        'identity_hash': digest(identity),
        'fold': 1,
        'candidate_rows': rows,
    }))
    fold_hash = hashlib.sha256(fold_path.read_bytes()).hexdigest()
    artifact = build_head_calibration_artifact(
        rows,
        times,
        instrument='TAIEX',
        source_identity=identity,
        source_result_sha256=source_hash,
        source_fold_sha256={'fold-1.json': fold_hash},
        calibration=CALIBRATION,
    )
    artifact_path = tmp_path / 'calibration-artifact.json'
    artifact_path.write_text(json.dumps(artifact))
    loaded, artifact_hash = load_head_calibration_artifact(
        artifact_path,
        instrument='TAIEX',
        source_identity=identity,
        source_result_sha256=source_hash,
        source_dir=tmp_path,
    )
    assert artifact_hash == hashlib.sha256(artifact_path.read_bytes()).hexdigest()

    class CertifiedData:
        def actual(self, instrument, day):
            assert instrument == 'TAIEX'
            return type('Actual', (), {'available_at': times[str(day)]})()

    verify_head_calibration_availability(
        CertifiedData(), 'TAIEX', loaded, date(2026, 1, 9)
    )
    wrong_availability = deepcopy(loaded)
    wrong_availability['residuals'][0]['available_at'] = (
        times[rows[0]['date']] + timedelta(hours=1)
    ).isoformat()
    with pytest.raises(ValueError, match='differs from certified data'):
        verify_head_calibration_availability(
            CertifiedData(), 'TAIEX', wrong_availability, date(2026, 1, 9)
        )
    # A forged future timestamp must not silently remove an older residual.
    wrong_availability['residuals'][0]['available_at'] = datetime(
        2026, 1, 10, 14, tzinfo=TZ
    ).isoformat()
    with pytest.raises(ValueError, match='differs from certified data'):
        verify_head_calibration_availability(
            CertifiedData(), 'TAIEX', wrong_availability, date(2026, 1, 9)
        )

    interval, usage = apply_head_calibration(
        {field: 100. for field in OHLC}, 100., loaded, date(2026, 1, 7)
    )
    assert interval['close'] == pytest.approx([100., 104.7])
    assert usage == {
        'samples': 4,
        'latest_label_date': '2026-01-06',
        'window': 4,
        'minimum': 4,
    }
    future_changed = deepcopy(loaded)
    future_changed['residuals'][-1]['normalized_signed_residual'] = {
        field: 900. for field in OHLC
    }
    assert apply_head_calibration(
        {field: 100. for field in OHLC}, 100., future_changed, date(2026, 1, 7)
    )[0] == interval
    delayed = deepcopy(loaded)
    delayed['residuals'][5]['available_at'] = datetime(
        2026, 1, 7, 9, tzinfo=TZ
    ).isoformat()
    delayed_interval, delayed_usage = apply_head_calibration(
        {field: 100. for field in OHLC}, 100., delayed, date(2026, 1, 7)
    )
    assert delayed_usage['latest_label_date'] == '2026-01-05'
    assert delayed_interval['close'] == pytest.approx([100., 103.7])
    with pytest.raises(ValueError, match='insufficient causal'):
        apply_head_calibration(
            {field: 100. for field in OHLC}, 100., loaded, date(2026, 1, 4)
        )

    changed_artifact = deepcopy(loaded)
    changed_artifact['residuals'][0]['normalized_signed_residual']['close'] += 0.01
    artifact_path.write_text(json.dumps(changed_artifact))
    with pytest.raises(ValueError, match='differs from source prediction'):
        load_head_calibration_artifact(
            artifact_path,
            instrument='TAIEX',
            source_identity=identity,
            source_result_sha256=source_hash,
            source_dir=tmp_path,
        )
    artifact_path.write_text(json.dumps(artifact))
    fold_path.write_text('{"tampered":true}')
    with pytest.raises(ValueError, match='source fold mismatch'):
        load_head_calibration_artifact(
            artifact_path,
            instrument='TAIEX',
            source_identity=identity,
            source_result_sha256=source_hash,
            source_dir=tmp_path,
        )
