from copy import deepcopy
from datetime import date, datetime, time, timedelta

import pytest

from tw_scenario.quant.contracts import OHLC, TZ
from tw_scenario.quant.head_calibration import calibrate_head_intervals


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
