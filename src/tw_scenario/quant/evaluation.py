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
