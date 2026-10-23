"""Prequential marginal intervals for existing walk-forward head predictions."""
from __future__ import annotations

from copy import deepcopy
from datetime import date

import numpy as np

from .contracts import OHLC, at


def calibrate_head_intervals(rows, label_available_at, *, window=126, minimum=40):
    """Use only older published residuals; preserve points, failures and warmup days.

    These are marginal price intervals, not coherent joint OHLC quantiles.
    Refit boundaries may carry older residuals from the same prediction pipeline.
    """
    if not 2 <= minimum <= window:
        raise ValueError("calibration requires 2 <= minimum <= window")
    dates = [row['date'] for row in rows]
    if dates != sorted(set(dates)):
        raise ValueError("calibration dates must be unique and chronological")
    pool, output = [], []
    for row in rows:
        result = deepcopy(row)
        day = date.fromisoformat(row['date'])
        cutoff = at(day)
        eligible = [r for r in pool if r['day'] < day and r['available_at'] <= cutoff][-window:]
        result['calibration'] = {
            'method': 'causal_signed_price_residual_marginals_v1',
            'samples': len(eligible), 'window': window, 'minimum': minimum,
            'latest_label_date': str(eligible[-1]['day']) if eligible else None,
            'status': 'INPUT_FAILED' if row['status'] != 'OK' else 'WARMUP_ORIGINAL_INTERVAL',
        }
        if row['status'] != 'OK':
            output.append(result)
            continue
        reference = float(row['previous_close'])
        point = np.array([row['predicted'][f] for f in OHLC], dtype=float)
        actual = np.array([row['actual'][f] for f in OHLC], dtype=float)
        published = label_available_at[row['date']]
        if published.utcoffset() is None or published <= cutoff:
            raise ValueError('daytime actual must become available after forecast cutoff')
        if (not np.isfinite(reference) or reference <= 0
                or not np.isfinite(point).all() or not np.isfinite(actual).all()
                or (point <= 0).any() or (actual <= 0).any()):
            raise ValueError('invalid calibration prices/reference')
        if len(eligible) >= minimum:
            residuals = np.stack([r['residual'] for r in eligible])
            lower, upper = np.quantile(residuals, [0.1, 0.9], axis=0, method='linear')
            lower = np.minimum(point, point + reference * lower)
            upper = np.maximum(point, point + reference * upper)
            if (lower <= 0).any() or not np.isfinite(upper).all():
                result.update(status='FAILED', error='calibration produced invalid marginal bounds')
                result['calibration']['status'] = 'INVALID_BOUNDS'
            else:
                result['interval'] = {f: [float(lower[i]), float(upper[i])]
                                      for i, f in enumerate(OHLC)}
                result['interval_method'] = 'causal signed residual marginal quantiles; not joint OHLC'
                result['calibration']['status'] = 'CALIBRATED'
        # Append only after this forecast. Its label cannot influence today's bounds.
        pool.append({'day': day, 'available_at': published, 'residual': (actual - point) / reference})
        output.append(result)
    return output


def evaluate_saved_head_intervals(dataset_dir, experiment_dir, config, calibration):
    """Verify saved fold provenance, then evaluate all dates without model inference."""
    import hashlib
    import json
    from collections import Counter
    from pathlib import Path

    from .backtest import load_dataset
    from .contracts import digest
    from .evaluation import metrics
    from .timesfm_head_training import development_acceptance

    experiment_dir = Path(experiment_dir)
    result_path = experiment_dir / 'result.json'
    source = json.loads(result_path.read_text())
    identity = source['identity']
    data, dataset_id = load_dataset(Path(dataset_dir))
    if (identity['dataset_id'] != dataset_id or identity['config_hash'] != digest(config)
            or identity['spec']['context'] != calibration['candidate_context']):
        raise ValueError('dataset/config/context differs from frozen experiment')
    for field in ('timesfm_code_commit', 'timesfm_weight_revision'):
        if identity[field] != config[field]:
            raise ValueError('pinned TimesFM revision mismatch')
    if (calibration['lower_quantile'], calibration['upper_quantile']) != (0.1, 0.9):
        raise ValueError('only frozen 80 percent marginal quantiles supported')
    market = identity['instrument']
    if market not in ('TAIEX', 'TX'):
        raise ValueError('independent market experiment required')
    candidate, baseline, hashes, improvements = [], [], {}, []
    for index, fold in enumerate(source['fold_results'], 1):
        path = experiment_dir / f'fold-{index}.json'
        saved = json.loads(path.read_text())
        if saved['identity_hash'] != digest(identity) or saved['fold'] != index:
            raise ValueError('fold identity mismatch')
        if saved['summary'] != fold:
            raise ValueError('fold summary mismatch')
        checkpoint = Path(fold['checkpoint']['path'])
        if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != fold['checkpoint']['sha256']:
            raise ValueError('checkpoint hash mismatch')
        expected = [str(d) for d in data.calendar.days
                    if fold['validation_split'][0] <= str(d) <= fold['validation_split'][1]]
        for key in ('candidate_rows', 'baseline_rows'):
            if [r['date'] for r in saved[key]] != expected:
                raise ValueError('fold dates missing, duplicated or unordered')
        candidate.extend(saved['candidate_rows'])
        baseline.extend(saved['baseline_rows'])
        improvements.append(fold['close_relative_improvement'])
        hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    times = {}
    for row in candidate + baseline:
        if row['status'] != 'OK':
            continue
        day = date.fromisoformat(row['date'])
        actual = data.actual(market, day)
        if (row['instrument'] != market or row['contract'] != data.contract(market, day)
                or row['previous_close'] != data.reference(market, day).close
                or row['actual'] != {f: getattr(actual, f) for f in OHLC}):
            raise ValueError('saved prediction label/reference differs from certified data')
        times[row['date']] = actual.available_at
    calibrated = calibrate_head_intervals(
        candidate, times, window=calibration['window'], minimum=calibration['minimum'])
    gate = development_acceptance(calibrated, baseline, config, fold_improvements=improvements)
    raw_gate = development_acceptance(candidate, baseline, config, fold_improvements=improvements)
    if raw_gate != source['development_gate']:
        raise ValueError('saved development gate not reproducible')
    report = {
        'schema_version': 'head-interval-recalibration-v1', 'instrument': market,
        'research_scope': 'development_only_not_independent_holdout',
        'production_switch': False, 'config': calibration, 'source_identity': identity,
        'source_result_sha256': hashlib.sha256(result_path.read_bytes()).hexdigest(),
        'source_fold_sha256': hashes,
        'calibration_code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'counts': dict(Counter(r['calibration']['status'] for r in calibrated)),
        'all_days': len(calibrated), 'raw_gate': raw_gate, 'development_gate': gate,
        'calibrated_subset_diagnostic_only': metrics(
            [r for r in calibrated if r['calibration']['status'] == 'CALIBRATED'],
            config['neutral_bps']),
        'points_unchanged': all(a.get('predicted') == b.get('predicted')
                                for a, b in zip(candidate, calibrated)),
        'limitations': ['Warmup days retain original intervals and remain in full-sample gates.',
                        'Marginal intervals do not imply joint OHLC coverage.',
                        'Interval calibration cannot repair point-forecast instability.'],
    }
    return report, calibrated
