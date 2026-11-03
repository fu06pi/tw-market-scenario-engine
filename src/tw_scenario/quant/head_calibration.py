"""Prequential marginal intervals for existing walk-forward head predictions."""
from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from copy import deepcopy
from datetime import date, datetime
from pathlib import Path

import numpy as np

from .contracts import OHLC, at, digest

CALIBRATION_ARTIFACT_VERSION = 'timesfm3-head-interval-calibration-artifact-v1'


def _validate_calibration_config(config):
    required = {
        'version': 'causal-head-marginal-residual-v1',
        'lower_quantile': 0.1,
        'upper_quantile': 0.9,
        'normalization': 'prediction_day_same_contract_previous_close',
        'update_policy': 'previous_walk_forward_residuals_available_by_0830_only',
        'point_policy': 'unchanged',
        'interval_policy': 'marginal_only_not_joint_OHLC_quantiles',
    }
    for key, expected in required.items():
        if config.get(key) != expected:
            raise ValueError(f'unsupported calibration {key}')
    window, minimum = config.get('window'), config.get('minimum')
    if (type(window) is not int or type(minimum) is not int
            or not 2 <= minimum <= window):
        raise ValueError('calibration requires 2 <= minimum <= window')


def build_head_calibration_artifact(
    rows,
    label_available_at,
    *,
    instrument,
    source_identity,
    source_result_sha256,
    source_fold_sha256,
    calibration,
):
    """Freeze only timestamped, normalized residuals needed by later forecasts."""
    _validate_calibration_config(calibration)
    if instrument not in ('TAIEX', 'TX'):
        raise ValueError('calibration artifact requires an independent market')
    if source_identity.get('instrument') != instrument:
        raise ValueError('calibration artifact instrument differs from source identity')
    if (source_identity.get('spec', {}).get('context')
            != calibration.get('candidate_context')):
        raise ValueError('calibration context differs from source identity')
    if len(source_result_sha256) != 64:
        raise ValueError('invalid source result hash')
    dates = [row['date'] for row in rows]
    if dates != sorted(set(dates)):
        raise ValueError('calibration dates must be unique and chronological')
    residuals = []
    for row in rows:
        if row['status'] != 'OK':
            continue
        if row.get('instrument') != instrument:
            raise ValueError('calibration row instrument mismatch')
        day = date.fromisoformat(row['date'])
        published = label_available_at[row['date']]
        if published.utcoffset() is None or published <= at(day):
            raise ValueError('daytime actual must become available after forecast cutoff')
        reference = float(row['previous_close'])
        point = np.array([row['predicted'][field] for field in OHLC], dtype=float)
        actual = np.array([row['actual'][field] for field in OHLC], dtype=float)
        if (not math.isfinite(reference) or reference <= 0
                or not np.isfinite(point).all() or not np.isfinite(actual).all()
                or (point <= 0).any() or (actual <= 0).any()):
            raise ValueError('invalid calibration prices/reference')
        values = (actual - point) / reference
        residuals.append({
            'date': row['date'],
            'available_at': published.isoformat(),
            'normalized_signed_residual': {
                field: float(values[index]) for index, field in enumerate(OHLC)
            },
        })
    return {
        'schema_version': CALIBRATION_ARTIFACT_VERSION,
        'instrument': instrument,
        'source_identity_hash': digest(source_identity),
        'source_result_sha256': source_result_sha256,
        'source_fold_sha256': dict(source_fold_sha256),
        'calibration': deepcopy(calibration),
        'residuals': residuals,
        'counts': {
            'source_rows': len(rows),
            'usable_residuals': len(residuals),
            'failed_rows': len(rows) - len(residuals),
        },
        'research_only': True,
        'production_switch': False,
    }


def _validate_head_calibration_artifact(
    artifact,
    *,
    instrument,
    source_identity,
    source_result_sha256,
    source_dir=None,
    source_fold_results=None,
):
    expected_keys = {
        'schema_version', 'instrument', 'source_identity_hash', 'source_result_sha256',
        'source_fold_sha256', 'calibration', 'residuals', 'counts', 'research_only',
        'production_switch',
    }
    if set(artifact) != expected_keys:
        raise ValueError('unexpected calibration artifact schema')
    if artifact['schema_version'] != CALIBRATION_ARTIFACT_VERSION:
        raise ValueError('unsupported calibration artifact version')
    if artifact['instrument'] != instrument:
        raise ValueError('calibration artifact instrument mismatch')
    if artifact['source_identity_hash'] != digest(source_identity):
        raise ValueError('calibration artifact source identity mismatch')
    if artifact['source_result_sha256'] != source_result_sha256:
        raise ValueError('calibration artifact source result mismatch')
    if artifact['research_only'] is not True or artifact['production_switch'] is not False:
        raise ValueError('calibration artifact cannot enable production')
    config = artifact['calibration']
    _validate_calibration_config(config)
    if config.get('candidate_context') != source_identity.get('spec', {}).get('context'):
        raise ValueError('calibration artifact context mismatch')
    fold_hashes = artifact['source_fold_sha256']
    if not isinstance(fold_hashes, dict):
        raise TypeError('invalid calibration fold hashes')
    expected_fold_names = {
        f'fold-{index}.json'
        for index, _ in enumerate(source_identity.get('folds', ()), 1)
    }
    if set(fold_hashes) != expected_fold_names:
        raise ValueError('calibration artifact fold set differs from source identity')
    if source_dir is not None:
        source_rows = []
        ordered_fold_names = sorted(
            fold_hashes,
            key=lambda name: int(name.removeprefix('fold-').removesuffix('.json')),
        )
        for index, name in enumerate(ordered_fold_names, 1):
            expected = fold_hashes[name]
            if not (name.startswith('fold-') and name.endswith('.json')):
                raise ValueError('invalid calibration fold filename')
            path = Path(source_dir) / name
            if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise ValueError('calibration artifact source fold mismatch')
            fold = json.loads(path.read_text(encoding='utf-8'))
            if fold.get('identity_hash') != digest(source_identity) or fold.get('fold') != index:
                raise ValueError('calibration source fold provenance mismatch')
            if source_fold_results is not None:
                if len(source_fold_results) != len(fold_hashes):
                    raise ValueError('calibration source fold count mismatch')
                if fold.get('summary') != source_fold_results[index - 1]:
                    raise ValueError('calibration source fold summary mismatch')
            if not isinstance(fold.get('candidate_rows'), list):
                raise TypeError('calibration source fold lacks candidate rows')
            validation_start, validation_end = source_identity['folds'][index - 1]['validation']
            if any(
                not validation_start <= row.get('date', '') <= validation_end
                for row in fold['candidate_rows']
            ):
                raise ValueError('calibration source prediction lies outside its fold')
            source_rows.extend(fold['candidate_rows'])
    records = artifact['residuals']
    if not isinstance(records, list):
        raise TypeError('invalid calibration residual records')
    dates = [record.get('date') for record in records]
    if dates != sorted(set(dates)):
        raise ValueError('calibration residual dates must be unique and chronological')
    for record in records:
        if set(record) != {'date', 'available_at', 'normalized_signed_residual'}:
            raise ValueError('unexpected calibration residual schema')
        day = date.fromisoformat(record['date'])
        available_at = datetime.fromisoformat(record['available_at'])
        if available_at.utcoffset() is None or available_at <= at(day):
            raise ValueError('invalid calibration residual availability')
        values = record['normalized_signed_residual']
        if set(values) != set(OHLC) or not all(math.isfinite(values[field]) for field in OHLC):
            raise ValueError('invalid calibration residual values')
    counts = artifact['counts']
    if (set(counts) != {'source_rows', 'usable_residuals', 'failed_rows'}
            or counts['usable_residuals'] != len(records)
            or counts['source_rows'] != counts['usable_residuals'] + counts['failed_rows']):
        raise ValueError('calibration artifact counts mismatch')
    if source_dir is not None:
        source_dates = [row.get('date') for row in source_rows]
        if source_dates != sorted(set(source_dates)):
            raise ValueError('calibration source prediction dates are not chronological')
        expected_records = []
        for row in source_rows:
            if row.get('status') != 'OK':
                continue
            if row.get('instrument') != instrument:
                raise ValueError('calibration source prediction instrument mismatch')
            reference = float(row['previous_close'])
            expected_records.append({
                'date': row['date'],
                'normalized_signed_residual': {
                    field: (float(row['actual'][field]) - float(row['predicted'][field]))
                    / reference
                    for field in OHLC
                },
            })
        if counts['source_rows'] != len(source_rows) or len(records) != len(expected_records):
            raise ValueError('calibration residual count differs from source predictions')
        for record, expected in zip(records, expected_records):
            if record['date'] != expected['date']:
                raise ValueError('calibration residual dates differ from source predictions')
            if any(
                not math.isclose(
                    record['normalized_signed_residual'][field],
                    expected['normalized_signed_residual'][field],
                    rel_tol=0,
                    abs_tol=1e-15,
                )
                for field in OHLC
            ):
                raise ValueError('calibration residual differs from source prediction')
    return artifact


def load_head_calibration_artifact(
    path,
    *,
    instrument,
    source_identity,
    source_result_sha256,
    source_dir=None,
    source_fold_results=None,
):
    """Load and bind a calibration artifact to its exact prediction source."""
    path = Path(path)
    artifact = json.loads(path.read_text(encoding='utf-8'))
    _validate_head_calibration_artifact(
        artifact,
        instrument=instrument,
        source_identity=source_identity,
        source_result_sha256=source_result_sha256,
        source_dir=source_dir,
        source_fold_results=source_fold_results,
    )
    return artifact, hashlib.sha256(path.read_bytes()).hexdigest()


def verify_head_calibration_availability(data, instrument, artifact, forecast_day):
    """Verify every residual that could affect today against certified publication times."""
    for record in artifact['residuals']:
        residual_day = date.fromisoformat(record['date'])
        claimed = datetime.fromisoformat(record['available_at'])
        if residual_day >= forecast_day:
            continue
        actual = data.actual(instrument, residual_day)
        if actual.available_at != claimed:
            raise ValueError('calibration residual availability differs from certified data')


def apply_head_calibration(point, previous_close, artifact, forecast_day):
    """Apply only residuals published by the forecast cutoff; fail on warmup."""
    config = artifact['calibration']
    _validate_calibration_config(config)
    cutoff = at(forecast_day)
    eligible = []
    for record in artifact['residuals']:
        residual_day = date.fromisoformat(record['date'])
        available_at = datetime.fromisoformat(record['available_at'])
        if residual_day < forecast_day and available_at <= cutoff:
            eligible.append(record)
    eligible = eligible[-config['window']:]
    if len(eligible) < config['minimum']:
        raise ValueError('insufficient causal calibration residuals for daily forecast')
    reference = float(previous_close)
    values = np.array([point[field] for field in OHLC], dtype=float)
    if (not math.isfinite(reference) or reference <= 0
            or not np.isfinite(values).all() or (values <= 0).any()):
        raise ValueError('invalid daily calibration point/reference')
    residuals = np.array([
        [record['normalized_signed_residual'][field] for field in OHLC]
        for record in eligible
    ], dtype=float)
    lower_q, upper_q = np.quantile(
        residuals,
        [config['lower_quantile'], config['upper_quantile']],
        axis=0,
        method='linear',
    )
    lower = np.minimum(values, values + reference * lower_q)
    upper = np.maximum(values, values + reference * upper_q)
    if (lower <= 0).any() or not np.isfinite(lower).all() or not np.isfinite(upper).all():
        raise ValueError('daily calibration produced invalid marginal bounds')
    return (
        {field: [float(lower[index]), float(upper[index])]
         for index, field in enumerate(OHLC)},
        {
            'samples': len(eligible),
            'latest_label_date': eligible[-1]['date'],
            'window': config['window'],
            'minimum': config['minimum'],
        },
    )


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
    from .backtest import load_dataset
    from .evaluation import metrics
    from .timesfm_head_training import development_acceptance

    experiment_dir = Path(experiment_dir)
    result_path = experiment_dir / 'result.json'
    source = json.loads(result_path.read_text())
    identity = source['identity']
    data, dataset_id = load_dataset(Path(dataset_dir))
    _validate_calibration_config(calibration)
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
    source_result_sha256 = hashlib.sha256(result_path.read_bytes()).hexdigest()
    artifact = build_head_calibration_artifact(
        candidate,
        times,
        instrument=market,
        source_identity=identity,
        source_result_sha256=source_result_sha256,
        source_fold_sha256=hashes,
        calibration=calibration,
    )
    report = {
        'schema_version': 'head-interval-recalibration-v1', 'instrument': market,
        'research_scope': 'development_only_not_independent_holdout',
        'production_switch': False, 'config': calibration, 'source_identity': identity,
        'source_result_sha256': source_result_sha256,
        'source_fold_sha256': hashes,
        'calibration_code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'counts': dict(Counter(r['calibration']['status'] for r in calibrated)),
        'all_days': len(calibrated), 'raw_gate': raw_gate, 'development_gate': gate,
        'calibrated_subset_diagnostic_only': metrics(
            [r for r in calibrated if r['calibration']['status'] == 'CALIBRATED'],
            config['neutral_bps']),
        'points_unchanged': all(a.get('predicted') == b.get('predicted')
                                for a, b in zip(candidate, calibrated)),
        'calibration_artifact_payload_hash': digest(artifact),
        'limitations': ['Warmup days retain original intervals and remain in full-sample gates.',
                        'Marginal intervals do not imply joint OHLC coverage.',
                        'Interval calibration cannot repair point-forecast instability.'],
    }
    return report, calibrated, artifact
