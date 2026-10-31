"""Aggregate paired development errors; no sample deletion or model selection."""
import argparse
import hashlib
import json
from pathlib import Path
from statistics import mean


def summarize(rows):
    if not rows:
        return {'days': 0}
    result = {'days': len(rows)}
    for source in ('new', 'old', 'baseline'):
        errors = {key: [] for key in ('close', 'gap', 'body', 'range')}
        for row in rows:
            pred, actual = row[source], row['actual']
            errors['close'].append(pred['close'] - actual['close'])
            errors['gap'].append(pred['open'] - actual['open'])
            errors['body'].append((pred['close'] - pred['open']) -
                                  (actual['close'] - actual['open']))
            errors['range'].append((pred['high'] - pred['low']) -
                                   (actual['high'] - actual['low']))
        result[source] = {k: {'mae': mean(map(abs, v)), 'bias': mean(v)}
                          for k, v in errors.items()}
    return result


def diagnose(new_dir, old_dir):
    output = {'scope': 'posthoc_development_diagnostic_not_holdout_or_selection',
              'regime_definition': 'actual C/O body: down <-0.5%, up >0.5%, else neutral; diagnostic only, unavailable preopen',
              'folds': [], 'source_hashes': {}}
    for fold in (1, 2, 3):
        paths = [p / f'fold-{fold}.json' for p in (new_dir, old_dir)]
        new, old = [json.loads(p.read_text()) for p in paths]
        for p in paths:
            output['source_hashes'][str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()
        groups = [new['candidate_rows'], old['candidate_rows'], new['baseline_rows'], old['baseline_rows']]
        dates = [[r['date'] for r in group] for group in groups]
        if not all(d == dates[0] for d in dates) or len(set(dates[0])) != len(dates[0]):
            raise ValueError('unpaired or duplicate dates')
        rows, failures = [], []
        for n, o, b, ob in zip(*groups):
            if any(r['status'] != 'OK' for r in (n, o, b, ob)):
                failures.append({'date': n['date'], 'statuses': [r['status'] for r in (n, o, b, ob)]})
                continue
            if not (n['actual'] == o['actual'] == b['actual'] == ob['actual'] and
                    b['predicted'] == ob['predicted'] and
                    n['previous_close'] == o['previous_close'] == b['previous_close']):
                raise ValueError('label/reference/baseline mismatch')
            body = n['actual']['close'] / n['actual']['open'] - 1
            rows.append({'date': n['date'], 'actual': n['actual'], 'new': n['predicted'],
                         'old': o['predicted'], 'baseline': b['predicted'],
                         'regime': 'down' if body < -.005 else 'up' if body > .005 else 'neutral'})
        excess = sorted([abs(r['new']['close'] - r['actual']['close']) -
                         abs(r['baseline']['close'] - r['actual']['close']) for r in rows], reverse=True)
        output['folds'].append({'fold': fold, 'expected_days': len(dates[0]),
                               'failures': failures, 'all_days': summarize(rows),
                               'regimes': {g: summarize([r for r in rows if r['regime'] == g])
                                           for g in ('down', 'neutral', 'up')},
                               'total_excess_absolute_error': sum(excess),
                               'largest_five_excess_absolute_error_sum': sum(excess[:5])})
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('new_dir', type=Path)
    parser.add_argument('old_dir', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    result = diagnose(args.new_dir, args.old_dir)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
