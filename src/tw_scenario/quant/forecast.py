from __future__ import annotations

from datetime import date

import numpy as np

from .contracts import OHLC, Bar, Calendar, Prices, at, decode, encode


class Dataset:
    def __init__(self, bars: list[Bar], calendar: Calendar):
        self.bars, self.calendar = bars, calendar
        self.index = {}
        for b in bars:
            key = (b.instrument, b.contract, b.session, b.trading_date)
            if key in self.index:
                raise ValueError('duplicate bar key')
            self.index[key] = b

    def get(self, instrument, contract, session, day, cutoff=None):
        b = self.index.get((instrument, contract, session, day))
        if b is None:
            raise ValueError(f'missing {instrument}/{contract}/{session}/{day}')
        if cutoff and (b.available_at > cutoff or b.session_end_at > cutoff):
            raise ValueError('data not available before cutoff')
        return b

    def contract(self, instrument, day):
        return '' if instrument == 'TAIEX' else self.calendar.contract_for(day)

    def reference(self, instrument, day):
        contract = self.contract(instrument, day)
        prev = self.calendar.previous(day)
        return self.get(instrument, contract, 'day', prev, at(day))

    def actual(self, instrument, day):
        return self.get(instrument, self.contract(instrument, day), 'day', day)

    def target(self, instrument, day, representation):
        return encode(self.actual(instrument, day), self.reference(instrument, day).close,
                      representation)

    def features(self, day):
        contract = self.contract('TX', day)
        night = self.get('TX', contract, 'night', day, at(day))
        tx_prev = self.reference('TX', day)
        cash_prev = self.reference('TAIEX', day)
        return np.array([np.log(night.close/tx_prev.close), np.log(night.high/night.low),
                         np.log1p(night.volume) if night.volume is not None else np.nan,
                         np.log(tx_prev.close/cash_prev.close),
                         (at(day)-night.session_end_at).total_seconds()/3600])

    def history(self, instrument, day, representation, window, minimum, with_features):
        dates = self.calendar.days[:self.calendar.days.index(day)]
        ys, xs, used = [], [], []
        # Missing history is never collapsed silently into an equally spaced series.
        for d in reversed(dates[-window:]):
            try:
                b = self.actual(instrument, d)
                if b.available_at > at(day):
                    raise ValueError('unpublished training label')
                y = self.target(instrument, d, representation)
                x = self.features(d) if with_features else np.empty(0)
                if not np.all(np.isfinite(x)):
                    raise ValueError('missing feature')
            except ValueError:
                break
            ys.append(y); xs.append(x); used.append(d)
        if len(ys) < minimum:
            raise ValueError(f'insufficient contiguous as-of history: {len(ys)} < {minimum}')
        return np.array(ys[::-1]), np.array(xs[::-1]), used[::-1]


def predict(data: Dataset, instrument: str, day: date, model: str,
            representation: str, config: dict):
    with_x = model in ('night_regression', 'arimax')
    y, x, dates = data.history(instrument, day, representation, config['window'],
                               config['min_history'], with_x)
    ref = data.reference(instrument, day).close
    if model == 'zero':
        target = encode(Prices(open=ref, high=ref, low=ref, close=ref), ref, representation)
    elif model == 'range':
        # Zero open/close return; trailing median excursions from previous close.
        excursions = np.array([data.actual(instrument,d).array()/data.reference(instrument,d).close
                               for d in dates])
        hi = max(1., float(np.median(excursions[:,1])))
        lo = min(1., float(np.median(excursions[:,2])))
        target = encode(Prices(open=ref, high=ref*hi, low=ref*lo, close=ref), ref, representation)
    elif model == 'night_regression':
        xf = data.features(day)
        if not np.all(np.isfinite(xf)):
            raise ValueError('missing forecast-period features')
        mean, scale = x.mean(0), x.std(0)
        scale[scale < 1e-8] = 1.
        design = np.column_stack([np.ones(len(x)), (x-mean)/scale])
        beta = np.linalg.lstsq(design, y, rcond=None)[0]
        target = np.r_[1., (xf-mean)/scale] @ beta
    elif model in ('arima', 'arimax'):
        import warnings

        from statsmodels.tsa.statespace.sarimax import SARIMAX
        exog = future = None
        if with_x:
            xf = data.features(day)
            if not np.all(np.isfinite(xf)):
                raise ValueError('missing forecast-period features')
            mean, scale = x.mean(0), x.std(0)
            scale[scale < 1e-8] = 1.
            exog, future = (x-mean)/scale, ((xf-mean)/scale)[None,:]
        means, scales = y.mean(0), y.std(0)
        scales[scales < 1e-8] = 1.
        target = []
        for j in range(4):
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                fit = SARIMAX((y[:,j]-means[j])/scales[j], exog=exog,
                              order=tuple(config['arima_order']), trend='c').fit(disp=False, maxiter=60)
            if not fit.mle_retvals.get('converged', False):
                raise ValueError('ARIMA nonconvergence')
            target.append(float(fit.forecast(1, exog=future)[0])*scales[j]+means[j])
        target = np.array(target)
    else:
        raise ValueError('unsupported model; TimesFM requires verified installed adapter')
    prices = decode(target, ref, representation)
    # Residuals calibrated on the actually published coherent prediction.
    target = encode(prices, ref, representation)
    return prices, target, ref, dates


def calibrated_interval(target, previous, representation, residuals, minimum=40):
    if len(residuals) < minimum:
        return None
    # Resample whole four-dimensional residual vectors, preserving dependence;
    # reconstruct each coherent OHLC path before marginal quantiles.
    draws = np.array([decode(target+r, previous, representation).array() for r in residuals])
    lo, hi = np.quantile(draws, [0.1, 0.9], axis=0)
    return {k: [float(lo[j]), float(hi[j])] for j, k in enumerate(OHLC)}
