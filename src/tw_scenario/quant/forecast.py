from __future__ import annotations

from datetime import date

import numpy as np

from .contracts import OHLC, Bar, Calendar, Prices, at, decode, encode


class Dataset:
    def __init__(self, bars: list[Bar], calendar: Calendar):
        self.bars, self.calendar = bars, calendar
        self.index = {}
        self._research_feature_cache = {}
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
        """Legacy five-feature contract used by the already-closed experiments."""
        contract = self.contract('TX', day)
        night = self.get('TX', contract, 'night', day, at(day))
        tx_prev = self.reference('TX', day)
        cash_prev = self.reference('TAIEX', day)
        return np.array([np.log(night.close/tx_prev.close), np.log(night.high/night.low),
                         np.log1p(night.volume) if night.volume is not None else np.nan,
                         np.log(tx_prev.close/cash_prev.close),
                         (at(day)-night.session_end_at).total_seconds()/3600])

    @staticmethod
    def research_feature_names():
        """Names for the expanded, pre-open-only output-head research features."""
        names = [
            "tx_night_close_return", "tx_night_range", "tx_night_body",
            "tx_night_upper_extension", "tx_night_lower_extension",
            "tx_night_log_volume", "prior_tx_taiex_basis", "night_staleness_hours",
            "prior_tx_log_open_interest", "prior_tx_open_interest_change",
            "roll_session", "sessions_to_expiry", "weekday_sin", "weekday_cos",
            "month_end_session",
        ]
        for window in (5, 20, 60):
            names.extend((
                f"self_return_{window}", f"self_volatility_{window}",
                f"self_range_{window}", f"self_log_volume_{window}",
            ))
        return tuple(names)

    def research_features(self, instrument, day):
        """Expanded causal features whose source bars were available by D 08:30.

        This deliberately lives beside, rather than replacing, ``features`` so the
        closed historical experiments retain their exact information contract.
        """
        cache_key = (instrument, day)
        if cache_key in self._research_feature_cache:
            return self._research_feature_cache[cache_key].copy()
        cutoff = at(day)
        contract = self.contract('TX', day)
        night = self.get('TX', contract, 'night', day, cutoff)
        tx_prev = self.reference('TX', day)
        cash_prev = self.reference('TAIEX', day)
        night_body = np.log(night.close / night.open)
        night_upper = np.log(night.high / max(night.open, night.close))
        night_lower = np.log(min(night.open, night.close) / night.low)

        calendar_index = self.calendar.days.index(day)
        previous_day = self.calendar.previous(day)
        prior_contract = self.contract('TX', previous_day)
        roll_session = float(prior_contract != contract)
        expiry = self.calendar.expiries.get(contract)
        if expiry is None or expiry not in self.calendar.days:
            sessions_to_expiry = np.nan
        else:
            sessions_to_expiry = float(self.calendar.days.index(expiry) - calendar_index)

        prior_prior_day = self.calendar.previous(previous_day)
        prior_tx = self.index.get(('TX', contract, 'day', prior_prior_day))
        if (
            tx_prev.open_interest is None
            or prior_tx is None
            or prior_tx.open_interest is None
            or prior_tx.available_at > cutoff
        ):
            oi_level = oi_change = np.nan
        else:
            oi_level = np.log1p(tx_prev.open_interest)
            oi_change = np.log1p(tx_prev.open_interest) - np.log1p(prior_tx.open_interest)

        history_days = self.calendar.days[max(0, calendar_index - 60):calendar_index]
        returns, ranges, volumes = [], [], []
        for history_day in history_days:
            actual = self.actual(instrument, history_day)
            reference = self.reference(instrument, history_day)
            if actual.available_at > cutoff:
                raise ValueError('unpublished feature history')
            returns.append(np.log(actual.close / reference.close))
            ranges.append(np.log(actual.high / actual.low))
            volumes.append(
                np.log1p(actual.volume) if actual.volume is not None else np.nan
            )
        if len(returns) < 60 or not np.all(np.isfinite(volumes)):
            raise ValueError('insufficient causal feature history')

        next_day = (
            self.calendar.days[calendar_index + 1]
            if calendar_index + 1 < len(self.calendar.days)
            else None
        )
        weekday_angle = 2 * np.pi * day.weekday() / 5
        values = [
            np.log(night.close / tx_prev.close), np.log(night.high / night.low),
            night_body, night_upper, night_lower,
            np.log1p(night.volume) if night.volume is not None else np.nan,
            np.log(tx_prev.close / cash_prev.close),
            (cutoff - night.session_end_at).total_seconds() / 3600,
            oi_level, oi_change, roll_session, sessions_to_expiry,
            np.sin(weekday_angle), np.cos(weekday_angle),
            float(next_day is None or next_day.month != day.month),
        ]
        for window in (5, 20, 60):
            window_returns = np.asarray(returns[-window:])
            values.extend((
                float(window_returns.sum()),
                float(window_returns.std(ddof=1)) if window > 1 else 0.0,
                float(np.median(ranges[-window:])),
                float(np.mean(volumes[-window:])),
            ))
        result = np.asarray(values, dtype=float)
        if result.shape != (len(self.research_feature_names()),):
            raise RuntimeError('expanded research feature schema mismatch')
        if not np.all(np.isfinite(result)):
            raise ValueError('missing expanded research feature')
        result.setflags(write=False)
        self._research_feature_cache[cache_key] = result
        return result.copy()

    def research_history(self, instrument, day, representation, window, minimum):
        """Return aligned targets/features using only information known before D."""
        dates = self.calendar.days[:self.calendar.days.index(day)]
        ys, xs, used = [], [], []
        for history_day in reversed(dates[-window:]):
            try:
                actual = self.actual(instrument, history_day)
                if actual.available_at > at(day):
                    raise ValueError('unpublished training label')
                y = self.target(instrument, history_day, representation)
                x = self.research_features(instrument, history_day)
            except ValueError:
                break
            ys.append(y)
            xs.append(x)
            used.append(history_day)
        if len(ys) < minimum:
            raise ValueError(
                f'insufficient contiguous expanded history: {len(ys)} < {minimum}'
            )
        return np.asarray(ys[::-1]), np.asarray(xs[::-1]), used[::-1]

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
