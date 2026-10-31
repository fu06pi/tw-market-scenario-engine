from __future__ import annotations

import csv
import hashlib
import io
import json
from datetime import date, timedelta

import pytest
import requests

from tw_scenario.quant.contracts import TZ, Calendar, at, digest
from tw_scenario.quant.providers import FINMIND_URL, FinMindProvider, OfficialProvider, redact


class FakeResponse:
    def __init__(self, payload: dict, status_code: int = 200):
        self.content = json.dumps(payload).encode()
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f'HTTP {self.status_code}')


class FakeSession:
    def __init__(self, response: FakeResponse):
        self.response = response
        self.calls = []

    def get(self, url, params, timeout):
        self.calls.append({'url': url, 'params': params, 'timeout': timeout})
        return self.response


def calendar() -> Calendar:
    friday = date(2025, 11, 28)
    monday = date(2025, 12, 1)
    return Calendar(
        days=[friday, monday],
        expiries={'202512': date(2025, 12, 17)},
        source='synthetic-test exchange calendar',
        available_at=at(friday),
    )


def cache_file(tmp_path, dataset: str, data_id: str | None, records: list[dict]):
    request = {
        'dataset': dataset,
        'start_date': '2025-12-01',
        'end_date': '2025-12-01',
    }
    if data_id is not None:
        request['data_id'] = data_id
    raw = json.dumps({'status': 200, 'msg': 'success', 'data': records}).encode()
    cached = {
        'request': request,
        'api_status': 200,
        'api_message': 'success',
        'fetched_at': '2025-12-02T12:00:00+08:00',
        'sha256': hashlib.sha256(raw).hexdigest(),
        'records_hash': digest(records),
        'records': records,
    }
    path = tmp_path / 'finmind.json'
    path.write_text(json.dumps(cached), encoding='utf-8')
    return path


def test_finmind_download_is_bounded_cached_and_never_persists_token(tmp_path):
    response = FakeResponse({
        'status': 200,
        'msg': 'success',
        'data': [{
            'date': '2025-12-01',
            'stock_id': 'TAIEX',
            'Trading_Volume': 123,
            'open': 100,
            'max': 103,
            'min': 99,
            'close': 102,
        }],
    })
    session = FakeSession(response)
    provider = FinMindProvider(tmp_path, token='synthetic-secret-token', session=session)

    path = provider.download(
        'TaiwanStockPrice', 'TAIEX', date(2025, 12, 1), date(2025, 12, 1)
    )
    assert session.calls == [{
        'url': FINMIND_URL,
        'params': {
            'dataset': 'TaiwanStockPrice',
            'data_id': 'TAIEX',
            'start_date': '2025-12-01',
            'end_date': '2025-12-01',
            'token': 'synthetic-secret-token',
        },
        'timeout': 60,
    }]
    assert 'synthetic-secret-token' not in path.read_text(encoding='utf-8')

    assert provider.download(
        'TaiwanStockPrice', 'TAIEX', date(2025, 12, 1), date(2025, 12, 1)
    ) == path
    assert len(session.calls) == 1

    with pytest.raises(ValueError, match='bounded'):
        provider.download(
            'TaiwanStockPrice', 'TAIEX', date(2024, 1, 1), date(2025, 1, 2)
        )


def test_redact_removes_both_provider_credentials(monkeypatch):
    monkeypatch.setenv('TEJ_API_KEY', 'tej-secret')
    monkeypatch.setenv('FINMIND_TOKEN', 'finmind-secret')
    assert redact({
        'token': 'finmind-secret',
        'detail': 'tej-secret and finmind-secret',
    }) == {'detail': '[REDACTED] and [REDACTED]'}


def test_finmind_http_and_api_failures_are_fail_closed(tmp_path):
    http_provider = FinMindProvider(
        tmp_path / 'http', session=FakeSession(FakeResponse({}, status_code=503))
    )
    with pytest.raises(RuntimeError, match='HTTP request failed'):
        http_provider.download(
            'TaiwanStockPrice', 'TAIEX', date(2025, 12, 1), date(2025, 12, 1)
        )

    api_provider = FinMindProvider(
        tmp_path / 'api', session=FakeSession(FakeResponse({
            'status': 402,
            'msg': 'rate limited',
            'data': [],
        }))
    )
    with pytest.raises(ValueError, match='status or response structure'):
        api_provider.download(
            'TaiwanStockPrice', 'TAIEX', date(2025, 12, 1), date(2025, 12, 1)
        )


def test_finmind_taiex_parser_preserves_provenance_and_non_pit_assumption(tmp_path):
    path = cache_file(tmp_path, 'TaiwanStockPrice', 'TAIEX', [{
        'date': '2025-12-01',
        'stock_id': 'TAIEX',
        'Trading_Volume': 7_418_235_597,
        'open': 27_674.16,
        'max': 27_697.06,
        'min': 27_342.53,
        'close': 27_342.53,
    }])

    bars = FinMindProvider.parse_taiex(path)
    assert len(bars) == 1
    bar = bars[0]
    assert bar.instrument == 'TAIEX'
    assert bar.volume == 7_418_235_597
    assert bar.source == FINMIND_URL
    assert bar.fetched_at.utcoffset() == TZ.utcoffset(bar.fetched_at)
    assert 'retrospective, not point-in-time' in bar.availability_basis


def test_finmind_trading_dates_support_future_calendar_tail_without_data_id(tmp_path):
    response = FakeResponse({
        'status': 200,
        'msg': 'success',
        'data': [{'date': '2025-12-01'}],
    })
    session = FakeSession(response)
    provider = FinMindProvider(tmp_path, token='synthetic-secret-token', session=session)

    days = provider.trading_dates(date(2025, 12, 1), date(2025, 12, 1))
    assert days == [date(2025, 12, 1)]
    assert session.calls[0]['params'] == {
        'dataset': 'TaiwanStockTradingDate',
        'start_date': '2025-12-01',
        'end_date': '2025-12-01',
        'token': 'synthetic-secret-token',
    }
    assert 'synthetic-secret-token' not in next(tmp_path.glob('*.json')).read_text()


def test_finmind_tx_maps_sessions_and_weekend_without_zero_settlement(tmp_path):
    rows = [
        {
            'date': '2025-12-01',
            'futures_id': 'TX',
            'contract_date': '202512',
            'open': 27_650,
            'max': 27_674,
            'min': 27_401,
            'close': 27_404,
            'volume': 64_629,
            'settlement_price': 27_406,
            'open_interest': 71_269,
            'trading_session': 'position',
        },
        {
            'date': '2025-12-01',
            'futures_id': 'TX',
            'contract_date': '202512',
            'open': 27_743,
            'max': 27_864,
            'min': 27_693,
            'close': 27_784,
            'volume': 28_406,
            'settlement_price': 0,
            'open_interest': 0,
            'trading_session': 'after_market',
        },
        {
            'date': '2025-12-01',
            'futures_id': 'TX',
            'contract_date': '202512/202601',
            'open': 20,
            'max': 25,
            'min': 20,
            'close': 24,
            'volume': 253,
            'settlement_price': 0,
            'open_interest': 0,
            'trading_session': 'position',
        },
    ]
    path = cache_file(tmp_path, 'TaiwanFuturesDaily', 'TX', rows)

    bars = FinMindProvider.parse_tx(path, calendar())
    assert len(bars) == 2
    day = next(bar for bar in bars if bar.session == 'day')
    night = next(bar for bar in bars if bar.session == 'night')
    assert day.settlement == 27_406
    assert night.settlement is None
    assert night.open_interest == 0
    assert night.trading_date == date(2025, 12, 1)
    assert night.session_start_at == at(date(2025, 11, 28), '15:00')
    assert night.session_end_at == at(date(2025, 11, 29), '05:00')
    assert night.session_end_at - night.session_start_at == timedelta(hours=14)


def test_finmind_cache_tampering_and_row_schema_fail_closed(tmp_path):
    path = cache_file(tmp_path, 'TaiwanStockPrice', 'TAIEX', [{
        'date': '2025-12-01',
        'stock_id': 'TAIEX',
    }])
    with pytest.raises(ValueError, match='schema or instrument changed'):
        FinMindProvider.parse_taiex(path)

    cached = json.loads(path.read_text(encoding='utf-8'))
    cached['records'][0]['stock_id'] = 'OTHER'
    path.write_text(json.dumps(cached), encoding='utf-8')
    with pytest.raises(ValueError, match='cache status, metadata, or structure'):
        FinMindProvider.parse_taiex(path)


def test_official_tx_zero_settlement_and_weekend_session_are_valid():
    header = [
        '交易日期', '契約', '到期月份(週別)', '開盤價', '最高價', '最低價', '收盤價',
        '成交量', '結算價', '未沖銷契約數', '交易時段',
    ]
    raw = io.StringIO()
    writer = csv.DictWriter(raw, fieldnames=header)
    writer.writeheader()
    writer.writerow({
        '交易日期': '2025/12/01',
        '契約': 'TX',
        '到期月份(週別)': '202512',
        '開盤價': '27743',
        '最高價': '27864',
        '最低價': '27693',
        '收盤價': '27784',
        '成交量': '28406',
        '結算價': '0',
        '未沖銷契約數': '0',
        '交易時段': '盤後',
    })
    raw.seek(0)

    bars = OfficialProvider._parse_tx_csv(
        csv.DictReader(raw),
        calendar(),
        'synthetic-official-test',
        {'fetched_at': '2025-12-02T12:00:00+08:00', 'sha256': '0' * 64},
    )
    assert len(bars) == 1
    assert bars[0].settlement is None
    assert bars[0].open_interest == 0
    assert bars[0].session_start_at == at(date(2025, 11, 28), '15:00')
    assert bars[0].session_end_at == at(date(2025, 11, 29), '05:00')
