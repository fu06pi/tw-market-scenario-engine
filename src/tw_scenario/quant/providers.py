from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import stat
import time
import zipfile
from datetime import date, datetime, timedelta
from pathlib import Path

import requests
from bs4 import BeautifulSoup

from .contracts import TZ, Bar, at, digest, write_once

FINMIND_URL = 'https://api.finmindtrade.com/api/v4/data'
FINMIND_DATASETS = {
    'TaiwanStockPrice': 'TAIEX',
    'TaiwanFuturesDaily': 'TX',
    'TaiwanStockTradingDate': None,
}


def load_key(env_file: Path = Path('.env')) -> str | None:
    key = os.environ.get('TEJ_API_KEY')
    if key:
        return key
    if env_file.exists():
        if env_file.is_symlink() or stat.S_IMODE(env_file.stat().st_mode) & 0o077:
            raise ValueError('.env must be a private regular file (chmod 600)')
        for line in env_file.read_text().splitlines():
            if line.startswith('TEJ_API_KEY='):
                key = line.split('=', 1)[1].strip().strip('\"\'')
                if key:
                    os.environ['TEJ_API_KEY'] = key
                    return key
    return None


class TejProvider:
    def __init__(self):
        key = load_key()
        if not key:
            raise ValueError('TEJ_API_KEY absent; configure private .env or environment')
        import tejapi
        tejapi.ApiConfig.api_key = key
        self.api = tejapi

    def inspect(self, tables: list[str]) -> dict:
        # SDK info includes the key and user identity. Never persist raw info/errors.
        try:
            info = self.api.ApiConfig.info()
            allowed = ('startDate', 'endDate', 'reqDayLimit', 'rowsDayLimit',
                       'rowsMonthLimit', 'todayReqCount', 'todayRows', 'monthRows')
            safe = {k: info.get(k) for k in allowed}
            user = info.get('user', {})
            safe['tables'] = user.get('tables', info.get('tables', []))
            safe['table_metadata'] = {t: self.api.table_info(t) for t in tables}
            safe['checked_at'] = datetime.now(TZ).isoformat()
            # Defense in depth against provider responses embedding the credential.
            return redact(safe)
        except Exception:  # noqa: BLE001 - SDK exceptions may contain the credential
            raise RuntimeError('TEJ inspection failed; verify access/quota locally') from None

    def download(self, table: str, date_column: str, start: date, end: date,
                 filters: dict, cache: Path) -> Path:
        if start > end or (end-start).days > 366 or not filters:
            raise ValueError('bounded <=366-day request and explicit instrument filters required')
        request = {'table': table, 'date_column': date_column, 'start': str(start),
                   'end': str(end), 'filters': filters}
        path = cache / (digest(request) + '.json')
        if path.exists():
            return path
        try:
            frame = self.api.get(table, **filters,
                                 **{date_column: {'gte': str(start), 'lte': str(end)}},
                                 paginate=True)
            records = json.loads(frame.to_json(orient='records', date_format='iso'))
            write_once(path, redact({'request': request, 'fetched_at': datetime.now(TZ),
                                    'records': records}))
        except Exception:  # noqa: BLE001 - redact all provider-specific failures
            raise RuntimeError('TEJ bounded download failed; no raw provider error logged') from None
        return path


def redact(value):
    secrets = tuple(
        secret for name in ('TEJ_API_KEY', 'FINMIND_TOKEN')
        if (secret := os.environ.get(name, ''))
    )
    if isinstance(value, dict):
        return {k: redact(v) for k, v in value.items()
                if str(k).lower() not in ('key', 'api_key', 'token', 'userid', 'shortname')}
    if isinstance(value, list):
        return [redact(v) for v in value]
    if isinstance(value, str):
        for secret in secrets:
            value = value.replace(secret, '[REDACTED]')
    return value


def load_finmind_token(env_file: Path = Path('.env')) -> str | None:
    """Load the optional token without ever exposing it in cached request metadata."""
    token = os.environ.get('FINMIND_TOKEN')
    if token:
        return token
    if not env_file.exists():
        return None
    if env_file.is_symlink() or stat.S_IMODE(env_file.stat().st_mode) & 0o077:
        raise ValueError('.env must be a private regular file (chmod 600)')
    for line in env_file.read_text().splitlines():
        if line.startswith('FINMIND_TOKEN='):
            token = line.split('=', 1)[1].strip().strip('\"\'')
            if token:
                os.environ['FINMIND_TOKEN'] = token
                return token
    return None


class FinMindProvider:
    """Bounded FinMind v4 downloads with write-once, credential-free raw caches."""

    def __init__(self, cache: Path, token: str | None = None, session=None):
        self.cache = cache
        self._token = token if token is not None else load_finmind_token()
        self.session = session or requests.Session()

    def download(self, dataset: str, data_id: str | None, start: date, end: date) -> Path:
        if dataset not in FINMIND_DATASETS or FINMIND_DATASETS[dataset] != data_id:
            raise ValueError('unsupported FinMind dataset/data_id pair')
        if start > end or (end - start).days > 366:
            raise ValueError('FinMind request must be bounded to at most one year')
        request = {
            'dataset': dataset,
            'start_date': str(start),
            'end_date': str(end),
        }
        if data_id is not None:
            request['data_id'] = data_id
        path = self.cache / f'{digest(request)}.json'
        if path.exists():
            self._read_cache(path, dataset, data_id)
            return path
        params = dict(request)
        if self._token:
            params['token'] = self._token
        try:
            response = self.session.get(FINMIND_URL, params=params, timeout=60)
            response.raise_for_status()
        except requests.RequestException:
            raise RuntimeError('FinMind HTTP request failed; provider details suppressed') from None
        raw = response.content
        try:
            payload = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError, TypeError):
            raise ValueError('FinMind returned invalid JSON') from None
        if (
            not isinstance(payload, dict)
            or payload.get('status') != 200
            or payload.get('msg') != 'success'
            or not isinstance(payload.get('data'), list)
            or not all(isinstance(row, dict) for row in payload['data'])
        ):
            raise ValueError('FinMind API status or response structure invalid')
        records = payload['data']
        cached = {
            'request': request,
            'api_status': payload['status'],
            'api_message': payload['msg'],
            'fetched_at': datetime.now(TZ).isoformat(),
            'sha256': hashlib.sha256(raw).hexdigest(),
            'records_hash': digest(records),
            'records': records,
        }
        write_once(path, cached)
        time.sleep(0.15)
        return path

    def taiex(self, start: date, end: date) -> list[Bar]:
        path = self.download('TaiwanStockPrice', 'TAIEX', start, end)
        return self.parse_taiex(path)

    def tx(self, start: date, end: date, calendar) -> list[Bar]:
        path = self.download('TaiwanFuturesDaily', 'TX', start, end)
        return self.parse_tx(path, calendar)

    def trading_dates(self, start: date, end: date) -> list[date]:
        path = self.download('TaiwanStockTradingDate', None, start, end)
        return self.parse_trading_dates(path)

    @staticmethod
    def _read_cache(path: Path, dataset: str, data_id: str | None) -> dict:
        try:
            cached = json.loads(path.read_text(encoding='utf-8'))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            raise ValueError('invalid FinMind cache file') from None
        request = cached.get('request') if isinstance(cached, dict) else None
        records = cached.get('records') if isinstance(cached, dict) else None
        if (
            not isinstance(request, dict)
            or request.get('dataset') != dataset
            or request.get('data_id') != data_id
            or cached.get('api_status') != 200
            or cached.get('api_message') != 'success'
            or not isinstance(cached.get('fetched_at'), str)
            or not isinstance(cached.get('sha256'), str)
            or len(cached['sha256']) != 64
            or not isinstance(records, list)
            or not all(isinstance(row, dict) for row in records)
            or cached.get('records_hash') != digest(records)
        ):
            raise ValueError('FinMind cache status, metadata, or structure invalid')
        try:
            start = date.fromisoformat(request['start_date'])
            end = date.fromisoformat(request['end_date'])
        except (KeyError, TypeError, ValueError):
            raise ValueError('FinMind cache request dates invalid') from None
        if start > end or (end - start).days > 366:
            raise ValueError('FinMind cache request is not bounded')
        return cached

    @classmethod
    def parse_trading_dates(cls, path: Path) -> list[date]:
        cached = cls._read_cache(path, 'TaiwanStockTradingDate', None)
        start = date.fromisoformat(cached['request']['start_date'])
        end = date.fromisoformat(cached['request']['end_date'])
        days = []
        for row in cached['records']:
            if set(row) != {'date'}:
                raise ValueError('FinMind TaiwanStockTradingDate schema changed')
            try:
                day = date.fromisoformat(row['date'])
            except (TypeError, ValueError):
                raise ValueError('FinMind trading date invalid') from None
            if not start <= day <= end:
                raise ValueError('FinMind trading date outside requested date range')
            days.append(day)
        if days != sorted(set(days)):
            raise ValueError('FinMind trading dates must be sorted and unique')
        return days

    @classmethod
    def parse_taiex(cls, path: Path) -> list[Bar]:
        cached = cls._read_cache(path, 'TaiwanStockPrice', 'TAIEX')
        required = {'date', 'stock_id', 'open', 'max', 'min', 'close', 'Trading_Volume'}
        start = date.fromisoformat(cached['request']['start_date'])
        end = date.fromisoformat(cached['request']['end_date'])
        bars = []
        for row in cached['records']:
            if not required.issubset(row) or row['stock_id'] != 'TAIEX':
                raise ValueError('FinMind TaiwanStockPrice schema or instrument changed')
            try:
                day = date.fromisoformat(row['date'])
            except (TypeError, ValueError):
                raise ValueError('FinMind TAIEX date invalid') from None
            if not start <= day <= end:
                raise ValueError('FinMind TAIEX row outside requested date range')
            bars.append(Bar(
                instrument='TAIEX', trading_date=day, session='day',
                open=number(row['open']), high=number(row['max']),
                low=number(row['min']), close=number(row['close']),
                volume=optional_number(row['Trading_Volume']),
                session_start_at=at(day, '09:00'), session_end_at=at(day, '13:30'),
                available_at=at(day, '18:00'), fetched_at=cached['fetched_at'],
                availability_basis=(
                    'ASSUMPTION: same-day 18:00; historical FinMind daily API is '
                    'retrospective, not point-in-time, and has no publication archive'
                ),
                source=FINMIND_URL, source_hash=cached['sha256'],
            ))
        return bars

    @classmethod
    def parse_tx(cls, path: Path, calendar) -> list[Bar]:
        cached = cls._read_cache(path, 'TaiwanFuturesDaily', 'TX')
        required = {
            'date', 'futures_id', 'contract_date', 'open', 'max', 'min', 'close',
            'volume', 'settlement_price', 'open_interest', 'trading_session',
        }
        start = date.fromisoformat(cached['request']['start_date'])
        end = date.fromisoformat(cached['request']['end_date'])
        bars = []
        for row in cached['records']:
            if not required.issubset(row) or row['futures_id'] != 'TX':
                raise ValueError('FinMind TaiwanFuturesDaily schema or instrument changed')
            contract = str(row['contract_date']).strip()
            if len(contract) != 6 or not contract.isdigit():
                # Calendar spreads are present in this endpoint but are not outright contracts.
                continue
            try:
                day = date.fromisoformat(row['date'])
            except (TypeError, ValueError):
                raise ValueError('FinMind TX date invalid') from None
            if not start <= day <= end:
                raise ValueError('FinMind TX row outside requested date range')
            if day not in calendar.days:
                raise ValueError('FinMind TX row lacks verified exchange-calendar date')
            prices = tuple(number(row[key]) for key in ('open', 'max', 'min', 'close'))
            if any(value <= 0 for value in prices):
                # FinMind uses zero OHLC sentinels on incomplete/no-trade contract rows.
                continue
            session_name = row['trading_session']
            if session_name == 'position':
                session = 'day'
                end_clock = '13:30' if calendar.expiries.get(contract) == day else '13:45'
                start_at, end_at, available = at(day, '08:45'), at(day, end_clock), at(day, '18:00')
            elif session_name == 'after_market':
                session = 'night'
                index = calendar.days.index(day)
                if index == 0:
                    continue
                start_day = calendar.days[index - 1]
                end_day = start_day + timedelta(days=1)
                start_at = at(start_day, '15:00')
                end_at, available = at(end_day, '05:00'), at(end_day, '07:30')
            else:
                raise ValueError(f'unknown FinMind TX trading_session: {session_name}')
            o, h, l, c = prices
            bars.append(Bar(
                instrument='TX', contract=contract, trading_date=day, session=session,
                open=o, high=h, low=l, close=c,
                volume=optional_number(row['volume']),
                open_interest=optional_number(row['open_interest']),
                settlement=optional_positive_number(row['settlement_price']),
                session_start_at=start_at, session_end_at=end_at, available_at=available,
                fetched_at=cached['fetched_at'], source=FINMIND_URL,
                source_hash=cached['sha256'],
                availability_basis=(
                    'ASSUMPTION: day 18:00/night 07:30 after physical session end; '
                    'historical FinMind daily API is retrospective, not point-in-time, '
                    'and has no publication archive'
                ),
            ))
        return bars


class OfficialProvider:
    """Raw immutable request cache. Revisions require a separate cache root."""
    def __init__(self, cache: Path):
        self.cache = cache
        self.session = requests.Session()

    def fetch(self, url: str, params: dict):
        request = {'url': url, 'params': params}
        path = self.cache / (digest(request) + '.json')
        if path.exists():
            return json.loads(path.read_text())
        response = self.session.get(url, params=params, timeout=30)
        response.raise_for_status()
        response.encoding = 'utf-8'
        body = response.text
        result = {'request': request, 'fetched_at': datetime.now(TZ).isoformat(),
                  'sha256': hashlib.sha256(body.encode()).hexdigest(), 'body': body}
        write_once(path, result)
        time.sleep(0.15)
        return result

    def fetch_post_bytes(self, url: str, form: dict, suffix: str):
        request = {'url': url, 'form': form}
        key = digest(request)
        path = self.cache / f'{key}.{suffix}'
        manifest = self.cache / f'{key}.manifest.json'
        if path.exists() and manifest.exists():
            return path, json.loads(manifest.read_text(encoding='utf-8'))
        response = self.session.post(url, data=form, timeout=120)
        response.raise_for_status()
        raw = response.content
        if not raw:
            raise ValueError('empty official download')
        self.cache.mkdir(parents=True, exist_ok=True)
        with path.open('xb') as f:
            f.write(raw)
            f.flush()
            os.fsync(f.fileno())
        metadata = {
            'request': request,
            'fetched_at': datetime.now(TZ).isoformat(),
            'sha256': hashlib.sha256(raw).hexdigest(),
            'bytes': len(raw),
        }
        write_once(manifest, metadata)
        return path, metadata

    def twse_month(self, year: int, month: int) -> list[Bar]:
        url = 'https://www.twse.com.tw/indicesReport/MI_5MINS_HIST'
        raw = self.fetch(url, {'date': f'{year:04d}{month:02d}01', 'response': 'json'})
        payload = json.loads(raw['body'])
        if payload.get('stat') != 'OK':
            raise ValueError('TWSE month unavailable')
        if payload['fields'] != ['日期', '開盤指數', '最高指數', '最低指數', '收盤指數']:
            raise ValueError('TWSE schema changed')
        bars = []
        for row in payload['data']:
            y, m, d = map(int, row[0].split('/'))
            day = date(y+1911, m, d)
            o, h, l, c = map(number, row[1:])
            bars.append(Bar(instrument='TAIEX', trading_date=day, session='day',
                            open=o, high=h, low=l, close=c,
                            session_start_at=at(day, '09:00'), session_end_at=at(day, '13:30'),
                            available_at=at(day, '18:00'), fetched_at=raw['fetched_at'],
                            availability_basis='ASSUMPTION: same-day 18:00; publication archive absent',
                            source=url, source_hash=raw['sha256']))
        return bars

    def tx_day(self, day: date, session: str) -> list[Bar]:
        url = 'https://www.taifex.com.tw/cht/3/futDailyMarketReport'
        raw = self.fetch(url, {'queryType': '2', 'marketCode': '0' if session == 'day' else '1',
                               'commodity_id': 'TX', 'queryDate': day.strftime('%Y/%m/%d')})
        soup = BeautifulSoup(raw['body'], 'html.parser')
        section = soup.find(id='printhere')
        if section is None:
            raise ValueError('TAIFEX report missing')
        text = section.get_text(' ', strip=True)
        if day.strftime('%Y/%m/%d') not in text:
            raise ValueError('TAIFEX returned wrong date')
        expected = '一般交易時段行情表' if session == 'day' else '盤後交易時段行情表'
        if expected not in text:
            raise ValueError('TAIFEX requested session unavailable')
        table = section.find('table')
        if table is None:
            raise ValueError('TAIFEX table missing')
        headers = [t.get_text('', strip=True) for t in table.find_all('th')]
        if headers[:6] != ['契約', '到期月份(週別)', '開盤價', '最高價', '最低價', '最後成交價']:
            raise ValueError('TAIFEX schema changed')
        bars = []
        for tr in table.select('tbody tr'):
            cols = [t.get_text('', strip=True) for t in tr.find_all('td')]
            if len(cols) < 9 or cols[0] != 'TX' or len(cols[1]) != 6 or not cols[1].isdigit():
                continue
            if any(v.strip() in ('-', '--', '') for v in cols[2:6]):
                # Preserve raw no-trade record; DQ reports missing selected contracts.
                continue
            o, h, l, c = map(number, cols[2:6])
            values = dict(zip(headers, cols))
            if session == 'day':
                start, end, available = at(day, '08:45'), at(day, '13:45'), at(day, '18:00')
                volume = number(values['*一般交易時段成交量'])
            else:
                # Parse physical session dates, especially Friday night -> Monday trading date.
                import re
                span = re.search(r'(\d{4}/\d{2}/\d{2})\s*15:00\s*[~～]\s*次日05:00', text)
                if not span:
                    raise ValueError('night physical session dates unavailable')
                start_day = datetime.strptime(span.group(1), '%Y/%m/%d').date()
                end_day = start_day + timedelta(days=1)
                start, end, available = at(start_day, '15:00'), at(end_day, '05:00'), at(end_day, '07:30')
                volume = number(values.get('成交量', values.get('*成交量', '')))
            bars.append(Bar(instrument='TX', contract=cols[1], trading_date=day, session=session,
                            open=o, high=h, low=l, close=c, volume=volume,
                            open_interest=optional_number(values.get('*未沖銷契約量')),
                            settlement=optional_positive_number(values.get('結算價')),
                            session_start_at=start, session_end_at=end, available_at=available,
                            fetched_at=raw['fetched_at'], source=url, source_hash=raw['sha256'],
                            availability_basis='ASSUMPTION: day 18:00/night 07:30; publication archive absent'))
        if not bars:
            raise ValueError('no traded TX monthly contracts')
        return bars

    def tx_year(self, year: int, calendar) -> list[Bar]:
        url = 'https://www.taifex.com.tw/cht/3/futDataDown'
        path, meta = self.fetch_post_bytes(
            url, {'down_type': '2', 'his_year': str(year)}, 'zip'
        )
        if not zipfile.is_zipfile(path):
            raise ValueError(f'TAIFEX annual download is not a ZIP: {year}')
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            if len(names) != 1 or not names[0].lower().endswith('.csv'):
                raise ValueError('unexpected TAIFEX annual archive shape')
            with archive.open(names[0]) as raw:
                text = io.TextIOWrapper(raw, encoding='cp950', newline='')
                return self._parse_tx_csv(csv.DictReader(text), calendar, url, meta)

    def tx_range(self, start: date, end: date, calendar) -> list[Bar]:
        if start > end or (end - start).days > 31:
            raise ValueError('TAIFEX range must be bounded to one month')
        url = 'https://www.taifex.com.tw/cht/3/futDataDown'
        form = {
            'down_type': '1',
            'queryStartDate': start.strftime('%Y/%m/%d'),
            'queryEndDate': end.strftime('%Y/%m/%d'),
            'commodity_id': 'TX',
        }
        path, meta = self.fetch_post_bytes(url, form, 'csv')
        with path.open('r', encoding='cp950', newline='') as text:
            return self._parse_tx_csv(csv.DictReader(text), calendar, url, meta)

    @staticmethod
    def _parse_tx_csv(rows, calendar, source: str, meta: dict) -> list[Bar]:
        expected = {
            '交易日期', '契約', '到期月份(週別)', '開盤價', '最高價', '最低價',
            '收盤價', '成交量', '結算價', '未沖銷契約數', '交易時段',
        }
        if rows.fieldnames is None or not expected.issubset(rows.fieldnames):
            raise ValueError('TAIFEX CSV schema changed')
        days = calendar.days
        bars = []
        for row in rows:
            if (row.get('契約') or '').strip() != 'TX':
                continue
            contract = (row.get('到期月份(週別)') or '').strip()
            if len(contract) != 6 or not contract.isdigit():
                continue
            values = [(row.get(k) or '').strip() for k in ('開盤價', '最高價', '最低價', '收盤價')]
            if any(v in ('', '-', '--') for v in values):
                continue
            day = datetime.strptime((row.get('交易日期') or '').strip(), '%Y/%m/%d').date()
            if day not in days:
                continue
            session_name = (row.get('交易時段') or '').strip()
            if session_name == '一般':
                session = 'day'
                end_clock = '13:30' if calendar.expiries.get(contract) == day else '13:45'
                start_at, end_at, available = at(day, '08:45'), at(day, end_clock), at(day, '18:00')
            elif session_name == '盤後':
                session = 'night'
                index = days.index(day)
                if index == 0:
                    continue
                start_day = days[index - 1]
                end_day = start_day + timedelta(days=1)
                start_at = at(start_day, '15:00')
                end_at, available = at(end_day, '05:00'), at(end_day, '07:30')
            else:
                raise ValueError(f'unknown TAIFEX session: {session_name}')
            o, h, l, c = map(number, values)
            bars.append(Bar(
                instrument='TX', contract=contract, trading_date=day, session=session,
                open=o, high=h, low=l, close=c,
                volume=optional_number(row['成交量']),
                open_interest=optional_number(row['未沖銷契約數']),
                settlement=optional_positive_number(row['結算價']),
                session_start_at=start_at, session_end_at=end_at, available_at=available,
                fetched_at=meta['fetched_at'], source=source, source_hash=meta['sha256'],
                availability_basis=(
                    'ASSUMPTION: day 18:00/night 07:30; session attribution and '
                    'contract are official, historical publication timestamps unavailable'
                ),
            ))
        return bars


def number(text):
    return float(str(text).replace(',', '').strip())


def optional_number(text):
    value = '' if text is None else str(text).strip()
    return None if value in ('', '-', '--') else number(value)


def optional_positive_number(text):
    value = optional_number(text)
    return value if value is not None and value > 0 else None


def load_bars(path: Path) -> list[Bar]:
    return [Bar.model_validate(json.loads(line)) for line in path.read_text().splitlines() if line.strip()]
