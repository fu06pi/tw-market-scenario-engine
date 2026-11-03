from __future__ import annotations

import hashlib
import json
import math
import os
from datetime import date, datetime, time
from pathlib import Path
from typing import Literal
from zoneinfo import ZoneInfo

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

TZ = ZoneInfo("Asia/Taipei")
OHLC = ("open", "high", "low", "close")
EPS = 1e-6


def at(day: date, clock: str = "08:30") -> datetime:
    return datetime.combine(day, time.fromisoformat(clock), TZ)


def digest(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":"), allow_nan=False,
                                    default=str).encode()).hexdigest()


def write_once(path: Path, obj) -> None:
    """Exclusive create, fsync, never overwrite an experiment or snapshot."""
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False, default=str)
    with path.open("x", encoding="utf-8") as f:
        f.write(raw + "\n")
        f.flush()
        os.fsync(f.fileno())
    fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


class Prices(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    open: float = Field(gt=0)
    high: float = Field(gt=0)
    low: float = Field(gt=0)
    close: float = Field(gt=0)

    @model_validator(mode="after")
    def coherent(self):
        if self.high < max(self.open, self.close) or self.low > min(self.open, self.close):
            raise ValueError("incoherent OHLC")
        return self

    def array(self):
        return np.array([getattr(self, k) for k in OHLC], dtype=float)


class Bar(Prices):
    instrument: Literal["TAIEX", "TX"]
    contract: str = ""
    trading_date: date
    session: Literal["day", "night"]
    session_start_at: datetime
    session_end_at: datetime
    available_at: datetime
    fetched_at: datetime
    availability_basis: str
    source: str
    source_hash: str
    volume: float | None = Field(default=None, ge=0)
    open_interest: float | None = Field(default=None, ge=0)
    settlement: float | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def timed(self):
        for k in ("session_start_at", "session_end_at", "available_at", "fetched_at"):
            if getattr(self, k).utcoffset() is None:
                raise ValueError("timezone required")
        if not self.session_start_at < self.session_end_at <= self.available_at:
            raise ValueError("invalid session/publication chronology")
        if self.session == "day" and self.session_start_at.astimezone(TZ).date() != self.trading_date:
            raise ValueError("day session date mismatch")
        if self.instrument == "TX" and (len(self.contract) != 6 or not self.contract.isdigit()):
            raise ValueError("TX requires explicit YYYYMM contract")
        if self.instrument == "TAIEX" and (self.contract or self.session != "day"):
            raise ValueError("TAIEX has no futures contract/night session")
        if not self.source or not self.source_hash or not self.availability_basis:
            raise ValueError("provenance required")
        return self


def encode(bar: Prices, previous: float, representation: str) -> np.ndarray:
    if not math.isfinite(previous) or previous <= 0:
        raise ValueError("invalid same-contract reference")
    o, h, l, c = bar.array()
    if representation == "relative":
        return np.log(bar.array() / previous)
    if representation != "gbud":
        raise ValueError("unknown representation")
    return np.array([np.log(o/previous), np.log(c/o),
                     np.log(np.log(h/max(o,c))+EPS),
                     np.log(np.log(min(o,c)/l)+EPS)])


def decode(y, previous: float, representation: str) -> Prices:
    y = np.asarray(y, dtype=float)
    if y.shape != (4,) or not np.all(np.isfinite(y)) or previous <= 0:
        raise ValueError("invalid target")
    with np.errstate(over="raise", invalid="raise", under="ignore"):
        if representation == "relative":
            o, h, l, c = previous * np.exp(y)
            h, l = max(h, o, c), min(l, o, c)
        elif representation == "gbud":
            g, b, zu, zd = y
            u, d = max(0.0, np.exp(zu)-EPS), max(0.0, np.exp(zd)-EPS)
            o = previous*np.exp(g)
            c = o*np.exp(b)
            h, l = max(o,c)*np.exp(u), min(o,c)*np.exp(-d)
        else:
            raise ValueError("unknown representation")
    return Prices(open=o, high=h, low=l, close=c)


class Calendar(BaseModel):
    """Exchange dates and expiries must be supplied, never weekday-inferred."""
    days: list[date]
    expiries: dict[str, date]
    source: str
    available_at: datetime

    @model_validator(mode="after")
    def valid(self):
        if self.days != sorted(set(self.days)) or not self.source:
            raise ValueError("calendar must be sorted, unique, sourced")
        if self.available_at.utcoffset() is None:
            raise ValueError("calendar timestamp timezone required")
        return self

    def previous(self, day: date) -> date:
        i = self.days.index(day)
        if i == 0:
            raise ValueError("calendar lacks previous exchange day")
        return self.days[i-1]

    def contract_for(self, day: date, roll_sessions: int = 5) -> str:
        if roll_sessions < 1:
            raise ValueError("roll_sessions must be positive")
        i = self.days.index(day)
        expiries = sorted(self.expiries.items(), key=lambda item: item[1])
        if not expiries:
            raise ValueError("expiry calendar is empty")
        for contract, expiry in expiries:
            # A bounded support calendar may begin shortly before an already
            # expired contract. It must not require enough leading sessions to
            # recompute a roll decision that predates the requested day.
            if expiry <= day:
                continue
            if expiry not in self.days:
                raise ValueError(
                    f"expiry {contract}/{expiry} is outside the verified exchange calendar"
                )
            ei = self.days.index(expiry)
            if ei < roll_sessions:
                raise ValueError(
                    f"insufficient leading exchange calendar for {contract} roll rule"
                )
            if i < ei-roll_sessions:
                return contract
        contract, expiry = expiries[-1]
        raise ValueError(
            "roll calendar tail lacks a verified successor expiry after "
            f"{contract}/{expiry}"
        )


def asof(bars: list[Bar], day: date) -> list[Bar]:
    cutoff = at(day)
    return [b for b in bars if b.available_at <= cutoff and b.session_end_at <= cutoff
            and not (b.session == "day" and b.trading_date >= day)]


def reference(bars: list[Bar], calendar: Calendar, day: date,
              instrument: str, contract: str) -> Bar:
    prev = calendar.previous(day)
    matches = [b for b in asof(bars, day) if b.instrument == instrument
               and b.contract == contract and b.session == "day" and b.trading_date == prev]
    if len(matches) != 1:
        raise ValueError("missing/duplicate previous exchange-day same-contract close")
    return matches[0]
