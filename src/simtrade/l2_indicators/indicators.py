from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

Candle = tuple[int, float, float, float, float, float]


def _to_df(candles: Sequence[Candle]) -> pd.DataFrame:
    df = pd.DataFrame(candles, columns=["ts", "open", "high", "low", "close", "volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    return df.set_index("ts")


def _rsi(close: pd.Series, length: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / length, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / length, adjust=False).mean()
    rsi = 100 - (100 / (1 + gain / loss.where(loss != 0)))
    rsi = rsi.where(~(loss.eq(0) & gain.gt(0)), 100.0)
    rsi = rsi.where(~(loss.eq(0) & gain.eq(0)), 50.0)
    return rsi


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14) -> pd.Series:
    prev = close.shift(1)
    tr = pd.concat([(high - low), (high - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / length, adjust=False).mean()


def _macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    sig = macd.ewm(span=signal, adjust=False).mean()
    hist = macd - sig
    return macd, sig, hist


def _bbands(close: pd.Series, length: int = 20, mult: float = 2.0):
    mid = close.rolling(length).mean()
    std = close.rolling(length).std(ddof=0)
    upper = mid + mult * std
    lower = mid - mult * std
    return upper, mid, lower


def compute_indicators(candles: Sequence[Candle]) -> dict[str, float | None]:
    """Compute a fixed indicator set on the most recent closed bar."""
    if not candles:
        return {}
    df = _to_df(candles)
    close = df["close"]
    high = df["high"]
    low = df["low"]
    vol = df["volume"]

    rsi = _rsi(close)
    atr = _atr(high, low, close)
    macd_line, macd_sig, macd_hist = _macd(close)
    bb_up, bb_mid, bb_lo = _bbands(close)
    vol_ma = vol.rolling(20).mean()

    def _last(series: pd.Series) -> float | None:
        if len(series) == 0 or pd.isna(series.iloc[-1]):
            return None
        return float(series.iloc[-1])

    return {
        "close": _last(close),
        "rsi": _last(rsi),
        "atr": _last(atr),
        "macd": _last(macd_line),
        "macd_signal": _last(macd_sig),
        "macd_hist": _last(macd_hist),
        "bb_upper": _last(bb_up),
        "bb_mid": _last(bb_mid),
        "bb_lower": _last(bb_lo),
        "volume_ma": _last(vol_ma),
    }


def compute_indicators_multi_tf(
    multi_tf_candles: dict[str, Sequence[Candle]],
) -> dict[str, dict[str, float | None]]:
    return {tf: compute_indicators(candles) for tf, candles in multi_tf_candles.items()}


def detect_regime(candles: Sequence[Candle], lookback_bars: int = 50) -> str:
    """Heuristic regime classifier from README §3.2.1.

    Returns one of: ranging, trending_up, trending_down, pre_breakout, post_breakout.
    """
    if len(candles) < lookback_bars:
        return "ranging"
    df = _to_df(candles).tail(lookback_bars)
    close = df["close"]
    high = df["high"]
    low = df["low"]

    atr_series = _atr(high, low, close)
    atr_recent = atr_series.tail(10).mean()
    atr_baseline = atr_series.tail(lookback_bars).mean()
    if pd.isna(atr_recent) or pd.isna(atr_baseline) or atr_baseline == 0:
        return "ranging"
    atr_ratio = float(atr_recent / atr_baseline)

    half = lookback_bars // 2
    early_mid = close.iloc[:half].mean()
    late_mid = close.iloc[half:].mean()
    drift = (late_mid - early_mid) / early_mid

    if atr_ratio < 0.7:
        return "pre_breakout"
    if atr_ratio > 1.5 and abs(drift) > 0.02:
        return "post_breakout"
    if drift > 0.03:
        return "trending_up"
    if drift < -0.03:
        return "trending_down"
    return "ranging"
