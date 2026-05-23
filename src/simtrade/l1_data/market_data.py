from __future__ import annotations

import time
from typing import Callable, Sequence

try:
    import ccxt  # type: ignore[import-untyped]
except ImportError:
    ccxt = None  # ccxt is optional at import time; required at runtime

from simtrade.l1_data.storage import Candle, OHLCVStore

TimeframeMs = {
    "1m": 60_000,
    "5m": 300_000,
    "15m": 900_000,
    "1h": 3_600_000,
    "4h": 14_400_000,
    "1d": 86_400_000,
}


class MarketData:
    """Minimal ccxt wrapper.

    Uses REST polling instead of ccxt.pro websockets so the platform runs
    without a paid subscription. `stream_ohlcv` simulates a feed by
    polling at one-bar-interval cadence.
    """

    def __init__(
        self,
        store: OHLCVStore,
        exchange_id: str = "binance",
        exchange_options: dict | None = None,
    ):
        if ccxt is None:
            raise RuntimeError("ccxt is required for MarketData; pip install ccxt")
        self.store = store
        cls = getattr(ccxt, exchange_id)
        self.exchange = cls(exchange_options or {"enableRateLimit": True})

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        since_ms: int | None = None,
        limit: int = 200,
    ) -> list[Candle]:
        raw = self.exchange.fetch_ohlcv(symbol, timeframe, since=since_ms, limit=limit)
        candles: list[Candle] = [(int(r[0]), r[1], r[2], r[3], r[4], r[5]) for r in raw]
        self.store.upsert(symbol, timeframe, candles, closed=True)
        return candles

    def snapshot_multi_tf(
        self,
        symbol: str,
        timeframes: Sequence[str] = ("15m", "1h", "4h", "1d"),
        bars_per_tf: int = 50,
    ) -> dict[str, list[Candle]]:
        out: dict[str, list[Candle]] = {}
        for tf in timeframes:
            out[tf] = self.fetch_ohlcv(symbol, tf, limit=bars_per_tf)
        return out

    def fetch_orderbook(self, symbol: str, depth: int = 20) -> dict:
        return self.exchange.fetch_order_book(symbol, limit=depth)

    def fetch_funding_rate(self, symbol: str) -> float | None:
        if not hasattr(self.exchange, "fetch_funding_rate"):
            return None
        try:
            data = self.exchange.fetch_funding_rate(symbol)
            return float(data.get("fundingRate", 0.0))
        except Exception:
            return None

    def stream_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        callback: Callable[[Candle], None],
        max_iterations: int | None = None,
    ) -> None:
        interval_s = TimeframeMs[timeframe] / 1000.0
        i = 0
        while max_iterations is None or i < max_iterations:
            try:
                candles = self.fetch_ohlcv(symbol, timeframe, limit=2)
                if candles:
                    callback(candles[-1])
            except Exception as exc:
                print(f"[stream_ohlcv] error: {exc!r}; retrying after backoff")
                time.sleep(min(interval_s, 30))
            time.sleep(interval_s)
            i += 1
