from __future__ import annotations

import sqlite3
from typing import Iterable, Sequence

Candle = tuple[int, float, float, float, float, float]


class OHLCVStore:
    """SQLite-backed OHLCV cache.

    Candles are upserted by (symbol, timeframe, ts). `closed=0` means the
    bar is still forming and should not be used for signal generation.
    """

    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn

    def upsert(
        self,
        symbol: str,
        timeframe: str,
        candles: Iterable[Candle],
        closed: bool = True,
    ) -> int:
        rows = [
            (symbol, timeframe, int(ts), o, h, l, c, v, 1 if closed else 0)
            for ts, o, h, l, c, v in candles
        ]
        if not rows:
            return 0
        self.conn.executemany(
            """
            INSERT INTO ohlcv (symbol, timeframe, ts, open, high, low, close, volume, closed)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(symbol, timeframe, ts) DO UPDATE SET
                open    = excluded.open,
                high    = excluded.high,
                low     = excluded.low,
                close   = excluded.close,
                volume  = excluded.volume,
                closed  = excluded.closed
            """,
            rows,
        )
        self.conn.commit()
        return len(rows)

    def fetch(
        self,
        symbol: str,
        timeframe: str,
        since_ms: int | None = None,
        limit: int = 200,
        only_closed: bool = True,
    ) -> list[Candle]:
        sql = "SELECT ts, open, high, low, close, volume FROM ohlcv WHERE symbol=? AND timeframe=?"
        params: list = [symbol, timeframe]
        if since_ms is not None:
            sql += " AND ts >= ?"
            params.append(since_ms)
        if only_closed:
            sql += " AND closed = 1"
        sql += " ORDER BY ts ASC LIMIT ?"
        params.append(limit)
        cur = self.conn.execute(sql, params)
        return [tuple(row) for row in cur.fetchall()]  # type: ignore[misc]

    def latest_ts(self, symbol: str, timeframe: str) -> int | None:
        cur = self.conn.execute(
            "SELECT MAX(ts) AS ts FROM ohlcv WHERE symbol=? AND timeframe=? AND closed=1",
            (symbol, timeframe),
        )
        row = cur.fetchone()
        return row["ts"] if row and row["ts"] is not None else None

    def close_at_or_after(self, symbol: str, timeframe: str, ts_ms: int) -> float | None:
        cur = self.conn.execute(
            """
            SELECT close FROM ohlcv
            WHERE symbol=? AND timeframe=? AND ts >= ? AND closed=1
            ORDER BY ts ASC LIMIT 1
            """,
            (symbol, timeframe, ts_ms),
        )
        row = cur.fetchone()
        return float(row["close"]) if row else None

    def range_between(
        self, symbol: str, timeframe: str, start_ms: int, end_ms: int
    ) -> Sequence[Candle]:
        cur = self.conn.execute(
            """
            SELECT ts, open, high, low, close, volume FROM ohlcv
            WHERE symbol=? AND timeframe=? AND ts BETWEEN ? AND ? AND closed=1
            ORDER BY ts ASC
            """,
            (symbol, timeframe, start_ms, end_ms),
        )
        return [tuple(row) for row in cur.fetchall()]  # type: ignore[misc]
