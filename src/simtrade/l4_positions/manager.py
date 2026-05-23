from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone

from simtrade.l3_orders.orders import Fill, OrderSide


@dataclass
class Position:
    id: str
    symbol: str
    side: OrderSide
    size: float
    entry_price: float
    entry_ts: str
    decision_id: str | None = None
    fees_paid: float = 0.0
    realized_pnl: float = 0.0
    exit_price: float | None = None
    exit_ts: str | None = None
    exit_reason: str | None = None

    def unrealized_pnl(self, mark_price: float) -> float:
        if self.side is OrderSide.LONG:
            return (mark_price - self.entry_price) * self.size
        return (self.entry_price - mark_price) * self.size


class PositionManager:
    """Single-direction position tracking (README §4.4 MVP).

    One open position per symbol. Same-side fills average in;
    opposite-side fills reduce or close. Partial closes supported.
    """

    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn
        self.open: dict[str, Position] = {}

    def apply_fill(self, fill: Fill, decision_id: str | None = None) -> Position | None:
        current = self.open.get(fill.symbol)
        if current is None:
            pos = Position(
                id=str(uuid.uuid4()),
                symbol=fill.symbol,
                side=fill.side,
                size=fill.size,
                entry_price=fill.price,
                entry_ts=_ts_to_iso(fill.ts_ms),
                decision_id=decision_id,
                fees_paid=fill.fee,
            )
            self.open[fill.symbol] = pos
            return pos

        if fill.side is current.side:
            total_size = current.size + fill.size
            current.entry_price = (
                current.entry_price * current.size + fill.price * fill.size
            ) / total_size
            current.size = total_size
            current.fees_paid += fill.fee
            return current

        close_size = min(current.size, fill.size)
        pnl = self._directional_pnl(current.side, current.entry_price, fill.price, close_size)
        current.realized_pnl += pnl
        current.fees_paid += fill.fee
        current.size -= close_size

        if current.size <= 1e-12:
            current.exit_price = fill.price
            current.exit_ts = _ts_to_iso(fill.ts_ms)
            current.exit_reason = current.exit_reason or "manual"
            self._persist(current)
            self.open.pop(fill.symbol, None)
            if fill.size > close_size:
                residual = Fill(
                    order_id=fill.order_id,
                    symbol=fill.symbol,
                    side=fill.side,
                    size=fill.size - close_size,
                    price=fill.price,
                    fee=0.0,
                    slippage=fill.slippage,
                    ts_ms=fill.ts_ms,
                )
                return self.apply_fill(residual, decision_id=decision_id)
            return None
        return current

    def mark_exit_reason(self, symbol: str, reason: str) -> None:
        pos = self.open.get(symbol)
        if pos is not None:
            pos.exit_reason = reason

    def open_position(self, symbol: str) -> Position | None:
        return self.open.get(symbol)

    @staticmethod
    def _directional_pnl(side: OrderSide, entry: float, exit_: float, size: float) -> float:
        return (exit_ - entry) * size if side is OrderSide.LONG else (entry - exit_) * size

    def _persist(self, pos: Position) -> None:
        self.conn.execute(
            """
            INSERT OR REPLACE INTO positions
            (id, decision_id, symbol, side, size, entry_price, entry_ts,
             exit_price, exit_ts, exit_reason, realized_pnl)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                pos.id,
                pos.decision_id,
                pos.symbol,
                pos.side.value,
                pos.size,
                pos.entry_price,
                pos.entry_ts,
                pos.exit_price,
                pos.exit_ts,
                pos.exit_reason,
                pos.realized_pnl,
            ),
        )
        self.conn.commit()


def _ts_to_iso(ts_ms: int) -> str:
    return datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone.utc).isoformat()
