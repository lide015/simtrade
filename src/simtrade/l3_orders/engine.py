from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass

from simtrade.l3_orders.orders import Fill, Order, OrderSide, OrderType


@dataclass
class FeeSchedule:
    taker_bps: float = 5.0
    maker_bps: float = 2.0


class OrderEngine:
    """Simulates fills against an orderbook snapshot.

    Slippage model (README §4.3):
        small (<0.5% of top-N depth): flat 0.05% slippage
        mid (0.5%-2%): walk the book linearly
        large (>2%): walk the book + 0.1% penalty
    Triggered stop/limit orders enter a queue and fill on the *next* tick
    to avoid hindsight bias.
    """

    def __init__(self, fees: FeeSchedule | None = None):
        self.fees = fees or FeeSchedule()
        self.pending_triggers: deque[Order] = deque()
        self.open_limits: list[Order] = []

    def submit(
        self,
        order: Order,
        orderbook: dict,
        ts_ms: int | None = None,
        is_taker: bool = True,
    ) -> Fill | None:
        ts_ms = ts_ms if ts_ms is not None else int(time.time() * 1000)
        if order.type is OrderType.MARKET:
            return self._fill_market(order, orderbook, ts_ms, is_taker=True)
        if order.type is OrderType.LIMIT:
            self.open_limits.append(order)
            order.status = "open"
            return None
        if order.type in (OrderType.STOP_MARKET, OrderType.STOP_LIMIT, OrderType.TAKE_PROFIT):
            order.status = "armed"
            self.pending_triggers.append(order)
            return None
        raise ValueError(f"unsupported order type: {order.type}")

    def on_tick(
        self, last_price: float, orderbook: dict, ts_ms: int | None = None
    ) -> list[Fill]:
        ts_ms = ts_ms if ts_ms is not None else int(time.time() * 1000)
        fills: list[Fill] = []

        still_pending: deque[Order] = deque()
        while self.pending_triggers:
            o = self.pending_triggers.popleft()
            if self._stop_triggered(o, last_price):
                fill = self._fill_market(o, orderbook, ts_ms, is_taker=True)
                if fill is not None:
                    fills.append(fill)
            else:
                still_pending.append(o)
        self.pending_triggers = still_pending

        remaining_limits: list[Order] = []
        for o in self.open_limits:
            if self._limit_crossed(o, last_price):
                fill = self._fill_at(o, o.price or last_price, ts_ms, is_taker=False)
                fills.append(fill)
            else:
                remaining_limits.append(o)
        self.open_limits = remaining_limits

        return fills

    def _stop_triggered(self, order: Order, last_price: float) -> bool:
        sp = order.stop_price
        if sp is None:
            return False
        if order.type is OrderType.TAKE_PROFIT:
            return last_price >= sp if order.side is OrderSide.LONG else last_price <= sp
        return last_price <= sp if order.side is OrderSide.LONG else last_price >= sp

    def _limit_crossed(self, order: Order, last_price: float) -> bool:
        if order.price is None:
            return False
        if order.side is OrderSide.LONG:
            return last_price <= order.price
        return last_price >= order.price

    def _fill_market(
        self, order: Order, orderbook: dict, ts_ms: int, is_taker: bool
    ) -> Fill | None:
        levels = orderbook.get("asks" if order.side is OrderSide.LONG else "bids", [])
        if not levels:
            return None
        mid = self._mid_price(orderbook)
        depth = sum(lvl[1] for lvl in levels)
        if depth == 0:
            return None
        ratio = order.size / depth
        if ratio < 0.005:
            slip_bps = 5.0
            avg_price = float(levels[0][0])
        elif ratio < 0.02:
            avg_price = self._walk_book(levels, order.size)
            slip_bps = max(5.0, abs(avg_price - mid) / mid * 10_000) if mid else 5.0
        else:
            avg_price = self._walk_book(levels, order.size) * (
                1.001 if order.side is OrderSide.LONG else 0.999
            )
            slip_bps = max(10.0, abs(avg_price - mid) / mid * 10_000) if mid else 10.0
        return self._fill_at(order, avg_price, ts_ms, is_taker=is_taker, slip_bps=slip_bps)

    def _fill_at(
        self,
        order: Order,
        price: float,
        ts_ms: int,
        is_taker: bool,
        slip_bps: float = 0.0,
    ) -> Fill:
        fee_bps = self.fees.taker_bps if is_taker else self.fees.maker_bps
        fee = price * order.size * fee_bps / 10_000.0
        order.status = "filled"
        return Fill(
            order_id=order.id,
            symbol=order.symbol,
            side=order.side,
            size=order.size,
            price=price,
            fee=fee,
            slippage=slip_bps,
            ts_ms=ts_ms,
        )

    def _walk_book(self, levels: list[list[float]], size: float) -> float:
        remaining = size
        cost = 0.0
        for price, qty in levels:
            take = min(remaining, qty)
            cost += take * price
            remaining -= take
            if remaining <= 0:
                break
        filled = size - max(remaining, 0.0)
        if filled == 0:
            return float(levels[-1][0])
        return cost / filled

    def _mid_price(self, orderbook: dict) -> float | None:
        bids = orderbook.get("bids", [])
        asks = orderbook.get("asks", [])
        if not bids or not asks:
            return None
        return (float(bids[0][0]) + float(asks[0][0])) / 2.0
