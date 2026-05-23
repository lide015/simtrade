from simtrade.l3_orders import Fill, OrderSide
from simtrade.l4_positions import PositionManager


def _fill(side: OrderSide, size: float, price: float, ts: int = 0) -> Fill:
    return Fill(
        order_id="x", symbol="BTC", side=side, size=size, price=price,
        fee=0.0, slippage=0.0, ts_ms=ts,
    )


def test_open_then_close_with_pnl(conn):
    pm = PositionManager(conn)
    pm.apply_fill(_fill(OrderSide.LONG, 1.0, 100.0))
    pm.apply_fill(_fill(OrderSide.SHORT, 1.0, 110.0))
    assert pm.open_position("BTC") is None
    cur = conn.execute("SELECT realized_pnl, exit_price FROM positions").fetchone()
    assert cur["realized_pnl"] == 10.0
    assert cur["exit_price"] == 110.0


def test_partial_close_keeps_position(conn):
    pm = PositionManager(conn)
    pm.apply_fill(_fill(OrderSide.LONG, 2.0, 100.0))
    pm.apply_fill(_fill(OrderSide.SHORT, 0.5, 105.0))
    pos = pm.open_position("BTC")
    assert pos is not None
    assert pos.size == 1.5
    assert pos.realized_pnl == 2.5


def test_averaging_in(conn):
    pm = PositionManager(conn)
    pm.apply_fill(_fill(OrderSide.LONG, 1.0, 100.0))
    pm.apply_fill(_fill(OrderSide.LONG, 1.0, 110.0))
    pos = pm.open_position("BTC")
    assert pos is not None
    assert pos.size == 2.0
    assert pos.entry_price == 105.0


def test_short_pnl(conn):
    pm = PositionManager(conn)
    pm.apply_fill(_fill(OrderSide.SHORT, 1.0, 100.0))
    pm.apply_fill(_fill(OrderSide.LONG, 1.0, 90.0))
    cur = conn.execute("SELECT realized_pnl FROM positions").fetchone()
    assert cur["realized_pnl"] == 10.0
