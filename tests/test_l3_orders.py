from simtrade.l3_orders import Order, OrderEngine, OrderSide, OrderType


BOOK = {
    "bids": [[99.0, 5.0], [98.5, 10.0], [98.0, 20.0]],
    "asks": [[100.0, 5.0], [100.5, 10.0], [101.0, 20.0]],
}


def test_market_long_small_order_uses_top_ask():
    eng = OrderEngine()
    o = Order(symbol="BTC", side=OrderSide.LONG, type=OrderType.MARKET, size=0.01)
    fill = eng.submit(o, BOOK, ts_ms=1)
    assert fill is not None
    assert fill.price == 100.0
    assert fill.slippage > 0


def test_market_short_mid_order_walks_book():
    eng = OrderEngine()
    o = Order(symbol="BTC", side=OrderSide.SHORT, type=OrderType.MARKET, size=8.0)
    fill = eng.submit(o, BOOK, ts_ms=1)
    assert fill is not None
    assert 98.5 <= fill.price <= 99.0


def test_stop_order_arms_and_fires_on_next_tick():
    eng = OrderEngine()
    stop = Order(
        symbol="BTC",
        side=OrderSide.LONG,
        type=OrderType.STOP_MARKET,
        size=0.05,
        stop_price=98.0,
    )
    eng.submit(stop, BOOK, ts_ms=1)
    assert len(eng.pending_triggers) == 1
    no_fills = eng.on_tick(last_price=99.5, orderbook=BOOK, ts_ms=2)
    assert no_fills == []
    fills = eng.on_tick(last_price=97.5, orderbook=BOOK, ts_ms=3)
    assert len(fills) == 1
    assert fills[0].side is OrderSide.LONG


def test_limit_order_fills_when_crossed():
    eng = OrderEngine()
    o = Order(symbol="BTC", side=OrderSide.LONG, type=OrderType.LIMIT, size=0.05, price=99.0)
    eng.submit(o, BOOK, ts_ms=1)
    fills = eng.on_tick(last_price=98.5, orderbook=BOOK, ts_ms=2)
    assert len(fills) == 1
    assert fills[0].price == 99.0
