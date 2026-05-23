from simtrade.l1_data import OHLCVStore


def test_upsert_and_fetch(conn):
    store = OHLCVStore(conn)
    candles = [(i * 60_000, 100.0, 101.0, 99.0, 100.5, 1.0) for i in range(10)]
    n = store.upsert("BTC/USDT", "1m", candles)
    assert n == 10
    out = store.fetch("BTC/USDT", "1m", limit=10)
    assert len(out) == 10
    assert out[0][0] == 0


def test_upsert_overwrites_same_ts(conn):
    store = OHLCVStore(conn)
    store.upsert("BTC/USDT", "1m", [(1000, 1, 1, 1, 1, 1)])
    store.upsert("BTC/USDT", "1m", [(1000, 2, 2, 2, 2, 2)])
    out = store.fetch("BTC/USDT", "1m")
    assert len(out) == 1
    assert out[0][1] == 2


def test_close_at_or_after(conn):
    store = OHLCVStore(conn)
    store.upsert("BTC/USDT", "1h", [(0, 1, 1, 1, 100.0, 1), (3_600_000, 1, 1, 1, 200.0, 1)])
    assert store.close_at_or_after("BTC/USDT", "1h", 0) == 100.0
    assert store.close_at_or_after("BTC/USDT", "1h", 3_600_000) == 200.0
    assert store.close_at_or_after("BTC/USDT", "1h", 7_200_000) is None
