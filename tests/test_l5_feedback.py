from simtrade.l5_feedback import compute_performance
from simtrade.l6_learning import DecisionRecord, DecisionStore


def _seed_decision(store: DecisionStore, pnl_R: float):
    rec = DecisionRecord.new(
        symbol="BTC/USDT",
        market_snapshot={},
        trader_state={
            "confidence": 3,
            "reasoning_tags": {
                "setup_type": "mean_reversion",
                "key_level": None,
                "indicator_trigger": [],
                "trader_emotion": "calm",
                "market_regime": "ranging",
            },
        },
        action={
            "side": "long", "size": 0.05, "entry": 100, "sl": 99, "tp": 102,
            "risk_amount": 100, "sl_distance_R": 1.0, "tp_distance_R": 2.0,
        },
    )
    store.insert(rec)
    store.fill_outcome(rec.id, {"pnl_R": pnl_R})


def test_empty_returns_zero_stats(conn):
    stats = compute_performance(conn)
    assert stats.n_trades == 0
    assert stats.cumulative_pnl_R == 0.0


def test_basic_stats(conn):
    store = DecisionStore(conn)
    for r in [1.5, -1.0, 2.0, -1.0, 0.5]:
        _seed_decision(store, r)
    stats = compute_performance(conn)
    assert stats.n_trades == 5
    assert stats.cumulative_pnl_R == 2.0
    assert stats.win_rate == 0.6
    assert stats.avg_win_R is not None and stats.avg_win_R > 0
    assert stats.avg_loss_R is not None and stats.avg_loss_R < 0


def test_max_drawdown_tracking(conn):
    store = DecisionStore(conn)
    for r in [2.0, -3.0, -1.0, 5.0]:
        _seed_decision(store, r)
    stats = compute_performance(conn)
    assert stats.max_drawdown_R >= 4.0
