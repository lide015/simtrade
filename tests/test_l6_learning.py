from datetime import datetime, timedelta, timezone

import pytest

from simtrade.l1_data import OHLCVStore
from simtrade.l6_learning import (
    DecisionRecord,
    DecisionStore,
    Reconciler,
    attribution_by_tag,
    derive_descriptive_tags,
    validate_reasoning_tags,
)


def _good_tags(**overrides):
    base = dict(
        setup_type="mean_reversion",
        key_level="round_number",
        indicator_trigger=["rsi_oversold_1h"],
        trader_emotion="calm",
        market_regime="ranging",
    )
    base.update(overrides)
    return base


def _good_action(side: str = "long", **overrides):
    base = dict(
        side=side,
        size=0.05,
        entry=100.0,
        sl=99.0,
        tp=102.0,
        risk_amount=100.0,
        sl_distance_R=1.0,
        tp_distance_R=2.0,
    )
    base.update(overrides)
    return base


def test_validate_reasoning_tags_accepts_good():
    assert validate_reasoning_tags(_good_tags()) == []


def test_validate_reasoning_tags_rejects_unknown_setup():
    issues = validate_reasoning_tags(_good_tags(setup_type="moonshot"))
    assert any("setup_type" in i for i in issues)


def test_validate_reasoning_tags_rejects_forbidden_result_tag():
    issues = validate_reasoning_tags(_good_tags(setup_type="bullish"))
    assert any("forbidden" in i for i in issues)


def test_validate_reasoning_tags_caps_indicator_triggers_at_two():
    issues = validate_reasoning_tags(
        _good_tags(indicator_trigger=[
            "rsi_oversold_1h", "macd_cross_bull", "bb_lower_touch",
        ])
    )
    assert any("max 2" in i for i in issues)


def test_decision_store_insert_and_get(conn):
    store = DecisionStore(conn)
    rec = DecisionRecord.new(
        symbol="BTC/USDT",
        market_snapshot={"price": 100.0},
        trader_state={"confidence": 4, "reasoning_tags": _good_tags()},
        action=_good_action(),
    )
    store.insert(rec)
    fetched = store.get(rec.id)
    assert fetched is not None
    assert fetched.symbol == "BTC/USDT"


def test_decision_store_rejects_missing_R_fields(conn):
    store = DecisionStore(conn)
    bad_action = _good_action()
    del bad_action["risk_amount"]
    rec = DecisionRecord.new(
        symbol="BTC/USDT",
        market_snapshot={},
        trader_state={"confidence": 4, "reasoning_tags": _good_tags()},
        action=bad_action,
    )
    with pytest.raises(ValueError, match="risk_amount"):
        store.insert(rec)


def test_decision_store_locks_outcome(conn):
    store = DecisionStore(conn)
    rec = DecisionRecord.new(
        symbol="BTC/USDT",
        market_snapshot={},
        trader_state={"confidence": 4, "reasoning_tags": _good_tags()},
        action=_good_action(),
    )
    store.insert(rec)
    store.fill_outcome(rec.id, {"pnl_R": 1.5})
    with pytest.raises(ValueError, match="locked"):
        store.fill_outcome(rec.id, {"pnl_R": 999.0})


def test_derive_descriptive_tags_session_inference():
    ts = datetime(2026, 5, 1, 3, 0, tzinfo=timezone.utc)
    tags = derive_descriptive_tags({"1h": {"rsi": 25, "close": 100, "bb_lower": 105}}, ts=ts)
    assert tags["session"] == "asia"
    assert "rsi_oversold_1h" in tags["derived_indicator_trigger"]
    assert "bb_lower_touch" in tags["derived_indicator_trigger"]


def test_reconciler_fills_outcome(conn):
    ohlcv = OHLCVStore(conn)
    decisions = DecisionStore(conn)
    entry_dt = datetime(2026, 5, 1, 0, 0, tzinfo=timezone.utc)
    for h in range(0, 26):
        ts_ms = int((entry_dt + timedelta(hours=h)).timestamp() * 1000)
        price = 100.0 + h * 0.1
        ohlcv.upsert("BTC/USDT", "1h", [(ts_ms, price, price + 0.5, price - 0.5, price, 1.0)])

    rec = DecisionRecord.new(
        symbol="BTC/USDT",
        market_snapshot={"price": 100.0},
        trader_state={"confidence": 4, "reasoning_tags": _good_tags()},
        action=_good_action(side="long", entry=100.0, sl=99.0, tp=102.0),
        ts=entry_dt,
    )
    decisions.insert(rec)
    rec_check = decisions.get(rec.id)
    assert rec_check.post_outcome is None

    reconciler = Reconciler(decisions, ohlcv)
    now = entry_dt + timedelta(hours=25)
    n = reconciler.run_once(now=now)
    assert n == 1
    filled = decisions.get(rec.id)
    assert filled.post_outcome is not None
    assert filled.post_outcome["pnl_R"] > 0


def test_attribution_by_tag(conn):
    decisions = DecisionStore(conn)
    for i in range(5):
        rec = DecisionRecord.new(
            symbol="BTC/USDT",
            market_snapshot={},
            trader_state={"confidence": 3, "reasoning_tags": _good_tags()},
            action=_good_action(),
        )
        decisions.insert(rec)
        decisions.fill_outcome(rec.id, {"pnl_R": 1.0 if i % 2 == 0 else -1.0})
    result = attribution_by_tag(decisions.completed(), dim="setup_type")
    assert "mean_reversion" in result
    assert result["mean_reversion"]["n"] == 5
    assert result["mean_reversion"]["warning"] == "n<20"
