import random
from datetime import datetime, timedelta, timezone

import pytest

from simtrade.l6_learning import DecisionRecord, DecisionStore
from simtrade.l7_discovery import (
    ExperimentStore,
    HiddenCorrelationScanner,
    MetaSkillDiagnostics,
    RegimeDecayDetector,
    SuggestedExperiment,
)


def _make_decisions(
    n: int,
    win_bias_for: tuple[str, str] | None = None,
    bias_strength: float = 0.85,
    seed: int = 0,
):
    rng = random.Random(seed)
    setups = ["mean_reversion", "trend_continuation", "range_play", "breakout"]
    regimes = ["ranging", "trending_up", "trending_down", "pre_breakout"]
    out: list[DecisionRecord] = []
    base_ts = datetime(2026, 1, 1, tzinfo=timezone.utc)
    for i in range(n):
        setup = rng.choice(setups)
        regime = rng.choice(regimes)
        if win_bias_for and setup == win_bias_for[0] and regime == win_bias_for[1]:
            pnl = 1.5 if rng.random() < bias_strength else -1.0
        else:
            pnl = 1.0 if rng.random() < 0.5 else -1.0
        rec = DecisionRecord.new(
            symbol="BTC/USDT",
            market_snapshot={"price": 100.0},
            trader_state={
                "confidence": rng.randint(2, 5),
                "reasoning_tags": {
                    "setup_type": setup,
                    "key_level": None,
                    "indicator_trigger": [],
                    "trader_emotion": rng.choice(["calm", "anxious", "fomo"]),
                    "market_regime": regime,
                },
            },
            action={
                "side": "long",
                "size": 0.05,
                "entry": 100,
                "sl": 99,
                "tp": 102,
                "risk_amount": 100,
                "sl_distance_R": 1.0,
                "tp_distance_R": 2.0,
            },
            ts=base_ts + timedelta(hours=i * 6),
        )
        rec.post_outcome = {"pnl_R": pnl, "prediction_correct": pnl > 0}
        out.append(rec)
    return out


def test_scanner_finds_planted_correlation(conn):
    records = _make_decisions(
        800, win_bias_for=("mean_reversion", "ranging"), bias_strength=0.95, seed=1
    )
    scanner = HiddenCorrelationScanner(conn, min_n=20)
    findings = scanner.scan(records)
    assert any(
        ("setup_type", "mean_reversion") in (f.label_a, f.label_b)
        and ("market_regime", "ranging") in (f.label_a, f.label_b)
        for f in findings
    )


def test_scanner_returns_empty_on_insufficient_data(conn):
    records = _make_decisions(5, seed=2)
    scanner = HiddenCorrelationScanner(conn, min_n=20)
    assert scanner.scan(records) == []


def test_scanner_persists_to_discovery_log(conn):
    records = _make_decisions(
        200, win_bias_for=("breakout", "trending_up"), bias_strength=0.85, seed=3
    )
    scanner = HiddenCorrelationScanner(conn, min_n=15)
    findings = scanner.scan(records)
    scanner.persist(findings)
    n = conn.execute("SELECT COUNT(*) AS c FROM discovery_log").fetchone()["c"]
    assert n == len(findings)


def test_meta_skills_report_shape():
    records = _make_decisions(50, seed=4)
    report = MetaSkillDiagnostics().compute(records)
    assert set(report.radar.keys()) == {
        "confidence_calibration",
        "emotion_control",
        "session_fit",
        "prediction_skill",
    }
    assert all(0 <= v <= 100 for v in report.radar.values())


def test_regime_decay_detector_flags_drop():
    setups = ["mean_reversion"]
    regimes = ["trending_up"]
    base_ts = datetime(2026, 5, 1, tzinfo=timezone.utc)
    records: list[DecisionRecord] = []
    for i in range(40):
        days_ago = 60 + (i % 30)
        pnl = 1.5
        rec = DecisionRecord.new(
            symbol="BTC/USDT",
            market_snapshot={},
            trader_state={
                "confidence": 4,
                "reasoning_tags": {
                    "setup_type": "mean_reversion",
                    "key_level": None,
                    "indicator_trigger": [],
                    "trader_emotion": "calm",
                    "market_regime": "trending_up",
                },
            },
            action={
                "side": "long", "size": 0.05, "entry": 100, "sl": 99, "tp": 102,
                "risk_amount": 100, "sl_distance_R": 1.0, "tp_distance_R": 2.0,
            },
            ts=base_ts - timedelta(days=days_ago),
        )
        rec.post_outcome = {"pnl_R": pnl}
        records.append(rec)
    for i in range(15):
        rec = DecisionRecord.new(
            symbol="BTC/USDT",
            market_snapshot={},
            trader_state={
                "confidence": 4,
                "reasoning_tags": {
                    "setup_type": "mean_reversion",
                    "key_level": None,
                    "indicator_trigger": [],
                    "trader_emotion": "calm",
                    "market_regime": "trending_up",
                },
            },
            action={
                "side": "long", "size": 0.05, "entry": 100, "sl": 99, "tp": 102,
                "risk_amount": 100, "sl_distance_R": 1.0, "tp_distance_R": 2.0,
            },
            ts=base_ts - timedelta(days=i),
        )
        rec.post_outcome = {"pnl_R": -1.0}
        records.append(rec)
    alerts = RegimeDecayDetector().detect(records, now=base_ts)
    assert any(a.setup == "mean_reversion" and a.regime == "trending_up" for a in alerts)


def test_experiment_store_round_trip(conn):
    store = ExperimentStore(conn)
    exp = SuggestedExperiment(
        id="e1",
        proposed_at=datetime(2026, 1, 1, tzinfo=timezone.utc).isoformat(),
        hypothesis="placeholder",
        conditions={"setup_type": "mean_reversion"},
        target_n=3,
    )
    store.propose(exp)
    active = store.list_active()
    assert len(active) == 1
    assert active[0].id == "e1"
