from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from simtrade.db import init_db
from simtrade.l1_data import MarketData, OHLCVStore
from simtrade.l2_indicators import (
    compute_indicators_multi_tf,
    detect_regime,
)
from simtrade.l3_orders import OrderEngine
from simtrade.l4_positions import PositionManager
from simtrade.l5_feedback import compute_performance
from simtrade.l6_learning import (
    DecisionRecord,
    DecisionStore,
    Reconciler,
    attribution_by_tag,
    attribution_cross,
)
from simtrade.l7_discovery import (
    ExperimentStore,
    HiddenCorrelationScanner,
    MetaSkillDiagnostics,
    RegimeDecayDetector,
    SuggestedExperiment,
)


@dataclass
class PlatformContext:
    conn: sqlite3.Connection
    ohlcv: OHLCVStore
    decisions: DecisionStore
    positions: PositionManager
    orders: OrderEngine
    experiments: ExperimentStore
    reconciler: Reconciler
    scanner: HiddenCorrelationScanner
    meta_skills: MetaSkillDiagnostics
    decay: RegimeDecayDetector
    market: MarketData | None = None


def boot(
    db_path: str | Path = "data/simtrade.db",
    exchange_id: str | None = "binance",
    with_market: bool = True,
) -> PlatformContext:
    conn = init_db(db_path)
    ohlcv = OHLCVStore(conn)
    decisions = DecisionStore(conn)
    positions = PositionManager(conn)
    orders = OrderEngine()
    experiments = ExperimentStore(conn)
    reconciler = Reconciler(decisions, ohlcv)
    scanner = HiddenCorrelationScanner(conn)
    meta = MetaSkillDiagnostics()
    decay = RegimeDecayDetector()
    market = None
    if with_market and exchange_id:
        try:
            market = MarketData(ohlcv, exchange_id=exchange_id)
        except RuntimeError:
            market = None
    return PlatformContext(
        conn=conn,
        ohlcv=ohlcv,
        decisions=decisions,
        positions=positions,
        orders=orders,
        experiments=experiments,
        reconciler=reconciler,
        scanner=scanner,
        meta_skills=meta,
        decay=decay,
        market=market,
    )


def record_decision(
    ctx: PlatformContext,
    symbol: str,
    market_snapshot: dict[str, Any],
    trader_state: dict[str, Any],
    action: dict[str, Any],
) -> DecisionRecord:
    rec = DecisionRecord.new(
        symbol=symbol,
        market_snapshot=market_snapshot,
        trader_state=trader_state,
        action=action,
    )
    ctx.decisions.insert(rec)
    return rec


def weekly_discovery_report(ctx: PlatformContext) -> dict:
    """README §6.4 — full scan + suggested experiments + decay alerts."""
    records = ctx.decisions.completed()
    findings = ctx.scanner.scan(records)
    ctx.scanner.persist(findings)
    decay_alerts = ctx.decay.detect(records)
    meta_report = ctx.meta_skills.compute(records)
    suggestions: list[SuggestedExperiment] = []
    for f in findings[:3]:
        exp = SuggestedExperiment.from_finding(f)
        ctx.experiments.propose(exp)
        suggestions.append(exp)
    perf = compute_performance(ctx.conn)
    return {
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        "performance": perf.to_dict(),
        "findings": [f.description() for f in findings],
        "decay_alerts": [a.description() for a in decay_alerts],
        "meta_skills": {
            "radar": meta_report.radar,
            "confidence_calibration": meta_report.confidence_calibration,
            "emotion_impact_p": meta_report.emotion_impact.get("p_value"),
            "session_sweet_spot": meta_report.session_fit.get("sweet_spot"),
            "prediction_trend": meta_report.prediction_skill_trend.get("trend"),
        },
        "suggested_experiments": [
            {"id": s.id, "hypothesis": s.hypothesis, "conditions": s.conditions}
            for s in suggestions
        ],
    }
