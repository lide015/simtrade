from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

from simtrade.l6_learning.decision import DecisionRecord

MIN_SAMPLE_WARNING = 20


def _pnl_R(rec: DecisionRecord) -> float | None:
    if not rec.post_outcome:
        return None
    val = rec.post_outcome.get("pnl_R")
    return float(val) if val is not None else None


def _tag_value(rec: DecisionRecord, dim: str) -> Any:
    return rec.trader_state.get("reasoning_tags", {}).get(dim)


def attribution_by_tag(
    records: list[DecisionRecord], dim: str = "setup_type"
) -> dict[str, dict]:
    """README §5.4 — group by tag value, report n / win_rate / EV / warning."""
    buckets: dict[str, list[float]] = defaultdict(list)
    for rec in records:
        pnl = _pnl_R(rec)
        if pnl is None:
            continue
        key = _tag_value(rec, dim)
        if isinstance(key, list):
            for k in key:
                buckets[str(k)].append(pnl)
        else:
            buckets[str(key)].append(pnl)

    return {
        tag: {
            "n": len(rs),
            "win_rate": sum(1 for r in rs if r > 0) / len(rs) if rs else None,
            "avg_R": float(np.mean(rs)) if rs else None,
            "EV_R": float(np.mean(rs)) if rs else None,
            "stdev_R": float(np.std(rs, ddof=1)) if len(rs) >= 2 else None,
            "warning": f"n<{MIN_SAMPLE_WARNING}" if len(rs) < MIN_SAMPLE_WARNING else None,
        }
        for tag, rs in buckets.items()
    }


def attribution_cross(
    records: list[DecisionRecord], dim_a: str = "setup_type", dim_b: str = "market_regime"
) -> dict[tuple[str, str], dict]:
    """README §5.5 — cartesian breakdown of two tag dimensions."""
    buckets: dict[tuple[str, str], list[float]] = defaultdict(list)
    for rec in records:
        pnl = _pnl_R(rec)
        if pnl is None:
            continue
        a = _tag_value(rec, dim_a)
        b = _tag_value(rec, dim_b)
        buckets[(str(a), str(b))].append(pnl)

    return {
        key: {
            "n": len(rs),
            "win_rate": sum(1 for r in rs if r > 0) / len(rs) if rs else None,
            "EV_R": float(np.mean(rs)) if rs else None,
            "warning": f"n<{MIN_SAMPLE_WARNING}" if len(rs) < MIN_SAMPLE_WARNING else None,
        }
        for key, rs in buckets.items()
    }


def confidence_calibration(records: list[DecisionRecord]) -> dict:
    """Linear fit of confidence (1..5) vs realized R.

    Healthy: slope > 0, R^2 > 0.3 (README §6.2.2).
    """
    xs: list[float] = []
    ys: list[float] = []
    for rec in records:
        pnl = _pnl_R(rec)
        if pnl is None:
            continue
        c = rec.trader_state.get("confidence")
        if c is None:
            continue
        xs.append(float(c))
        ys.append(pnl)
    if len(xs) < 5:
        return {"n": len(xs), "warning": "n<5, fit unstable"}
    x = np.array(xs)
    y = np.array(ys)
    slope, intercept = np.polyfit(x, y, 1)
    y_pred = slope * x + intercept
    ss_res = float(np.sum((y - y_pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else None
    return {
        "n": len(xs),
        "slope": float(slope),
        "intercept": float(intercept),
        "r_squared": r2,
        "healthy": bool(slope > 0 and (r2 or 0) > 0.3),
    }
