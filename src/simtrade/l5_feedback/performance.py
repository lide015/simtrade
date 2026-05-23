from __future__ import annotations

import json
import math
import sqlite3
from dataclasses import dataclass

import numpy as np


@dataclass
class PerformanceStats:
    n_trades: int
    cumulative_pnl_usdt: float
    cumulative_pnl_R: float
    win_rate: float | None
    avg_win_R: float | None
    avg_loss_R: float | None
    payoff_ratio: float | None
    max_drawdown_R: float
    longest_losing_streak: int
    sharpe_weekly: float | None

    def to_dict(self) -> dict:
        return self.__dict__


def _pnl_R(row: sqlite3.Row) -> float | None:
    if not row["post_outcome"]:
        return None
    try:
        outcome = json.loads(row["post_outcome"])
    except (TypeError, json.JSONDecodeError):
        return None
    val = outcome.get("pnl_R")
    return float(val) if val is not None else None


def _pnl_usdt(row: sqlite3.Row) -> float:
    try:
        outcome = json.loads(row["post_outcome"]) if row["post_outcome"] else {}
        action = json.loads(row["action"])
    except (TypeError, json.JSONDecodeError):
        return 0.0
    pnl_R = outcome.get("pnl_R")
    risk = action.get("risk_amount")
    if pnl_R is None or risk is None:
        return 0.0
    return float(pnl_R) * float(risk)


def compute_performance(conn: sqlite3.Connection) -> PerformanceStats:
    cur = conn.execute(
        "SELECT ts, action, post_outcome FROM decisions "
        "WHERE post_outcome IS NOT NULL ORDER BY ts ASC"
    )
    rows = cur.fetchall()
    pnl_Rs = [(_pnl_R(r) or 0.0) for r in rows]
    pnl_usdts = [_pnl_usdt(r) for r in rows]
    n = len(pnl_Rs)
    if n == 0:
        return PerformanceStats(
            n_trades=0,
            cumulative_pnl_usdt=0.0,
            cumulative_pnl_R=0.0,
            win_rate=None,
            avg_win_R=None,
            avg_loss_R=None,
            payoff_ratio=None,
            max_drawdown_R=0.0,
            longest_losing_streak=0,
            sharpe_weekly=None,
        )

    wins = [r for r in pnl_Rs if r > 0]
    losses = [r for r in pnl_Rs if r < 0]
    cum = np.cumsum(pnl_Rs)
    peak = np.maximum.accumulate(cum)
    drawdown = peak - cum
    max_dd = float(drawdown.max()) if len(drawdown) else 0.0

    longest_losing = 0
    cur_streak = 0
    for r in pnl_Rs:
        if r < 0:
            cur_streak += 1
            longest_losing = max(longest_losing, cur_streak)
        else:
            cur_streak = 0

    sharpe = None
    if n >= 5:
        weekly = _bucket_weekly([r["ts"] for r in rows], pnl_Rs)
        if len(weekly) >= 2:
            arr = np.array(weekly, dtype=float)
            std = arr.std(ddof=1)
            if std > 0:
                sharpe = float(arr.mean() / std * math.sqrt(52))

    avg_win = float(np.mean(wins)) if wins else None
    avg_loss = float(np.mean(losses)) if losses else None
    payoff = (avg_win / abs(avg_loss)) if (avg_win and avg_loss) else None

    return PerformanceStats(
        n_trades=n,
        cumulative_pnl_usdt=float(sum(pnl_usdts)),
        cumulative_pnl_R=float(sum(pnl_Rs)),
        win_rate=len(wins) / n if n else None,
        avg_win_R=avg_win,
        avg_loss_R=avg_loss,
        payoff_ratio=payoff,
        max_drawdown_R=max_dd,
        longest_losing_streak=longest_losing,
        sharpe_weekly=sharpe,
    )


def _bucket_weekly(timestamps: list[str], values: list[float]) -> list[float]:
    from datetime import datetime

    weekly: dict[str, float] = {}
    for ts, v in zip(timestamps, values):
        try:
            dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        except ValueError:
            continue
        year, week, _ = dt.isocalendar()
        key = f"{year}-W{week:02d}"
        weekly[key] = weekly.get(key, 0.0) + v
    return [weekly[k] for k in sorted(weekly.keys())]
