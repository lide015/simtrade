from __future__ import annotations

import json
import sqlite3
import uuid
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import combinations
from typing import Any, Iterable

import numpy as np
from scipy.stats import chi2_contingency

from simtrade.l6_learning.decision import DecisionRecord

MIN_N_PER_CELL = 20
DEFAULT_ALPHA = 0.05


@dataclass
class Finding:
    label_a: tuple[str, str]
    label_b: tuple[str, str]
    n_subset: int
    n_total: int
    win_rate_subset: float
    win_rate_baseline: float
    effect_size: float
    p_value: float
    p_value_corrected: float

    def description(self) -> str:
        a_dim, a_val = self.label_a
        b_dim, b_val = self.label_b
        return (
            f"{a_dim}={a_val} & {b_dim}={b_val} "
            f"-> win {self.win_rate_subset:.0%} (n={self.n_subset}) "
            f"vs base {self.win_rate_baseline:.0%}, "
            f"p={self.p_value:.4f} (BH={self.p_value_corrected:.4f})"
        )


class HiddenCorrelationScanner:
    """Pairwise tag-combination scanner (README §6.2.1).

    For each (label_A_value, label_B_value) pair across reasoning_tags
    dimensions, compute chi-square test of win rate vs baseline, then
    apply Benjamini-Hochberg correction.
    """

    def __init__(
        self,
        conn: sqlite3.Connection,
        dims: Iterable[str] = (
            "setup_type",
            "key_level",
            "market_regime",
            "trader_emotion",
        ),
        alpha: float = DEFAULT_ALPHA,
        min_n: int = MIN_N_PER_CELL,
    ):
        self.conn = conn
        self.dims = list(dims)
        self.alpha = alpha
        self.min_n = min_n

    def scan(self, records: list[DecisionRecord]) -> list[Finding]:
        completed = [r for r in records if self._pnl_R(r) is not None]
        if len(completed) < self.min_n:
            return []

        baseline_win = sum(1 for r in completed if self._pnl_R(r) > 0) / len(completed)
        raw: list[Finding] = []

        for dim_a, dim_b in combinations(self.dims, 2):
            values_a = self._values(completed, dim_a)
            values_b = self._values(completed, dim_b)
            for va in values_a:
                for vb in values_b:
                    subset = [
                        r
                        for r in completed
                        if self._has_value(r, dim_a, va) and self._has_value(r, dim_b, vb)
                    ]
                    if len(subset) < self.min_n:
                        continue
                    subset_wins = sum(1 for r in subset if self._pnl_R(r) > 0)
                    subset_losses = len(subset) - subset_wins
                    other_wins = sum(1 for r in completed if self._pnl_R(r) > 0) - subset_wins
                    other_losses = (len(completed) - len(subset)) - other_wins
                    table = [[subset_wins, subset_losses], [other_wins, other_losses]]
                    if any(sum(row) == 0 for row in table) or any(
                        sum(col) == 0 for col in zip(*table)
                    ):
                        continue
                    chi2, p, _, _ = chi2_contingency(table, correction=False)
                    subset_win_rate = subset_wins / len(subset)
                    raw.append(
                        Finding(
                            label_a=(dim_a, va),
                            label_b=(dim_b, vb),
                            n_subset=len(subset),
                            n_total=len(completed),
                            win_rate_subset=subset_win_rate,
                            win_rate_baseline=baseline_win,
                            effect_size=subset_win_rate - baseline_win,
                            p_value=float(p),
                            p_value_corrected=float(p),
                        )
                    )

        corrected = _benjamini_hochberg(raw, alpha=self.alpha)
        return sorted(
            corrected, key=lambda f: (abs(f.effect_size), -f.p_value_corrected), reverse=True
        )

    def persist(self, findings: list[Finding]) -> None:
        for f in findings:
            self.conn.execute(
                """
                INSERT INTO discovery_log
                (id, detected_at, type, pattern_description, p_value, effect_size, n_samples)
                VALUES (?, ?, 'correlation', ?, ?, ?, ?)
                """,
                (
                    str(uuid.uuid4()),
                    datetime.now(tz=timezone.utc).isoformat(),
                    f.description(),
                    f.p_value_corrected,
                    f.effect_size,
                    f.n_subset,
                ),
            )
        self.conn.commit()

    @staticmethod
    def _pnl_R(rec: DecisionRecord) -> float | None:
        if not rec.post_outcome:
            return None
        v = rec.post_outcome.get("pnl_R")
        return float(v) if v is not None else None

    @staticmethod
    def _values(records: list[DecisionRecord], dim: str) -> set[str]:
        out: set[str] = set()
        for r in records:
            v = r.trader_state.get("reasoning_tags", {}).get(dim)
            if v is None:
                continue
            if isinstance(v, list):
                out.update(str(x) for x in v)
            else:
                out.add(str(v))
        return out

    @staticmethod
    def _has_value(rec: DecisionRecord, dim: str, value: str) -> bool:
        v = rec.trader_state.get("reasoning_tags", {}).get(dim)
        if v is None:
            return False
        if isinstance(v, list):
            return value in (str(x) for x in v)
        return str(v) == value


def _benjamini_hochberg(findings: list[Finding], alpha: float) -> list[Finding]:
    """BH-FDR correction. Returns only findings passing corrected threshold."""
    if not findings:
        return []
    ranked = sorted(findings, key=lambda f: f.p_value)
    m = len(ranked)
    kept: list[Finding] = []
    for i, f in enumerate(ranked, start=1):
        bh_threshold = (i / m) * alpha
        p_corr = min(f.p_value * m / i, 1.0)
        f.p_value_corrected = p_corr
        if f.p_value <= bh_threshold:
            kept.append(f)
    return kept
