from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import numpy as np

from simtrade.l6_learning.decision import DecisionRecord


@dataclass
class DecayAlert:
    setup: str
    regime: str
    ev_90d: float
    ev_60d: float
    ev_30d: float
    n_90d: int
    severity: str

    def description(self) -> str:
        return (
            f"{self.setup} @ {self.regime} decay: "
            f"90d={self.ev_90d:+.2f}R (n={self.n_90d}), "
            f"60d={self.ev_60d:+.2f}R, 30d={self.ev_30d:+.2f}R [{self.severity}]"
        )


class RegimeDecayDetector:
    """README §6.2.3 — flag (setup x regime) combos whose 30d EV
    fell below half of 90d EV, when 90d EV was non-trivial.
    """

    def detect(
        self, records: list[DecisionRecord], now: datetime | None = None
    ) -> list[DecayAlert]:
        now = now or datetime.now(tz=timezone.utc)
        completed = [r for r in records if self._pnl_R(r) is not None]
        buckets: dict[tuple[str, str], list[tuple[datetime, float]]] = defaultdict(list)
        for r in completed:
            setup = r.trader_state.get("reasoning_tags", {}).get("setup_type")
            regime = r.trader_state.get("reasoning_tags", {}).get("market_regime")
            if not setup or not regime:
                continue
            ts = _parse_iso(r.ts)
            buckets[(setup, regime)].append((ts, self._pnl_R(r)))

        alerts: list[DecayAlert] = []
        for (setup, regime), items in buckets.items():
            ev_30 = self._rolling_ev(items, now, days=30)
            ev_60 = self._rolling_ev(items, now, days=60)
            ev_90 = self._rolling_ev(items, now, days=90)
            n_90 = sum(1 for t, _ in items if (now - t).days <= 90)
            if ev_90 is None or n_90 < 10:
                continue
            if abs(ev_90) <= 0.2:
                continue
            if ev_30 is None:
                continue
            if ev_30 < ev_90 * 0.5:
                severity = "critical" if ev_30 < 0 < ev_90 else "warning"
                alerts.append(
                    DecayAlert(
                        setup=setup,
                        regime=regime,
                        ev_90d=ev_90,
                        ev_60d=ev_60 if ev_60 is not None else 0.0,
                        ev_30d=ev_30,
                        n_90d=n_90,
                        severity=severity,
                    )
                )
        return alerts

    @staticmethod
    def _pnl_R(rec: DecisionRecord) -> float | None:
        if not rec.post_outcome:
            return None
        v = rec.post_outcome.get("pnl_R")
        return float(v) if v is not None else None

    @staticmethod
    def _rolling_ev(
        items: list[tuple[datetime, float]], now: datetime, days: int
    ) -> float | None:
        cutoff = now - timedelta(days=days)
        window = [v for t, v in items if t >= cutoff]
        if not window:
            return None
        return float(np.mean(window))


def _parse_iso(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))
