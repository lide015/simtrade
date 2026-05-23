from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
from scipy.stats import f_oneway

from simtrade.l6_learning.attribution import confidence_calibration
from simtrade.l6_learning.decision import DecisionRecord


@dataclass
class MetaSkillReport:
    confidence_calibration: dict
    emotion_impact: dict
    session_fit: dict
    prediction_skill_trend: dict
    radar: dict[str, float]


class MetaSkillDiagnostics:
    """README §6.2.2 — four-axis meta-skill report + 0-100 radar."""

    def compute(self, records: list[DecisionRecord]) -> MetaSkillReport:
        completed = [r for r in records if self._pnl_R(r) is not None]
        return MetaSkillReport(
            confidence_calibration=confidence_calibration(completed),
            emotion_impact=self._anova_by_tag(completed, "trader_emotion"),
            session_fit=self._anova_by_session(completed),
            prediction_skill_trend=self._prediction_trend(completed),
            radar=self._radar(completed),
        )

    def _anova_by_tag(self, records: list[DecisionRecord], dim: str) -> dict:
        groups = defaultdict(list)
        for r in records:
            tag = r.trader_state.get("reasoning_tags", {}).get(dim)
            if tag is None:
                continue
            groups[str(tag)].append(self._pnl_R(r))
        valid = {k: v for k, v in groups.items() if len(v) >= 3}
        if len(valid) < 2:
            return {"n_groups": len(valid), "warning": "need >=2 groups with n>=3"}
        f_stat, p = f_oneway(*valid.values())
        return {
            "groups": {k: {"n": len(v), "mean_R": float(np.mean(v))} for k, v in valid.items()},
            "f_stat": float(f_stat),
            "p_value": float(p),
            "significant": bool(p < 0.05),
        }

    def _anova_by_session(self, records: list[DecisionRecord]) -> dict:
        groups = defaultdict(list)
        for r in records:
            session = (
                r.market_snapshot.get("session")
                or r.trader_state.get("reasoning_tags", {}).get("session")
            )
            if session is None:
                continue
            groups[str(session)].append(self._pnl_R(r))
        valid = {k: v for k, v in groups.items() if len(v) >= 3}
        if len(valid) < 2:
            return {"n_groups": len(valid), "warning": "need >=2 sessions with n>=3"}
        f_stat, p = f_oneway(*valid.values())
        sweet = max(valid.items(), key=lambda kv: float(np.mean(kv[1])))[0]
        return {
            "groups": {k: {"n": len(v), "mean_R": float(np.mean(v))} for k, v in valid.items()},
            "sweet_spot": sweet,
            "f_stat": float(f_stat),
            "p_value": float(p),
            "significant": bool(p < 0.05),
        }

    def _prediction_trend(self, records: list[DecisionRecord]) -> dict:
        monthly: dict[str, list[bool]] = defaultdict(list)
        for r in records:
            pc = (r.post_outcome or {}).get("prediction_correct")
            if pc is None:
                continue
            month_key = r.ts[:7]
            monthly[month_key].append(bool(pc))
        if not monthly:
            return {"warning": "no prediction_correct data"}
        series = [(k, sum(v) / len(v)) for k, v in sorted(monthly.items())]
        rates = [r for _, r in series]
        trend = "improving" if len(rates) >= 2 and rates[-1] > rates[0] else "flat_or_declining"
        return {
            "monthly": series,
            "trend": trend,
            "latest_rate": rates[-1] if rates else None,
        }

    def _radar(self, records: list[DecisionRecord]) -> dict[str, float]:
        calib = confidence_calibration(records)
        emotion = self._anova_by_tag(records, "trader_emotion")
        session = self._anova_by_session(records)
        trend = self._prediction_trend(records)

        def _bound(x: float) -> float:
            return float(max(0.0, min(100.0, x)))

        calib_score = (
            _bound(((calib.get("r_squared") or 0.0) * 100))
            if calib.get("slope", 0) > 0
            else 0.0
        )
        emotion_p = emotion.get("p_value") or 1.0
        emotion_score = _bound((1.0 - emotion_p) * 100)
        session_p = session.get("p_value") or 1.0
        session_score = _bound((1.0 - session_p) * 100)
        prediction_score = _bound((trend.get("latest_rate") or 0.0) * 100)

        return {
            "confidence_calibration": calib_score,
            "emotion_control": emotion_score,
            "session_fit": session_score,
            "prediction_skill": prediction_score,
        }

    @staticmethod
    def _pnl_R(rec: DecisionRecord) -> float | None:
        if not rec.post_outcome:
            return None
        v = rec.post_outcome.get("pnl_R")
        return float(v) if v is not None else None
