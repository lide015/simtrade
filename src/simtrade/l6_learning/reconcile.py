from __future__ import annotations

from datetime import datetime, timedelta, timezone

from simtrade.l1_data.storage import OHLCVStore
from simtrade.l6_learning.decision import DecisionRecord, DecisionStore


class Reconciler:
    """Auto-fill post_outcome 24h after entry (README §5.3).

    Reads the close price at entry_ts + 24h from the OHLCV cache, computes
    pnl_R, MFE, MAE, prediction_correct, and writes once. Locks after.
    """

    def __init__(
        self,
        decisions: DecisionStore,
        ohlcv: OHLCVStore,
        horizon: timedelta = timedelta(hours=24),
        timeframe: str = "1h",
    ):
        self.decisions = decisions
        self.ohlcv = ohlcv
        self.horizon = horizon
        self.timeframe = timeframe

    def run_once(self, now: datetime | None = None) -> int:
        now = now or datetime.now(tz=timezone.utc)
        filled = 0
        for rec in self.decisions.pending_outcomes():
            entry_dt = _parse_iso(rec.ts)
            if now - entry_dt < self.horizon:
                continue
            outcome = self._compute_outcome(rec, entry_dt)
            if outcome is None:
                continue
            self.decisions.fill_outcome(rec.id, outcome)
            filled += 1
        return filled

    def _compute_outcome(self, rec: DecisionRecord, entry_dt: datetime) -> dict | None:
        target_dt = entry_dt + self.horizon
        target_ms = int(target_dt.timestamp() * 1000)

        close_after = self.ohlcv.close_at_or_after(rec.symbol, self.timeframe, target_ms)
        if close_after is None:
            return None

        action = rec.action
        entry = float(action["entry"])
        side = action["side"]
        sl = float(action["sl"])
        tp = float(action.get("tp", entry))
        sl_dist = abs(entry - sl)
        if sl_dist == 0:
            return None

        if side == "long":
            move = close_after - entry
            mfe_move = self._extreme_move(rec.symbol, entry_dt, target_dt, entry, "long")
            mae_move = self._extreme_move(rec.symbol, entry_dt, target_dt, entry, "long_adverse")
            tp_hit = mfe_move >= (tp - entry) if tp > entry else False
            sl_hit = mae_move <= (sl - entry) if sl < entry else False
            prediction_correct = move > 0
        else:
            move = entry - close_after
            mfe_move = self._extreme_move(rec.symbol, entry_dt, target_dt, entry, "short")
            mae_move = self._extreme_move(rec.symbol, entry_dt, target_dt, entry, "short_adverse")
            tp_hit = mfe_move >= (entry - tp) if tp < entry else False
            sl_hit = mae_move <= (entry - sl) if sl > entry else False
            prediction_correct = move > 0

        if sl_hit:
            exit_reason = "sl_hit"
        elif tp_hit:
            exit_reason = "tp_hit"
        else:
            exit_reason = "timeout"

        return {
            "filled_at": datetime.now(tz=timezone.utc).isoformat(),
            "exit_price": close_after,
            "exit_reason": exit_reason,
            "pnl_R": move / sl_dist,
            "pct_change": (close_after - entry) / entry,
            "prediction_correct": prediction_correct,
            "price_after_24h": close_after,
            "max_favorable_excursion_R": mfe_move / sl_dist,
            "max_adverse_excursion_R": mae_move / sl_dist,
        }

    def _extreme_move(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        entry: float,
        kind: str,
    ) -> float:
        start_ms = int(start.timestamp() * 1000)
        end_ms = int(end.timestamp() * 1000)
        candles = self.ohlcv.range_between(symbol, self.timeframe, start_ms, end_ms)
        if not candles:
            return 0.0
        highs = [c[2] for c in candles]
        lows = [c[3] for c in candles]
        if kind == "long":
            return max(highs) - entry
        if kind == "short":
            return entry - min(lows)
        if kind == "long_adverse":
            return min(lows) - entry
        if kind == "short_adverse":
            return entry - max(highs)
        return 0.0


def _parse_iso(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))
