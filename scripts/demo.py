"""End-to-end smoke test using synthetic data — no exchange connection needed.

Generates a year of OHLCV bars, records 60 decisions with varied tag
combinations, reconciles outcomes, then runs the L7 report.
"""
from __future__ import annotations

import math
import random
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from simtrade.l2_indicators import compute_indicators_multi_tf, detect_regime  # noqa: E402
from simtrade.l5_feedback import compute_performance  # noqa: E402
from simtrade.l6_learning import (  # noqa: E402
    DecisionRecord,
    attribution_by_tag,
    attribution_cross,
)
from simtrade.platform import boot, weekly_discovery_report  # noqa: E402

SYMBOL = "BTC/USDT"
TIMEFRAME = "1h"
SETUPS = ["mean_reversion", "trend_continuation", "range_play", "breakout"]
REGIMES = ["ranging", "trending_up", "trending_down", "pre_breakout"]
EMOTIONS = ["calm", "anxious", "fomo"]
KEY_LEVELS = [None, "prev_day_high", "round_number"]


def _generate_candles(n: int, start: datetime, base_price: float = 70_000.0):
    rng = random.Random(42)
    candles = []
    price = base_price
    ts = start
    for _ in range(n):
        drift = rng.gauss(0, 0.003)
        price *= 1 + drift
        o = price
        h = price * (1 + abs(rng.gauss(0, 0.002)))
        l = price * (1 - abs(rng.gauss(0, 0.002)))
        c = price * (1 + rng.gauss(0, 0.001))
        v = rng.uniform(100, 1000)
        candles.append((int(ts.timestamp() * 1000), o, h, l, c, v))
        price = c
        ts += timedelta(hours=1)
    return candles


def main(db_path: str | None = None) -> int:
    db = Path(db_path) if db_path else Path("data/demo.db")
    if db.exists():
        db.unlink()
    db.parent.mkdir(parents=True, exist_ok=True)
    ctx = boot(db_path=str(db), with_market=False)

    start = datetime.now(tz=timezone.utc) - timedelta(days=120)
    candles = _generate_candles(120 * 24, start)
    ctx.ohlcv.upsert(SYMBOL, TIMEFRAME, candles)
    print(f"seeded {len(candles)} hourly bars")

    rng = random.Random(7)
    n_decisions = 60
    for i in range(n_decisions):
        entry_dt = start + timedelta(hours=i * 24 + 6)
        recent = ctx.ohlcv.fetch(
            SYMBOL,
            TIMEFRAME,
            since_ms=int((entry_dt - timedelta(days=5)).timestamp() * 1000),
            limit=120,
        )
        if len(recent) < 30:
            continue

        indicators = compute_indicators_multi_tf({TIMEFRAME: recent})
        regime = detect_regime(recent)
        last_close = recent[-1][4]

        setup = rng.choice(SETUPS)
        emotion = rng.choices(EMOTIONS, weights=[0.6, 0.3, 0.1])[0]
        side = "long" if rng.random() > 0.4 else "short"

        if side == "long":
            entry = last_close
            sl = entry * 0.99
            tp = entry * 1.015
        else:
            entry = last_close
            sl = entry * 1.01
            tp = entry * 0.985

        market_snapshot = {
            "price": last_close,
            "indicators": indicators,
            "session": "asia",
            "regime": regime,
        }
        trader_state = {
            "prediction_text": f"{setup} on {side} from {entry:.0f}",
            "confidence": rng.randint(2, 5),
            "reasoning_tags": {
                "setup_type": setup,
                "key_level": rng.choice(KEY_LEVELS),
                "indicator_trigger": (
                    ["rsi_oversold_1h"] if setup == "mean_reversion" and side == "long" else []
                ),
                "trader_emotion": emotion,
                "market_regime": regime,
            },
        }
        action = {
            "side": side,
            "size": 0.05,
            "entry": entry,
            "sl": sl,
            "tp": tp,
            "risk_amount": 100.0,
            "sl_distance_R": 1.0,
            "tp_distance_R": abs(tp - entry) / abs(entry - sl),
        }
        rec = DecisionRecord.new(
            symbol=SYMBOL,
            market_snapshot=market_snapshot,
            trader_state=trader_state,
            action=action,
            ts=entry_dt,
        )
        ctx.decisions.insert(rec)

    print(f"recorded {n_decisions} decisions")

    fake_now = start + timedelta(days=120)
    n_filled = ctx.reconciler.run_once(now=fake_now)
    print(f"reconciled {n_filled} outcomes")

    perf = compute_performance(ctx.conn)
    print("\n[performance]")
    print(f"  trades={perf.n_trades}  cum_R={perf.cumulative_pnl_R:+.2f}  "
          f"win_rate={perf.win_rate:.0%}  max_dd_R={perf.max_drawdown_R:.2f}")
    print(f"  sharpe(weekly)={perf.sharpe_weekly}")

    print("\n[attribution by setup_type]")
    by_setup = attribution_by_tag(ctx.decisions.completed(), dim="setup_type")
    for k, v in sorted(by_setup.items(), key=lambda kv: -(kv[1]["EV_R"] or 0)):
        print(f"  {k:24s} n={v['n']:3d}  EV={v['EV_R']:+.2f}R  "
              f"win={v['win_rate']:.0%}  warn={v['warning']}")

    print("\n[L7 weekly discovery]")
    report = weekly_discovery_report(ctx)
    print(f"  findings: {len(report['findings'])}")
    for f in report["findings"][:5]:
        print(f"    - {f}")
    print(f"  decay_alerts: {len(report['decay_alerts'])}")
    for a in report["decay_alerts"]:
        print(f"    - {a}")
    print(f"  meta radar: {report['meta_skills']['radar']}")
    print(f"  suggested experiments: {len(report['suggested_experiments'])}")
    for s in report["suggested_experiments"]:
        print(f"    - {s['hypothesis']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
