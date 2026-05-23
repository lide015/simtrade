from __future__ import annotations

from datetime import datetime, timezone

SETUP_TYPES = {
    "mean_reversion",
    "trend_continuation",
    "range_play",
    "breakout",
    "breakout_retest",
}

KEY_LEVELS = {
    None,
    "prev_day_high",
    "prev_day_low",
    "round_number",
    "horizontal_support",
    "horizontal_resistance",
}

INDICATOR_TRIGGERS = {
    "rsi_oversold_1h",
    "rsi_overbought_1h",
    "rsi_divergence",
    "macd_cross_bull",
    "macd_cross_bear",
    "bb_lower_touch",
    "bb_upper_touch",
    "atr_expansion",
}

TRADER_EMOTIONS = {"calm", "anxious", "fomo", "revenge"}

MARKET_REGIMES = {
    "ranging",
    "trending_up",
    "trending_down",
    "pre_breakout",
    "post_breakout",
}

CONFIDENCE_RANGE = range(1, 6)

FORBIDDEN_RESULT_TAGS = {"bullish", "bearish", "should_pump", "probably_up", "probably_down"}


def validate_reasoning_tags(tags: dict) -> list[str]:
    """Return a list of human-readable issues. Empty list = valid."""
    issues: list[str] = []

    setup = tags.get("setup_type")
    if setup not in SETUP_TYPES:
        issues.append(f"setup_type must be one of {sorted(SETUP_TYPES)}, got {setup!r}")

    key_level = tags.get("key_level")
    if key_level not in KEY_LEVELS:
        issues.append(f"key_level must be one of {sorted(x for x in KEY_LEVELS if x)} or null")

    triggers = tags.get("indicator_trigger", []) or []
    if not isinstance(triggers, list):
        issues.append("indicator_trigger must be a list")
    else:
        if len(triggers) > 2:
            issues.append("indicator_trigger: max 2 entries (rule §3.2.1)")
        for t in triggers:
            if t not in INDICATOR_TRIGGERS:
                issues.append(f"unknown indicator_trigger: {t!r}")

    emotion = tags.get("trader_emotion")
    if emotion not in TRADER_EMOTIONS:
        issues.append(f"trader_emotion must be one of {sorted(TRADER_EMOTIONS)}")

    regime = tags.get("market_regime")
    if regime not in MARKET_REGIMES:
        issues.append(f"market_regime must be one of {sorted(MARKET_REGIMES)}")

    flat = set()
    for v in tags.values():
        if isinstance(v, str):
            flat.add(v)
        elif isinstance(v, list):
            flat.update(v)
    bad = flat & FORBIDDEN_RESULT_TAGS
    if bad:
        issues.append(
            f"forbidden result-prediction tags (rule §3.2.2 #4): {sorted(bad)}"
        )
    return issues


def derive_descriptive_tags(
    indicators_by_tf: dict[str, dict[str, float | None]],
    funding_rate: float | None = None,
    ts: datetime | None = None,
) -> dict:
    """Auto-extract descriptive tags from market state (README §3.2.2 rule 2).

    These are *describing the market*, not *explaining your decision*.
    """
    out: dict = {}
    ts = ts or datetime.now(tz=timezone.utc)
    hour_utc = ts.hour
    if 0 <= hour_utc < 8:
        out["session"] = "asia"
    elif 8 <= hour_utc < 16:
        out["session"] = "eu"
    else:
        out["session"] = "us"

    if funding_rate is not None:
        if funding_rate > 0.0005:
            out["funding"] = "high"
        elif funding_rate < 0:
            out["funding"] = "negative"
        else:
            out["funding"] = "neutral"

    ind_1h = indicators_by_tf.get("1h", {}) or {}
    rsi_1h = ind_1h.get("rsi")
    derived_triggers: list[str] = []
    if rsi_1h is not None:
        if rsi_1h < 30:
            derived_triggers.append("rsi_oversold_1h")
        elif rsi_1h > 70:
            derived_triggers.append("rsi_overbought_1h")

    close = ind_1h.get("close")
    bb_lo = ind_1h.get("bb_lower")
    bb_up = ind_1h.get("bb_upper")
    if close is not None and bb_lo is not None and close <= bb_lo:
        derived_triggers.append("bb_lower_touch")
    if close is not None and bb_up is not None and close >= bb_up:
        derived_triggers.append("bb_upper_touch")

    if derived_triggers:
        out["derived_indicator_trigger"] = derived_triggers
    return out
