import math

from simtrade.l2_indicators import (
    compute_indicators,
    compute_indicators_multi_tf,
    detect_regime,
)


def _trend_candles(n: int, start: float, step: float):
    return [
        (i * 60_000, start + i * step, start + i * step + 1, start + i * step - 1,
         start + i * step + 0.5, 100.0)
        for i in range(n)
    ]


def test_compute_indicators_basic_shape():
    candles = _trend_candles(40, 100.0, 1.0)
    ind = compute_indicators(candles)
    assert ind["rsi"] is not None
    assert ind["atr"] is not None
    assert ind["macd_hist"] is not None
    assert ind["bb_upper"] > ind["bb_mid"] > ind["bb_lower"]


def test_compute_indicators_handles_empty():
    assert compute_indicators([]) == {}


def test_compute_indicators_multi_tf_dict_shape():
    tf = {"1h": _trend_candles(40, 100.0, 1.0), "4h": _trend_candles(40, 100.0, 2.0)}
    out = compute_indicators_multi_tf(tf)
    assert set(out.keys()) == {"1h", "4h"}


def test_detect_regime_uptrend():
    candles = _trend_candles(60, 100.0, 1.0)
    assert detect_regime(candles) in {"trending_up", "post_breakout"}


def test_detect_regime_too_few_bars_defaults_ranging():
    assert detect_regime([(0, 1, 1, 1, 1, 1)]) == "ranging"
