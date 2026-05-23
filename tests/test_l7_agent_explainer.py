"""Tests for L7AgentExplainer (Gemini variant).

Mocks google.genai so the tests don't make real API calls. Goal is to
verify the wiring: model, system_instruction, adaptive thinking config,
streaming, and usage extraction.
"""
from __future__ import annotations

import sys
import types as pytypes
from dataclasses import dataclass
from unittest.mock import MagicMock

import pytest

# Stub google.genai module tree so `from google import genai` works
# even when google-genai isn't installed in the test environment.
if "google.genai" not in sys.modules:
    google_mod = sys.modules.setdefault("google", pytypes.ModuleType("google"))
    genai_mod = pytypes.ModuleType("google.genai")
    types_mod = pytypes.ModuleType("google.genai.types")
    types_mod.GenerateContentConfig = MagicMock()
    types_mod.ThinkingConfig = MagicMock()
    genai_mod.Client = MagicMock()
    genai_mod.types = types_mod
    google_mod.genai = genai_mod
    sys.modules["google.genai"] = genai_mod
    sys.modules["google.genai.types"] = types_mod

import simtrade.l7_discovery.agent_explainer as ae


@dataclass
class _FakeUsage:
    prompt_token_count: int = 1200
    candidates_token_count: int = 500
    cached_content_token_count: int = 0
    thoughts_token_count: int = 800


@dataclass
class _FakeChunk:
    text: str = ""
    usage_metadata: _FakeUsage | None = None


def _make_fake_client(markdown: str) -> MagicMock:
    """Return a client whose stream yields two chunks; last carries usage."""
    client = MagicMock()
    mid = len(markdown) // 2
    chunks = [
        _FakeChunk(text=markdown[:mid]),
        _FakeChunk(text=markdown[mid:], usage_metadata=_FakeUsage()),
    ]
    client.models.generate_content_stream.return_value = iter(chunks)
    return client


SAMPLE_REPORT = {
    "performance": {
        "n_trades": 60,
        "cumulative_pnl_R": 11.95,
        "win_rate": 0.58,
        "max_drawdown_R": 11.08,
        "sharpe_weekly": 1.82,
    },
    "findings": [
        "setup_type=mean_reversion & market_regime=ranging -> win 71% (n=14) "
        "vs base 52%, p=0.003 (BH=0.04)"
    ],
    "decay_alerts": [],
    "meta_skills": {
        "radar": {
            "confidence_calibration": 0.0,
            "emotion_control": 80.0,
            "session_fit": 30.0,
            "prediction_skill": 60.0,
        },
        "session_sweet_spot": "asia",
        "prediction_trend": "flat_or_declining",
    },
    "suggested_experiments": [],
}


def test_explainer_invokes_client_with_correct_model(monkeypatch):
    fake_client = _make_fake_client("### Headline\nUp +11.95R...")
    fake_genai = MagicMock()
    fake_genai.Client.return_value = fake_client
    monkeypatch.setattr(ae, "genai", fake_genai)
    monkeypatch.setattr(ae, "genai_types", MagicMock())

    explainer = ae.L7AgentExplainer(api_key="test")
    result = explainer.explain(SAMPLE_REPORT)

    assert result.markdown.startswith("### Headline")
    assert result.input_tokens == 1200
    assert result.output_tokens == 500
    assert result.thoughts_tokens == 800
    fake_client.models.generate_content_stream.assert_called_once()
    kwargs = fake_client.models.generate_content_stream.call_args.kwargs
    assert kwargs["model"] == "gemini-2.5-flash"


def test_explainer_passes_payload_in_prompt(monkeypatch):
    fake_client = _make_fake_client("ok")
    fake_genai = MagicMock()
    fake_genai.Client.return_value = fake_client
    monkeypatch.setattr(ae, "genai", fake_genai)
    monkeypatch.setattr(ae, "genai_types", MagicMock())

    explainer = ae.L7AgentExplainer(api_key="test")
    explainer.explain(SAMPLE_REPORT)

    kwargs = fake_client.models.generate_content_stream.call_args.kwargs
    contents = kwargs["contents"]
    assert "mean_reversion" in contents
    assert "11.95" in contents


def test_explainer_configures_adaptive_thinking(monkeypatch):
    fake_client = _make_fake_client("ok")
    fake_genai = MagicMock()
    fake_genai.Client.return_value = fake_client
    fake_types = MagicMock()
    monkeypatch.setattr(ae, "genai", fake_genai)
    monkeypatch.setattr(ae, "genai_types", fake_types)

    explainer = ae.L7AgentExplainer(api_key="test")
    explainer.explain(SAMPLE_REPORT)

    fake_types.ThinkingConfig.assert_called_once_with(thinking_budget=-1)
    fake_types.GenerateContentConfig.assert_called_once()
    cfg_kwargs = fake_types.GenerateContentConfig.call_args.kwargs
    assert "system_instruction" in cfg_kwargs
    assert cfg_kwargs["max_output_tokens"] == 8000


def test_explainer_raises_when_genai_missing(monkeypatch):
    monkeypatch.setattr(ae, "genai", None)
    with pytest.raises(RuntimeError, match="google-genai SDK is required"):
        ae.L7AgentExplainer()


def test_usage_summary_format():
    result = ae.ExplainerResult(
        markdown="x",
        model="gemini-2.5-flash",
        input_tokens=100,
        output_tokens=200,
        cached_tokens=0,
        thoughts_tokens=500,
    )
    s = result.usage_summary()
    assert "gemini-2.5-flash" in s
    assert "in=100" in s
    assert "out=200" in s
    assert "thoughts=500" in s
