"""LLM-powered explainer for L7 Discovery findings (Gemini).

Uses Google's google-genai SDK with Gemini 2.5 Flash as the default.
Requires GEMINI_API_KEY env var (free key at
https://aistudio.google.com/apikey, no credit card required).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

try:
    from google import genai
    from google.genai import types as genai_types
except ImportError:
    genai = None
    genai_types = None

DEFAULT_MODEL = "gemini-2.5-flash"
DEFAULT_MAX_TOKENS = 8000

ANALYST_SYSTEM_PROMPT = """\
You are a senior quantitative trading analyst reviewing automated weekly
discovery reports from a simulated trading platform. Your job is to turn
raw statistical findings into a coaching report the trader can act on
this week.

## Platform context

Every decision is recorded with these tag dimensions (the "6 dim tag
system"):

- setup_type (required): mean_reversion, trend_continuation, range_play,
  breakout, breakout_retest
- key_level (0-1): prev_day_high, prev_day_low, round_number,
  horizontal_support, horizontal_resistance, null
- indicator_trigger (0-2): rsi_oversold_1h, rsi_overbought_1h,
  rsi_divergence, macd_cross_bull, macd_cross_bear, bb_lower_touch,
  bb_upper_touch, atr_expansion
- trader_emotion (required): calm, anxious, fomo, revenge
- market_regime (required): ranging, trending_up, trending_down,
  pre_breakout, post_breakout
- confidence (1-5): trader's pre-trade self-rating
- session (auto): asia, eu, us — derived from entry UTC hour

Outcomes are recorded as R-multiples (pnl_R = move / sl_distance). Sample
sizes below 20 are flagged "n<20" — take those findings as suggestive,
not conclusive. The hidden correlation scanner uses Benjamini-Hochberg
FDR correction, so flagged findings have already survived multiple-
testing correction at alpha=0.05.

## Discovery report you will receive

A JSON payload with:

- performance: n_trades, cumulative_pnl_R, win_rate, avg_win_R,
  avg_loss_R, payoff_ratio, max_drawdown_R, longest_losing_streak,
  sharpe_weekly
- findings: list of strings describing tag combinations
- decay_alerts: list of strings — (setup x regime) combos whose 30d EV
  dropped below 50% of 90d EV, where |90d EV| > 0.2R
- meta_skills.radar: 0-100 scores across four axes:
  - confidence_calibration (linear fit of confidence vs realized R,
    healthy when slope > 0 and R^2 > 0.3)
  - emotion_control (ANOVA p-value across emotion tags — high score
    means emotion does NOT predict EV, which is the healthy direction)
  - session_fit (ANOVA across sessions — high score means session
    matters and trader can exploit a sweet spot)
  - prediction_skill (latest month's prediction_correct rate)
- meta_skills.confidence_calibration: detail dict for the radar axis
- meta_skills.emotion_impact_p: ANOVA p-value
- meta_skills.session_sweet_spot: which session has highest mean R
- meta_skills.prediction_trend: improving | flat_or_declining
- suggested_experiments: hypothesis + conditions to test next week

## Output format

Produce a Markdown report with these exact section headers in this order:

### Headline
One sentence summarizing the week. State the cumulative R and one most
important insight. No filler.

### Top hidden correlations
For each significant finding (max 3): one paragraph explaining what
behavior pattern this points to, what to do about it, and a sample-size
caveat if n<20. Do not just restate the numbers — interpret them.

### Meta-skill diagnosis
Identify the WEAKEST axis on the radar and the STRONGEST. For the
weakest, state concretely what behavior needs to change. Use the raw
calibration slope and R^2 if relevant.

### Decay warnings
For each decay alert: state the specific setup+regime combination that
is breaking down, suggest whether to pause or adjust. If no alerts,
write a single short line acknowledging that.

### This week's experiments
For each suggested experiment: explain WHY this matters
(what unknown does it resolve?), what a clear success criterion looks
like, and the minimum sample size that would make it actionable.

### Bottom line
One concrete instruction for the next 5 trades. Must be specific enough
that the trader can recognize the condition when they see it and execute
the rule.

## Style rules

- Direct. No hedging like "it might be worth considering."
- No empty validation. The trader is technical and busy.
- Use R-multiples natively (write "+1.8R", not "1.8 units of risk").
- If a finding has n<20, say so plainly inside the same paragraph that
  cites it.
- Do not invent numbers not present in the payload.
- Do not use emoji.
"""


@dataclass
class ExplainerResult:
    markdown: str
    model: str
    input_tokens: int
    output_tokens: int
    cached_tokens: int
    thoughts_tokens: int

    def usage_summary(self) -> str:
        return (
            f"model={self.model}  in={self.input_tokens}  "
            f"out={self.output_tokens}  cached={self.cached_tokens}  "
            f"thoughts={self.thoughts_tokens}"
        )


class L7AgentExplainer:
    """Calls Gemini to explain a weekly Discovery report.

    Uses adaptive thinking (thinking_budget=-1 lets the model decide
    how much to think). Streams to avoid HTTP timeouts on long outputs.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key: str | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ):
        if genai is None:
            raise RuntimeError(
                "google-genai SDK is required for L7AgentExplainer; "
                "install with `pip install -e .[llm]`"
            )
        self.model = model
        self.max_tokens = max_tokens
        self.client = (
            genai.Client(api_key=api_key) if api_key else genai.Client()
        )

    def explain(self, discovery_report: dict[str, Any]) -> ExplainerResult:
        payload = json.dumps(discovery_report, indent=2, default=str)
        prompt = (
            "Here is this week's Discovery report. Produce the coaching "
            "report per the system instruction format.\n\n"
            "```json\n" + payload + "\n```"
        )
        config = genai_types.GenerateContentConfig(
            system_instruction=ANALYST_SYSTEM_PROMPT,
            max_output_tokens=self.max_tokens,
            thinking_config=genai_types.ThinkingConfig(thinking_budget=-1),
        )

        chunks: list[str] = []
        final_chunk = None
        for chunk in self.client.models.generate_content_stream(
            model=self.model,
            contents=prompt,
            config=config,
        ):
            if chunk.text:
                chunks.append(chunk.text)
            final_chunk = chunk

        markdown = "".join(chunks)
        usage = final_chunk.usage_metadata if final_chunk else None
        return ExplainerResult(
            markdown=markdown,
            model=self.model,
            input_tokens=getattr(usage, "prompt_token_count", 0) or 0,
            output_tokens=getattr(usage, "candidates_token_count", 0) or 0,
            cached_tokens=getattr(usage, "cached_content_token_count", 0) or 0,
            thoughts_tokens=getattr(usage, "thoughts_token_count", 0) or 0,
        )
