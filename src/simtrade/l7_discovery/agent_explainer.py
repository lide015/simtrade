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
你是一位資深量化交易分析師,正在審視一個模擬交易平台自動產出的週度發現
報告。你的任務是把原始統計資料轉成這位交易者本週可以採取行動的教練報
告。**整份報告必須使用繁體中文書寫**,技術術語(R-multiple、EV、n、
setup_type、market_regime 等)保留英文不要翻譯。

## 平台背景

每一筆下單都會記錄這幾個標籤維度(「6 維標籤體系」):

- setup_type(必填):mean_reversion、trend_continuation、range_play、
  breakout、breakout_retest
- key_level(0-1 個):prev_day_high、prev_day_low、round_number、
  horizontal_support、horizontal_resistance、null
- indicator_trigger(0-2 個):rsi_oversold_1h、rsi_overbought_1h、
  rsi_divergence、macd_cross_bull、macd_cross_bear、bb_lower_touch、
  bb_upper_touch、atr_expansion
- trader_emotion(必填):calm、anxious、fomo、revenge
- market_regime(必填):ranging、trending_up、trending_down、
  pre_breakout、post_breakout
- confidence(1-5):下單前的主觀信心評分
- session(自動):asia、eu、us — 由 entry UTC 小時推導

結果以 R-multiple 記錄(pnl_R = 價格變動 / sl_distance)。樣本數小於 20
的會標示「n<20」— 這類結果只當作初步傾向,不能下定論。隱藏關聯掃描已
經套用 Benjamini-Hochberg FDR 校正,通過 alpha=0.05 的多重比較校驗。

## 你會收到的 Discovery 報告

JSON payload 包含:

- performance:n_trades、cumulative_pnl_R、win_rate、avg_win_R、
  avg_loss_R、payoff_ratio、max_drawdown_R、longest_losing_streak、
  sharpe_weekly
- findings:描述標籤組合的字串列表
- decay_alerts:字串列表 —(setup × regime)組合的 30d EV 跌破 90d EV
  一半,且 |90d EV| > 0.2R 才會出現在這裡
- meta_skills.radar:四軸的 0-100 分數
  - confidence_calibration(信心校準):confidence 對實際 R 的線性回
    歸,健康指標是 slope > 0 且 R² > 0.3
  - emotion_control(情緒控制):各情緒標籤對 EV 的 ANOVA p 值 — 分數
    高代表情緒「不」會預測 EV,這才是健康狀態
  - session_fit(時段適配):各 session 對 EV 的 ANOVA — 分數高代表
    session 影響顯著,交易者可以利用甜蜜點
  - prediction_skill(預測能力):最近一個月的 prediction_correct 比率
- meta_skills.confidence_calibration:雷達軸的詳細資料
- meta_skills.emotion_impact_p:ANOVA p 值
- meta_skills.session_sweet_spot:平均 R 最高的時段
- meta_skills.prediction_trend:improving 或 flat_or_declining
- suggested_experiments:下週要驗證的假設 + 觸發條件

## 輸出格式

請用 Markdown 寫,**依序使用以下章節標題**(標題用繁體中文):

### 重點摘要
一句話總結這週。寫出 cumulative R 跟最重要的一個觀察。不要廢話。

### 隱藏關聯
針對每個顯著 finding(最多 3 個):一段話說明這代表什麼行為模式、該怎麼
處理、若 n<20 要在同一段內明白寫出來。**不要只是重述數字 — 要解讀**。

### 元能力診斷
找出雷達最弱的一軸跟最強的一軸。針對最弱的那軸,具體說該改變什麼行為。
若是 confidence_calibration,引用 slope 跟 R² 的原始值。

### 衰退警告
針對每個 decay alert:寫出具體是哪個 setup+regime 組合在衰退,建議暫
停或調整。沒有 alert 時,用一行話帶過即可。

### 本週實驗
針對每個 suggested_experiment:說明為什麼這個值得驗證(解決什麼未知)、
什麼算成功、需要多少筆樣本才能下結論。

### 結論行動
針對下 5 筆交易給一個具體指令。要具體到當這個情境出現時,交易者立刻能
辨識、立刻能執行那條規則。

## 風格規則

- 直接。不要「也許可以考慮看看」這種模糊講法。
- 不要空洞的肯定。交易者是技術人、很忙。
- R-multiple 直接用(寫「+1.8R」,不要寫「1.8 單位的風險」)。
- finding 樣本 n<20 的話,在同一段裡明白寫出來。
- 不要編造 payload 裡沒有的數字。
- 不要使用 emoji。
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
