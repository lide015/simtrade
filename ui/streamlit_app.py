"""GOLD TRADER LITE · 加密貨幣決策實驗室

Runs locally (streamlit run ui/streamlit_app.py) and on Streamlit Community
Cloud. On Cloud, set GEMINI_API_KEY in App Secrets; locally set the env var
or run python scripts/demo.py first then `streamlit run`.
"""
from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from simtrade.l5_feedback import compute_performance  # noqa: E402
from simtrade.l6_learning import attribution_by_tag, attribution_cross  # noqa: E402
from simtrade.platform import boot, weekly_discovery_report  # noqa: E402

st.set_page_config(
    page_title="GOLD TRADER LITE · 加密貨幣決策實驗室",
    page_icon=":crown:",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Gold + Black palette ──────────────────────────────────────────────────
GOLD = "#d4af37"           # rich gold (primary)
GOLD_BRIGHT = "#ffd700"    # bright gold (highlights)
GOLD_DIM = "#8b7355"       # muted gold (borders, subtle text)
BG = "#0a0a0a"             # near-black
CARD = "#141414"           # slightly lifted card
CARD_HOVER = "#1c1c1c"
TEXT = "#f5e6c8"           # warm off-white
TEXT_DIM = "#9b8b6e"       # muted gold-gray
POS = "#d4af37"            # gold = up
NEG = "#cc3333"            # muted red = down
BORDER = "#2a2418"         # dark gold border

st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@600;700&family=Inter:wght@400;500;600&display=swap');

    .block-container {{ padding-top: 1rem; padding-bottom: 3rem; max-width: 1400px; }}
    [data-testid="stHeader"] {{ background: transparent; }}
    body, .stApp {{ font-family: 'Inter', sans-serif; }}

    /* Sidebar */
    [data-testid="stSidebar"] {{
        background-color: {CARD};
        border-right: 1px solid {BORDER};
    }}

    /* Headings */
    h1, h2, h3, h4 {{ font-weight: 600; letter-spacing: -0.01em; color: {TEXT}; }}

    /* Metric cards */
    div[data-testid="stMetric"] {{
        background: linear-gradient(135deg, {CARD} 0%, rgba(20,20,20,0.5) 100%);
        border: 1px solid {BORDER};
        border-radius: 12px;
        padding: 1rem 1.25rem;
        transition: all 0.2s;
    }}
    div[data-testid="stMetric"]:hover {{
        border-color: {GOLD};
        box-shadow: 0 4px 12px rgba(212,175,55,0.15);
    }}
    div[data-testid="stMetricValue"] {{
        font-size: 1.75rem; font-weight: 700; color: {GOLD};
        font-variant-numeric: tabular-nums;
    }}
    div[data-testid="stMetricDelta"] {{ font-size: 0.8rem; opacity: 0.7; }}
    div[data-testid="stMetricLabel"] {{
        font-size: 0.75rem; text-transform: uppercase;
        letter-spacing: 0.08em; color: {TEXT_DIM}; font-weight: 600;
    }}

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 8px; border-bottom: 1px solid {BORDER};
    }}
    .stTabs [data-baseweb="tab"] {{
        padding: 12px 18px; border-radius: 8px 8px 0 0;
        font-weight: 500; font-size: 0.95rem; color: {TEXT_DIM};
    }}
    .stTabs [aria-selected="true"] {{
        background: linear-gradient(180deg, rgba(212,175,55,0.12) 0%, transparent 100%);
        color: {GOLD} !important;
        border-bottom: 2px solid {GOLD};
    }}

    /* Buttons */
    .stButton > button {{
        border-radius: 8px; font-weight: 600; transition: all 0.15s;
        border: 1px solid {GOLD_DIM};
        color: {TEXT};
        background: transparent;
    }}
    .stButton > button:hover {{ border-color: {GOLD}; color: {GOLD}; }}
    .stButton > button[kind="primary"] {{
        background: linear-gradient(135deg, {GOLD} 0%, {GOLD_BRIGHT} 100%);
        border: none; color: {BG}; font-weight: 700;
        box-shadow: 0 2px 12px rgba(212,175,55,0.35);
    }}
    .stButton > button[kind="primary"]:hover {{
        transform: translateY(-1px);
        box-shadow: 0 4px 18px rgba(212,175,55,0.5);
    }}

    /* Selectbox, text input */
    .stSelectbox [data-baseweb="select"], .stTextInput input {{
        background-color: {CARD} !important;
        border: 1px solid {BORDER} !important;
        color: {TEXT} !important;
    }}

    /* Hero */
    .hero {{
        padding: 1.5rem 0 2rem; margin-bottom: 1.5rem;
        border-bottom: 1px solid {BORDER};
        display: flex; align-items: center; gap: 1.5rem;
    }}
    .hero-title {{
        font-family: 'Cinzel', serif;
        font-size: 2.5rem; font-weight: 700; margin: 0;
        letter-spacing: 0.05em;
        background: linear-gradient(135deg, {GOLD_BRIGHT} 0%, {GOLD} 60%, {GOLD_DIM} 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-shadow: 0 0 30px rgba(212,175,55,0.3);
    }}
    .hero-tagline {{
        margin: 0.25rem 0 0; color: {TEXT_DIM}; font-size: 0.9rem;
        letter-spacing: 0.15em; text-transform: uppercase;
    }}
    .hero-subtitle {{
        margin: 0.5rem 0 0; color: {TEXT_DIM}; font-size: 0.875rem;
    }}

    /* Ticker strip */
    .ticker-strip {{
        display: flex; gap: 0.5rem; padding: 0.75rem; margin-bottom: 1.5rem;
        background: {CARD}; border: 1px solid {BORDER}; border-radius: 12px;
        overflow-x: auto;
    }}
    .ticker {{
        flex: 1; min-width: 140px; padding: 0.5rem 0.85rem;
        border-left: 3px solid {GOLD};
    }}
    .ticker-symbol {{
        font-size: 0.7rem; color: {TEXT_DIM}; text-transform: uppercase;
        letter-spacing: 0.08em; font-weight: 600;
    }}
    .ticker-price {{
        font-size: 1.15rem; font-weight: 700; color: {TEXT};
        margin: 0.15rem 0; font-variant-numeric: tabular-nums;
    }}
    .ticker-change {{ font-size: 0.85rem; font-weight: 600; font-variant-numeric: tabular-nums; }}
    .pos {{ color: {POS}; }}
    .neg {{ color: {NEG}; }}

    /* Category pill */
    .category-pill {{
        display: inline-block; padding: 4px 12px; border-radius: 20px;
        background: rgba(212,175,55,0.1); color: {GOLD};
        font-size: 0.75rem; font-weight: 600; letter-spacing: 0.05em;
        border: 1px solid {GOLD_DIM}; text-transform: uppercase;
    }}

    /* Footer */
    .footer {{
        margin-top: 4rem; padding-top: 1.5rem;
        border-top: 1px solid {BORDER};
        text-align: center; color: {TEXT_DIM}; font-size: 0.8rem;
    }}
    .footer a {{ color: {GOLD}; text-decoration: none; }}
    .footer a:hover {{ color: {GOLD_BRIGHT}; }}
    </style>
    """,
    unsafe_allow_html=True,
)

try:
    if "GEMINI_API_KEY" in st.secrets and not os.environ.get("GEMINI_API_KEY"):
        os.environ["GEMINI_API_KEY"] = st.secrets["GEMINI_API_KEY"]
except Exception:
    pass


# ── Market categories ─────────────────────────────────────────────────────
MARKETS = {
    "現貨主流": {
        "symbols": ["BTC/USDT", "ETH/USDT", "BNB/USDT"],
        "tv_prefix": "BINANCE",
        "tv_suffix": "",
        "type": "spot",
    },
    "現貨代幣": {
        "symbols": ["SOL/USDT", "XRP/USDT", "ADA/USDT", "AVAX/USDT", "DOGE/USDT"],
        "tv_prefix": "BINANCE",
        "tv_suffix": "",
        "type": "spot",
    },
    "合約主流": {
        "symbols": ["BTC/USDT:USDT", "ETH/USDT:USDT", "BNB/USDT:USDT"],
        "tv_prefix": "BINANCE",
        "tv_suffix": ".P",
        "type": "future",
    },
    "合約代幣": {
        "symbols": ["SOL/USDT:USDT", "XRP/USDT:USDT", "ADA/USDT:USDT", "AVAX/USDT:USDT", "DOGE/USDT:USDT"],
        "tv_prefix": "BINANCE",
        "tv_suffix": ".P",
        "type": "future",
    },
}


def to_tv_symbol(ccxt_sym: str, category: str) -> str:
    """Convert ccxt symbol (BTC/USDT or BTC/USDT:USDT) to TradingView format."""
    cat = MARKETS[category]
    base, quote = ccxt_sym.split("/")[0], "USDT"
    return f"{cat['tv_prefix']}:{base}{quote}{cat['tv_suffix']}"


# ── Live market data (cached) ─────────────────────────────────────────────
@st.cache_data(ttl=60, show_spinner=False)
def fetch_tickers(symbols_tuple: tuple, market_type: str) -> dict:
    try:
        import ccxt

        opts = {"enableRateLimit": True}
        if market_type == "future":
            opts["options"] = {"defaultType": "future"}
        ex = ccxt.binance(opts)
        out = {}
        for s in symbols_tuple:
            try:
                t = ex.fetch_ticker(s)
                out[s] = {
                    "price": t.get("last"),
                    "change_pct": t.get("percentage"),
                    "volume": t.get("quoteVolume"),
                    "high": t.get("high"),
                    "low": t.get("low"),
                }
            except Exception:
                continue
        return out
    except Exception:
        return {}


@st.cache_data(ttl=300, show_spinner=False)
def fetch_funding_rate(symbol: str) -> float | None:
    try:
        import ccxt

        ex = ccxt.binance({
            "enableRateLimit": True,
            "options": {"defaultType": "future"},
        })
        fr = ex.fetch_funding_rate(symbol)
        return fr.get("fundingRate")
    except Exception:
        return None


# ── Sidebar — market category ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 設定")
    category = st.selectbox(
        "市場類型",
        list(MARKETS.keys()),
        index=0,
        help="現貨 = 直接買賣;合約 = 永續期貨,可槓桿+做空,有資金費率。",
    )
    db_path = st.text_input("資料庫路徑", value="data/simtrade.db")
    st.divider()
    st.markdown("### 關於")
    st.caption(
        "**GOLD TRADER LITE** 是 Tier 4 模擬交易平台,核心理念是「資料找你」"
        "而不是「你查資料」 — 系統主動發現你沒注意到的交易模式,並用 AI "
        "教練給你週度建議。"
    )
    st.markdown(
        "- [架構文件](https://github.com/lide015/Simulated-trading)\n"
        "- [Gemini API key](https://aistudio.google.com/apikey)"
    )

cat_info = MARKETS[category]


# ── Hero header with optional logo ────────────────────────────────────────
logo_path = Path(__file__).resolve().parent / "assets" / "logo.png"
hero_cols = st.columns([1, 6]) if logo_path.exists() else (None, None)

if logo_path.exists():
    with hero_cols[0]:
        st.image(str(logo_path), width=120)
    title_col = hero_cols[1]
else:
    title_col = st.container()

with title_col:
    st.markdown(
        f"""
        <div class="hero" style="border-bottom: none; padding-bottom: 0.5rem;">
          <div>
            <h1 class="hero-title">GOLD TRADER LITE</h1>
            <p class="hero-tagline">PRECISION · POWER · WIN</p>
            <p class="hero-subtitle">加密貨幣決策實驗室 · 主動發現你的交易盲點 · L1-L7 七層架構</p>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown(
    f'<div style="margin: 0 0 1rem;"><span class="category-pill">當前市場 · {category}</span></div>',
    unsafe_allow_html=True,
)


# ── Ticker strip (current category) ───────────────────────────────────────
ticker_symbols = tuple(cat_info["symbols"])
tickers = fetch_tickers(ticker_symbols, cat_info["type"])

if tickers:
    parts = ['<div class="ticker-strip">']
    for sym in ticker_symbols:
        if sym not in tickers:
            continue
        t = tickers[sym]
        price = t.get("price") or 0
        change = t.get("change_pct") or 0
        cls = "pos" if change >= 0 else "neg"
        arrow = "▲" if change >= 0 else "▼"
        if price >= 100:
            price_s = f"${price:,.2f}"
        elif price >= 1:
            price_s = f"${price:.3f}"
        else:
            price_s = f"${price:.5f}"
        display_sym = sym.split(":")[0]
        parts.append(
            f'<div class="ticker"><div class="ticker-symbol">{display_sym}</div>'
            f'<div class="ticker-price">{price_s}</div>'
            f'<div class="ticker-change {cls}">{arrow} {abs(change):.2f}%</div></div>'
        )
    parts.append("</div>")
    st.markdown("".join(parts), unsafe_allow_html=True)
else:
    st.info("即時行情暫時無法取得(Binance API 可能對部署區域限流)。")


# ── Boot platform context ─────────────────────────────────────────────────
db_file = Path(db_path)
if not db_file.exists():
    st.warning(f"找不到資料庫 `{db_path}`")
    st.write("首次啟動,請按下方按鈕產生 60 筆合成決策資料。")
    if st.button("產生範例資料", type="primary"):
        db_file.parent.mkdir(parents=True, exist_ok=True)
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from scripts.demo import main as demo_main  # noqa: E402

        with st.spinner("正在產生 60 筆合成交易(約 30 秒)..."):
            demo_main(db_path=str(db_file))
        st.success("完成,正在重新載入...")
        st.rerun()
    st.stop()

ctx = boot(db_path=db_path, with_market=False)
all_records = ctx.decisions.all()
completed = ctx.decisions.completed()


def _records_to_df(records) -> pd.DataFrame:
    rows = []
    for r in records:
        outcome = r.post_outcome or {}
        action = r.action
        tags = r.trader_state.get("reasoning_tags", {})
        rows.append({
            "id": r.id[:8],
            "ts": pd.to_datetime(r.ts),
            "symbol": r.symbol,
            "side": action.get("side"),
            "entry": action.get("entry"),
            "sl": action.get("sl"),
            "tp": action.get("tp"),
            "pnl_R": outcome.get("pnl_R"),
            "exit_reason": outcome.get("exit_reason"),
            "setup_type": tags.get("setup_type"),
            "market_regime": tags.get("market_regime"),
            "trader_emotion": tags.get("trader_emotion"),
            "confidence": r.trader_state.get("confidence"),
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("ts").reset_index(drop=True)
        df["cum_R"] = df["pnl_R"].fillna(0).cumsum()
    return df


df = _records_to_df(all_records)
df_completed = df[df["pnl_R"].notna()].copy() if not df.empty else df
perf = compute_performance(ctx.conn)


# ── Tabs ──────────────────────────────────────────────────────────────────
tab_market, tab_compare, tab_perf, tab_patterns, tab_discovery, tab_coach, tab_trades = st.tabs(
    ["📈 即時行情", "⚖️ 多標的比較", "📊 績效",
     "📑 標籤歸因", "🔍 主動發現", "🤖 AI 教練", "📝 交易紀錄"]
)


# ── Market: TradingView widget ────────────────────────────────────────────
with tab_market:
    c1, c2 = st.columns([2, 1])
    with c1:
        sym = st.selectbox(
            "標的", cat_info["symbols"],
            format_func=lambda s: s.split(":")[0] + (" (永續)" if ":" in s else " (現貨)"),
        )
    with c2:
        interval = st.selectbox(
            "時間框架",
            ["15", "60", "240", "D"],
            index=1,
            format_func=lambda i: {"15": "15分", "60": "1小時", "240": "4小時", "D": "1日"}[i],
        )

    # Funding rate for futures
    if cat_info["type"] == "future":
        fr = fetch_funding_rate(sym)
        if fr is not None:
            fr_pct = fr * 100
            fr_color = POS if fr > 0 else NEG
            st.markdown(
                f'<div style="margin-bottom:1rem"><span class="category-pill" style="color:{fr_color}; border-color:{fr_color};">'
                f'資金費率 8h · {fr_pct:+.4f}%</span></div>',
                unsafe_allow_html=True,
            )

    tv_symbol = to_tv_symbol(sym, category)

    components.html(f"""
    <div class="tradingview-widget-container" style="height:600px; width:100%;">
      <div id="tv_main"></div>
    </div>
    <script src="https://s3.tradingview.com/tv.js"></script>
    <script>
    new TradingView.widget({{
      "container_id": "tv_main",
      "width": "100%",
      "height": 600,
      "symbol": "{tv_symbol}",
      "interval": "{interval}",
      "timezone": "Asia/Taipei",
      "theme": "dark",
      "style": "1",
      "locale": "zh_TW",
      "toolbar_bg": "#0a0a0a",
      "enable_publishing": false,
      "hide_side_toolbar": false,
      "withdateranges": true,
      "studies": ["MASimple@tv-basicstudies", "RSI@tv-basicstudies"],
      "allow_symbol_change": true
    }});
    </script>
    """, height=620)


# ── Compare: side-by-side TradingView mini charts ─────────────────────────
with tab_compare:
    st.subheader("多標的比較")
    st.caption("並排顯示當前分類下三個標的的 TradingView 走勢,適合快速掃描相對強弱。")

    compare_symbols = st.multiselect(
        "選擇要比較的標的(最多 3 個)",
        cat_info["symbols"],
        default=cat_info["symbols"][:min(3, len(cat_info["symbols"]))],
        max_selections=3,
    )

    if compare_symbols:
        cols = st.columns(len(compare_symbols))
        for col, s in zip(cols, compare_symbols):
            with col:
                tv_s = to_tv_symbol(s, category)
                display_name = s.split(":")[0]
                st.markdown(f"**{display_name}**")
                components.html(f"""
                <div class="tradingview-widget-container">
                  <div id="tv_mini_{display_name}"></div>
                </div>
                <script src="https://s3.tradingview.com/tv.js"></script>
                <script>
                new TradingView.MediumWidget({{
                  "container_id": "tv_mini_{display_name}",
                  "symbols": [["{tv_s}", "{display_name}"]],
                  "chartOnly": false,
                  "width": "100%",
                  "height": 400,
                  "locale": "zh_TW",
                  "colorTheme": "dark",
                  "gridLineColor": "#2a2418",
                  "trendLineColor": "#d4af37",
                  "fontColor": "#9b8b6e",
                  "underLineColor": "rgba(212,175,55,0.15)",
                  "isTransparent": true,
                  "autosize": true,
                  "showFloatingTooltip": true
                }});
                </script>
                """, height=420)


# ── Performance ───────────────────────────────────────────────────────────
with tab_perf:
    st.subheader("模擬交易績效")

    cols = st.columns(4)
    cum_r = perf.cumulative_pnl_R
    cols[0].metric(
        "累計 R", f"{cum_r:+.2f}",
        delta=f"${perf.cumulative_pnl_usdt:+,.0f} USDT", delta_color="off"
    )
    cols[1].metric(
        "勝率", f"{(perf.win_rate or 0):.0%}",
        delta=f"{perf.n_trades} 筆", delta_color="off",
    )
    cols[2].metric(
        "盈虧比", f"{(perf.payoff_ratio or 0):.2f}",
        delta=f"贏 {(perf.avg_win_R or 0):+.2f}R / 輸 {(perf.avg_loss_R or 0):+.2f}R",
        delta_color="off",
    )
    cols[3].metric(
        "最大回撤", f"-{perf.max_drawdown_R:.2f}R",
        delta=f"最長連虧 {perf.longest_losing_streak} 筆", delta_color="off",
    )

    st.markdown("####")
    c1, c2 = st.columns([2, 1])
    with c1:
        st.markdown("**資金曲線(累計 R)**")
        if not df_completed.empty:
            fig = px.area(
                df_completed, x="ts", y="cum_R",
                color_discrete_sequence=[GOLD],
            )
            fig.update_layout(
                height=320, margin=dict(l=0, r=0, t=10, b=0),
                xaxis_title=None, yaxis_title="R-multiple",
                showlegend=False,
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                yaxis=dict(gridcolor=BORDER, color=TEXT_DIM),
                xaxis=dict(gridcolor=BORDER, color=TEXT_DIM),
            )
            fig.update_traces(fillcolor="rgba(212,175,55,0.2)")
            st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown("**R-multiple 分布**")
        if not df_completed.empty:
            fig = go.Figure(go.Histogram(
                x=df_completed["pnl_R"], nbinsx=20,
                marker_color=GOLD, opacity=0.85,
            ))
            fig.add_vline(x=0, line_dash="dash", line_color=TEXT_DIM)
            fig.update_layout(
                height=320, margin=dict(l=0, r=0, t=10, b=0),
                xaxis_title="pnl_R", yaxis_title="筆數", bargap=0.05,
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                yaxis=dict(gridcolor=BORDER, color=TEXT_DIM),
                xaxis=dict(gridcolor=BORDER, color=TEXT_DIM),
            )
            st.plotly_chart(fig, use_container_width=True)


# ── Patterns ──────────────────────────────────────────────────────────────
with tab_patterns:
    st.subheader("標籤歸因")
    st.caption("EV(每筆預期 R 倍數)。n<20 的列會顯示但會標示警告 — 樣本太少容易誤導。")

    c1, c2 = st.columns(2)
    with c1:
        dim = st.selectbox(
            "標籤維度",
            ["setup_type", "key_level", "market_regime", "trader_emotion"],
            key="patterns_dim",
        )
    with c2:
        cross = st.selectbox(
            "交叉維度(選填)",
            ["(無)", "setup_type", "key_level", "market_regime", "trader_emotion"],
            key="patterns_cross",
        )

    if cross == "(無)":
        result = attribution_by_tag(completed, dim=dim)
        rows = [
            {"標籤": str(tag), "n": v["n"], "勝率": v["win_rate"] or 0,
             "EV (R)": v["EV_R"] or 0, "警告": v.get("warning") or ""}
            for tag, v in result.items()
            if tag not in (None, "None")
        ]
        if rows:
            pdf = pd.DataFrame(rows).sort_values("EV (R)", ascending=False)
            st.dataframe(
                pdf, hide_index=True, use_container_width=True,
                column_config={
                    "勝率": st.column_config.ProgressColumn(
                        "勝率", min_value=0, max_value=1, format="%.0f%%"
                    ),
                    "EV (R)": st.column_config.NumberColumn(format="%+.2f"),
                    "n": st.column_config.NumberColumn(format="%d"),
                },
            )
        else:
            st.info(f"`{dim}` 維度沒有資料(可能還沒結算或標籤為空)。")
    else:
        result = attribution_cross(completed, dim_a=dim, dim_b=cross)
        rows = [
            {"A": str(k[0]), "B": str(k[1]), "n": v["n"],
             "EV (R)": v["EV_R"] or 0, "勝率": v["win_rate"] or 0,
             "警告": v.get("warning") or ""}
            for k, v in result.items()
            if k[0] not in (None, "None") and k[1] not in (None, "None")
        ]
        if rows:
            pdf = pd.DataFrame(rows)
            try:
                pivot = pdf.pivot(index="A", columns="B", values="EV (R)")
                st.markdown("**EV 熱力圖(A × B)**")
                fig = px.imshow(
                    pivot, color_continuous_scale="RdYlGn",
                    color_continuous_midpoint=0,
                    aspect="auto", text_auto=".2f",
                )
                fig.update_layout(
                    height=380, margin=dict(l=0, r=0, t=10, b=0),
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font_color=TEXT_DIM,
                )
                st.plotly_chart(fig, use_container_width=True)
            except Exception:
                st.warning("交叉維度資料不足以繪製熱力圖,顯示明細表。")

            st.markdown("**明細表**")
            st.dataframe(
                pdf.sort_values("EV (R)", ascending=False),
                hide_index=True, use_container_width=True,
                column_config={
                    "勝率": st.column_config.ProgressColumn(
                        "勝率", min_value=0, max_value=1, format="%.0f%%"
                    ),
                    "EV (R)": st.column_config.NumberColumn(format="%+.2f"),
                },
            )
        else:
            st.info("交叉維度沒有共同資料。")


# ── Discovery ─────────────────────────────────────────────────────────────
with tab_discovery:
    st.subheader("L7 週度主動發現")

    if "discovery_report" not in st.session_state:
        st.session_state.discovery_report = None

    c1, c2 = st.columns([3, 1])
    with c1:
        st.caption(
            "隱藏關聯(經 Benjamini-Hochberg 多重比較校正)+ 策略衰退警告 + "
            "元能力雷達。每次執行會更新快取。"
        )
    with c2:
        if st.button("執行掃描", type="primary", use_container_width=True):
            with st.spinner("掃描中..."):
                st.session_state.discovery_report = weekly_discovery_report(ctx)

    report = st.session_state.discovery_report
    if report is None:
        st.info("點 **執行掃描** 產生本週的主動發現報告。")
    else:
        # Shorter Chinese labels prevent radar label clipping
        AXIS_LABELS = {
            "confidence_calibration": "信心校準",
            "emotion_control": "情緒控制",
            "session_fit": "時段適配",
            "prediction_skill": "預測能力",
        }
        radar = report["meta_skills"]["radar"]
        axes = [AXIS_LABELS.get(k, k) for k in radar.keys()]
        values = list(radar.values())

        fig = go.Figure(go.Scatterpolar(
            r=values + [values[0]],
            theta=axes + [axes[0]],
            fill="toself",
            line_color=GOLD,
            fillcolor="rgba(212,175,55,0.25)",
        ))
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True, range=[0, 100],
                    gridcolor=BORDER, color=TEXT_DIM,
                ),
                angularaxis=dict(gridcolor=BORDER, color=TEXT),
                bgcolor="rgba(0,0,0,0)",
            ),
            showlegend=False, height=420,
            margin=dict(l=60, r=60, t=40, b=40),
            paper_bgcolor="rgba(0,0,0,0)",
            title=dict(text="元能力雷達(0-100)", font=dict(color=TEXT)),
            font=dict(color=TEXT),
        )

        c1, c2 = st.columns([1, 1])
        c1.plotly_chart(fig, use_container_width=True)

        with c2:
            st.markdown("**隱藏關聯**")
            findings = report["findings"]
            if findings:
                for f in findings:
                    st.markdown(f"- {f}")
            else:
                st.write("✓ 本週沒有顯著關聯。")

            st.markdown("**策略衰退警告**")
            alerts = report["decay_alerts"]
            if alerts:
                for a in alerts:
                    st.warning(a, icon="⚠️")
            else:
                st.write("✓ 沒有衰退中的策略。")

        st.markdown("**下週建議實驗**")
        for i, s in enumerate(report["suggested_experiments"], 1):
            with st.container(border=True):
                st.markdown(f"**實驗 {i}.** {s['hypothesis']}")
                if s.get("conditions"):
                    st.caption(f"觸發條件: {s['conditions']}")
                if s.get("target_n"):
                    st.caption(f"目標樣本數: n={s['target_n']}")


# ── AI Coach ──────────────────────────────────────────────────────────────
with tab_coach:
    st.subheader("AI 教練報告")
    st.caption(
        "把 L7 發現的關聯、元能力診斷、建議實驗,經 Gemini 整理成可執行的"
        "繁體中文週度教練筆記。完全免費(Gemini AI Studio 1500 req/天)。"
    )

    has_key = bool(
        os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    )

    if not has_key:
        st.error(
            "⛔ **尚未設定 GEMINI_API_KEY**  \n"
            "Streamlit Cloud:右上角選單(⋯)→ Settings → Secrets,貼入:  \n"
            "`GEMINI_API_KEY = \"AIzaSy...\"`  \n"
            "免費 key(不用信用卡):https://aistudio.google.com/apikey"
        )
    else:
        c1, c2 = st.columns([2, 1])
        with c1:
            model = st.selectbox(
                "模型",
                ["gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.0-flash"],
                help="Flash:快、免費額度 1500/天。Pro:更聰明但額度較低。",
            )
        with c2:
            st.write("")
            st.write("")
            run = st.button("生成報告", type="primary", use_container_width=True)

        if run:
            from simtrade.l7_discovery import L7AgentExplainer

            with st.spinner(f"呼叫 {model} 中..."):
                try:
                    report = weekly_discovery_report(ctx)
                    explainer = L7AgentExplainer(model=model)
                    result = explainer.explain(report)
                except Exception as e:
                    st.error(f"呼叫失敗:{type(e).__name__}\n\n{str(e)[:500]}")
                    st.info(
                        "如果是 `API_KEY_INVALID`,代表你貼的 key 已失效。"
                        "去 https://aistudio.google.com/apikey 重新 Create Key。"
                    )
                    st.stop()

            with st.container(border=True):
                st.markdown(result.markdown)

            c1, c2 = st.columns([1, 4])
            c1.download_button(
                "⬇ 下載 .md",
                data=result.markdown,
                file_name=f"coaching_{datetime.now():%Y%m%d}.md",
                mime="text/markdown",
                use_container_width=True,
            )
            c2.caption(result.usage_summary())


# ── Trades ────────────────────────────────────────────────────────────────
with tab_trades:
    st.subheader("決策紀錄")

    if df.empty:
        st.info("尚無交易紀錄。")
    else:
        c1, c2, c3, c4 = st.columns(4)
        side_filter = c1.multiselect(
            "方向", options=sorted(df["side"].dropna().unique()),
        )
        setup_filter = c2.multiselect(
            "Setup", options=sorted(df["setup_type"].dropna().unique()),
        )
        regime_filter = c3.multiselect(
            "市場狀態", options=sorted(df["market_regime"].dropna().unique()),
        )
        outcome_filter = c4.selectbox(
            "結果", ["全部", "僅獲利", "僅虧損", "僅未結算"]
        )

        view = df.copy()
        if side_filter:
            view = view[view["side"].isin(side_filter)]
        if setup_filter:
            view = view[view["setup_type"].isin(setup_filter)]
        if regime_filter:
            view = view[view["market_regime"].isin(regime_filter)]
        if outcome_filter == "僅獲利":
            view = view[view["pnl_R"] > 0]
        elif outcome_filter == "僅虧損":
            view = view[view["pnl_R"] < 0]
        elif outcome_filter == "僅未結算":
            view = view[view["pnl_R"].isna()]

        st.caption(f"顯示 {len(view)} / {len(df)} 筆")
        st.dataframe(
            view[["ts", "symbol", "side", "setup_type", "market_regime",
                  "confidence", "entry", "pnl_R", "exit_reason", "id"]],
            hide_index=True, use_container_width=True,
            column_config={
                "ts": st.column_config.DatetimeColumn("時間", format="YYYY-MM-DD HH:mm"),
                "symbol": "標的",
                "side": "方向",
                "setup_type": "Setup",
                "market_regime": "市場狀態",
                "confidence": st.column_config.NumberColumn("信心", format="%d"),
                "entry": st.column_config.NumberColumn("進場價", format="%.2f"),
                "pnl_R": st.column_config.NumberColumn("pnl R", format="%+.2f"),
                "exit_reason": "出場原因",
                "id": "ID",
            },
        )


# ── Footer ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="footer">
      <strong style="color:#d4af37; letter-spacing:0.1em;">GOLD TRADER LITE</strong>
      &nbsp;·&nbsp; Tier 4 主動發現型模擬交易平台 &nbsp;·&nbsp; MIT License
      <br>
      L1 資料層 → L2 指標 → L3 訂單模擬 → L4 持倉 → L5 反饋 → L6 學習 → L7 主動發現
      <br><br>
      <a href="https://github.com/lide015/Simulated-trading" target="_blank">GitHub</a>
      &nbsp;·&nbsp;
      <a href="https://aistudio.google.com/apikey" target="_blank">Gemini API</a>
      &nbsp;·&nbsp;
      <a href="https://www.tradingview.com/" target="_blank">TradingView</a>
    </div>
    """,
    unsafe_allow_html=True,
)
