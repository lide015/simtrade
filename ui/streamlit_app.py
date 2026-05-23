"""Simtrade dashboard — crypto decision lab.

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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from simtrade.l5_feedback import compute_performance  # noqa: E402
from simtrade.l6_learning import attribution_by_tag, attribution_cross  # noqa: E402
from simtrade.platform import boot, weekly_discovery_report  # noqa: E402

st.set_page_config(
    page_title="Simtrade · 加密貨幣決策實驗室",
    page_icon=":chart_with_upwards_trend:",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Theme colors ──────────────────────────────────────────────────────────
GREEN = "#10b981"
RED = "#ef4444"
AMBER = "#f59e0b"
BLUE = "#3b82f6"
PURPLE = "#8b5cf6"
GRAY = "#6b7280"
BG = "#0a0e1a"
CARD = "#131825"
BORDER = "#1f2937"

st.markdown(
    f"""
    <style>
    /* Base */
    .block-container {{ padding-top: 1rem; padding-bottom: 3rem; max-width: 1400px; }}
    [data-testid="stHeader"] {{ background: transparent; }}
    [data-testid="stSidebar"] {{ background-color: {CARD}; border-right: 1px solid {BORDER}; }}

    /* Headings */
    h1, h2, h3, h4 {{ font-weight: 600; letter-spacing: -0.01em; }}

    /* Metrics */
    div[data-testid="stMetric"] {{
        background: linear-gradient(135deg, {CARD} 0%, rgba(19,24,37,0.5) 100%);
        border: 1px solid {BORDER};
        border-radius: 12px;
        padding: 1rem 1.25rem;
        transition: border-color 0.2s;
    }}
    div[data-testid="stMetric"]:hover {{ border-color: {BLUE}; }}
    div[data-testid="stMetricValue"] {{ font-size: 1.75rem; font-weight: 700; }}
    div[data-testid="stMetricDelta"] {{ font-size: 0.8rem; opacity: 0.8; }}
    div[data-testid="stMetricLabel"] {{
        font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.05em;
        color: {GRAY};
    }}

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 8px; border-bottom: 1px solid {BORDER}; padding-bottom: 0;
    }}
    .stTabs [data-baseweb="tab"] {{
        padding: 12px 18px; border-radius: 8px 8px 0 0;
        font-weight: 500; font-size: 0.95rem;
    }}
    .stTabs [aria-selected="true"] {{ background-color: rgba(59,130,246,0.1); }}

    /* Buttons */
    .stButton > button {{
        border-radius: 8px; font-weight: 500; transition: all 0.15s;
        border: 1px solid {BORDER};
    }}
    .stButton > button[kind="primary"] {{
        background: linear-gradient(135deg, {BLUE} 0%, {PURPLE} 100%);
        border: none; box-shadow: 0 2px 8px rgba(59,130,246,0.3);
    }}
    .stButton > button[kind="primary"]:hover {{
        transform: translateY(-1px); box-shadow: 0 4px 12px rgba(59,130,246,0.4);
    }}

    /* Containers */
    [data-testid="stContainer"][data-baseweb="card"], .st-emotion-cache-r6ttn4 {{
        background: {CARD}; border: 1px solid {BORDER}; border-radius: 12px;
    }}

    /* Custom hero */
    .hero {{
        padding: 1.5rem 0 2rem; border-bottom: 1px solid {BORDER};
        margin-bottom: 1.5rem;
    }}
    .hero-title {{
        font-size: 2.25rem; font-weight: 700; margin: 0; letter-spacing: -0.02em;
        background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 50%, #ec4899 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        background-clip: text;
    }}
    .hero-tagline {{
        margin: 0.25rem 0 0; color: {GRAY}; font-size: 0.95rem;
    }}

    /* Ticker strip */
    .ticker-strip {{
        display: flex; gap: 1rem; padding: 1rem; margin-bottom: 1.5rem;
        background: {CARD}; border: 1px solid {BORDER}; border-radius: 12px;
        overflow-x: auto;
    }}
    .ticker {{
        flex: 1; min-width: 140px; padding: 0.5rem 0.75rem;
        border-left: 3px solid {BLUE};
    }}
    .ticker-symbol {{
        font-size: 0.75rem; color: {GRAY}; text-transform: uppercase;
        letter-spacing: 0.05em; font-weight: 500;
    }}
    .ticker-price {{
        font-size: 1.15rem; font-weight: 700; margin: 0.15rem 0; font-variant-numeric: tabular-nums;
    }}
    .ticker-change {{ font-size: 0.85rem; font-weight: 600; font-variant-numeric: tabular-nums; }}
    .pos {{ color: {GREEN}; }}
    .neg {{ color: {RED}; }}

    /* Footer */
    .footer {{
        margin-top: 4rem; padding-top: 1.5rem; border-top: 1px solid {BORDER};
        text-align: center; color: {GRAY}; font-size: 0.8rem;
    }}
    .footer a {{ color: {BLUE}; text-decoration: none; }}
    </style>
    """,
    unsafe_allow_html=True,
)

try:
    if "GEMINI_API_KEY" in st.secrets and not os.environ.get("GEMINI_API_KEY"):
        os.environ["GEMINI_API_KEY"] = st.secrets["GEMINI_API_KEY"]
except Exception:
    pass


# ── Live market data (cached) ─────────────────────────────────────────────
@st.cache_data(ttl=60, show_spinner=False)
def fetch_tickers(symbols: tuple[str, ...]) -> dict:
    try:
        import ccxt

        ex = ccxt.binance({"enableRateLimit": True})
        out = {}
        for s in symbols:
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
def fetch_candles(symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
    try:
        import ccxt

        ex = ccxt.binance({"enableRateLimit": True})
        ohlcv = ex.fetch_ohlcv(symbol, timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=["ts", "open", "high", "low", "close", "vol"])
        df["ts"] = pd.to_datetime(df["ts"], unit="ms")
        df["ma20"] = df["close"].rolling(20).mean()
        df["ma50"] = df["close"].rolling(50).mean()
        return df
    except Exception:
        return pd.DataFrame()


# ── Hero header ───────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="hero">
      <h1 class="hero-title">SIMTRADE</h1>
      <p class="hero-tagline">加密貨幣決策實驗室 · 主動發現你的交易盲點 · L1-L7 七層架構</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ── Ticker strip ──────────────────────────────────────────────────────────
TICKER_SYMBOLS = ("BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT")
tickers = fetch_tickers(TICKER_SYMBOLS)

if tickers:
    ticker_html = '<div class="ticker-strip">'
    for sym in TICKER_SYMBOLS:
        if sym not in tickers:
            continue
        t = tickers[sym]
        price = t.get("price") or 0
        change = t.get("change_pct") or 0
        cls = "pos" if change >= 0 else "neg"
        arrow = "▲" if change >= 0 else "▼"
        # Format price compactly
        if price >= 100:
            price_s = f"${price:,.2f}"
        elif price >= 1:
            price_s = f"${price:.3f}"
        else:
            price_s = f"${price:.5f}"
        ticker_html += f"""
          <div class="ticker">
            <div class="ticker-symbol">{sym}</div>
            <div class="ticker-price">{price_s}</div>
            <div class="ticker-change {cls}">{arrow} {abs(change):.2f}%</div>
          </div>
        """
    ticker_html += "</div>"
    st.markdown(ticker_html, unsafe_allow_html=True)
else:
    st.info(
        "即時行情暫時無法取得(Binance API 可能對部署區域限流)。"
        "其他功能不受影響。"
    )


# ── Sidebar ───────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 設定")
    db_path = st.text_input("資料庫路徑", value="data/simtrade.db")
    st.divider()
    st.markdown("### 關於")
    st.caption(
        "**Simtrade** 是一個 Tier 4 模擬交易平台,核心理念是「資料找你」而不是"
        "「你查資料」 — 系統主動發現你沒注意到的交易模式,並用 AI 教練給你"
        "週度建議。"
    )
    st.markdown(
        "- [L1-L7 架構文件](https://github.com/lide015/Simulated-trading)\n"
        "- [取得 Gemini key](https://aistudio.google.com/apikey)\n"
        "- [Streamlit Cloud](https://share.streamlit.io)"
    )

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


# ── Data prep ─────────────────────────────────────────────────────────────
def _records_to_df(records) -> pd.DataFrame:
    rows = []
    for r in records:
        outcome = r.post_outcome or {}
        action = r.action
        tags = r.trader_state.get("reasoning_tags", {})
        rows.append(
            {
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
            }
        )
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("ts").reset_index(drop=True)
        df["cum_R"] = df["pnl_R"].fillna(0).cumsum()
    return df


df = _records_to_df(all_records)
df_completed = df[df["pnl_R"].notna()].copy() if not df.empty else df
perf = compute_performance(ctx.conn)


# ── Tabs ──────────────────────────────────────────────────────────────────
tab_market, tab_perf, tab_patterns, tab_discovery, tab_coach, tab_trades = st.tabs(
    [":chart_with_upwards_trend: 即時行情",
     ":bar_chart: 績效",
     ":bookmark_tabs: 標籤歸因",
     ":mag: 主動發現",
     ":robot_face: AI 教練",
     ":memo: 交易紀錄"]
)


# ── Market tab ────────────────────────────────────────────────────────────
with tab_market:
    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        sym = st.selectbox(
            "標的",
            list(TICKER_SYMBOLS),
            index=0,
        )
    with c2:
        tf = st.selectbox("時間框架", ["15m", "1h", "4h", "1d"], index=1)
    with c3:
        n_bars = st.selectbox("K 棒數", [50, 100, 200, 500], index=1)

    candles = fetch_candles(sym, tf, n_bars)
    if candles.empty:
        st.warning("無法取得 K 線資料(API 可能對部署區域限流)。")
    else:
        last_price = candles["close"].iloc[-1]
        first_price = candles["close"].iloc[0]
        change_pct = (last_price - first_price) / first_price * 100
        high_period = candles["high"].max()
        low_period = candles["low"].min()
        vol = candles["vol"].sum()

        cols = st.columns(4)
        cols[0].metric(
            "目前價格",
            f"${last_price:,.2f}",
            delta=f"{change_pct:+.2f}% (期間)",
        )
        cols[1].metric("期間最高", f"${high_period:,.2f}")
        cols[2].metric("期間最低", f"${low_period:,.2f}")
        cols[3].metric("成交量", f"${vol/1e6:,.1f}M")

        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=candles["ts"],
            open=candles["open"], high=candles["high"],
            low=candles["low"], close=candles["close"],
            increasing_line_color=GREEN, decreasing_line_color=RED,
            name=sym,
        ))
        fig.add_trace(go.Scatter(
            x=candles["ts"], y=candles["ma20"],
            line=dict(color=BLUE, width=1), name="MA20",
        ))
        fig.add_trace(go.Scatter(
            x=candles["ts"], y=candles["ma50"],
            line=dict(color=PURPLE, width=1), name="MA50",
        ))
        fig.update_layout(
            height=500, margin=dict(l=0, r=0, t=10, b=0),
            xaxis_rangeslider_visible=False,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            yaxis=dict(gridcolor=BORDER), xaxis=dict(gridcolor=BORDER),
            legend=dict(orientation="h", y=1.02, yanchor="bottom"),
        )
        st.plotly_chart(fig, use_container_width=True)

        # Volume bar
        vol_fig = go.Figure()
        vol_colors = [
            GREEN if c >= o else RED
            for c, o in zip(candles["close"], candles["open"])
        ]
        vol_fig.add_trace(go.Bar(
            x=candles["ts"], y=candles["vol"],
            marker_color=vol_colors, opacity=0.7,
        ))
        vol_fig.update_layout(
            height=140, margin=dict(l=0, r=0, t=0, b=0),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            yaxis=dict(gridcolor=BORDER, title="成交量"),
            xaxis=dict(gridcolor=BORDER),
            showlegend=False, bargap=0.05,
        )
        st.plotly_chart(vol_fig, use_container_width=True)


# ── Performance tab ───────────────────────────────────────────────────────
with tab_perf:
    st.subheader("模擬交易績效")

    cols = st.columns(4)
    cum_r = perf.cumulative_pnl_R
    cols[0].metric(
        "累計 R", f"{cum_r:+.2f}",
        delta=f"${perf.cumulative_pnl_usdt:+,.0f} USDT", delta_color="off"
    )
    cols[1].metric(
        "勝率",
        f"{(perf.win_rate or 0):.0%}",
        delta=f"{perf.n_trades} 筆", delta_color="off",
    )
    cols[2].metric(
        "盈虧比",
        f"{(perf.payoff_ratio or 0):.2f}",
        delta=f"贏 {(perf.avg_win_R or 0):+.2f}R / 輸 {(perf.avg_loss_R or 0):+.2f}R",
        delta_color="off",
    )
    cols[3].metric(
        "最大回撤",
        f"-{perf.max_drawdown_R:.2f}R",
        delta=f"最長連虧 {perf.longest_losing_streak} 筆",
        delta_color="off",
    )

    st.markdown("####")
    c1, c2 = st.columns([2, 1])
    with c1:
        st.markdown("**資金曲線(累計 R)**")
        if not df_completed.empty:
            fig = px.area(
                df_completed, x="ts", y="cum_R",
                color_discrete_sequence=[BLUE],
            )
            fig.update_layout(
                height=320, margin=dict(l=0, r=0, t=10, b=0),
                xaxis_title=None, yaxis_title="R-multiple",
                showlegend=False,
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                yaxis=dict(gridcolor=BORDER), xaxis=dict(gridcolor=BORDER),
            )
            fig.update_traces(fillcolor="rgba(59,130,246,0.2)")
            st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown("**R-multiple 分布**")
        if not df_completed.empty:
            fig = go.Figure(go.Histogram(
                x=df_completed["pnl_R"], nbinsx=20,
                marker_color=BLUE, opacity=0.85,
            ))
            fig.add_vline(x=0, line_dash="dash", line_color=GRAY)
            fig.update_layout(
                height=320, margin=dict(l=0, r=0, t=10, b=0),
                xaxis_title="pnl_R", yaxis_title="筆數",
                bargap=0.05,
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                yaxis=dict(gridcolor=BORDER), xaxis=dict(gridcolor=BORDER),
            )
            st.plotly_chart(fig, use_container_width=True)


# ── Patterns ──────────────────────────────────────────────────────────────
with tab_patterns:
    st.subheader("標籤歸因")
    st.caption(
        "EV(每筆預期 R 倍數)。n<20 的列會顯示但會標示警告 — 樣本太少容易誤導。"
    )

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
            {"標籤": tag, "n": v["n"], "勝率": v["win_rate"],
             "EV (R)": v["EV_R"], "警告": v.get("warning") or ""}
            for tag, v in result.items()
            if tag is not None
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
            st.info("這個維度沒有資料。")
    else:
        result = attribution_cross(completed, dim_a=dim, dim_b=cross)
        rows = [
            {"A": k[0], "B": k[1], "n": v["n"],
             "EV (R)": v["EV_R"], "勝率": v["win_rate"],
             "警告": v.get("warning") or ""}
            for k, v in result.items()
            if k[0] is not None and k[1] is not None
        ]
        if rows:
            pdf = pd.DataFrame(rows)
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
            )
            st.plotly_chart(fig, use_container_width=True)

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
        radar = report["meta_skills"]["radar"]
        axes = list(radar.keys())
        values = list(radar.values())
        fig = go.Figure(go.Scatterpolar(
            r=values + [values[0]],
            theta=axes + [axes[0]],
            fill="toself",
            line_color=BLUE,
            fillcolor="rgba(59,130,246,0.25)",
        ))
        fig.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 100], gridcolor=BORDER),
                angularaxis=dict(gridcolor=BORDER),
                bgcolor="rgba(0,0,0,0)",
            ),
            showlegend=False, height=400,
            margin=dict(l=40, r=40, t=20, b=20),
            paper_bgcolor="rgba(0,0,0,0)",
            title="元能力雷達(0-100)",
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
                st.write(":heavy_check_mark: 本週沒有顯著關聯。")

            st.markdown("**策略衰退警告**")
            alerts = report["decay_alerts"]
            if alerts:
                for a in alerts:
                    st.warning(a, icon=":warning:")
            else:
                st.write(":heavy_check_mark: 沒有衰退中的策略。")

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
        "週度教練筆記。完全免費(Gemini AI Studio 1500 req/天)。"
    )

    has_key = bool(
        os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    )

    if not has_key:
        st.error(
            ":no_entry_sign: **尚未設定 GEMINI_API_KEY**  \n"
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
                report = weekly_discovery_report(ctx)
                explainer = L7AgentExplainer(model=model)
                result = explainer.explain(report)

            with st.container(border=True):
                st.markdown(result.markdown)

            c1, c2 = st.columns([1, 4])
            c1.download_button(
                ":arrow_down: 下載 .md",
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
      Simtrade · Tier 4 主動發現型模擬交易平台 · MIT License<br>
      L1 資料層 → L2 指標 → L3 訂單模擬 → L4 持倉 → L5 反饋 → L6 學習引擎 → L7 主動發現
      <br><br>
      <a href="https://github.com/lide015/Simulated-trading" target="_blank">GitHub</a>
      &nbsp;·&nbsp;
      <a href="https://aistudio.google.com/apikey" target="_blank">Gemini API</a>
      &nbsp;·&nbsp;
      <a href="https://share.streamlit.io" target="_blank">Streamlit Cloud</a>
    </div>
    """,
    unsafe_allow_html=True,
)
