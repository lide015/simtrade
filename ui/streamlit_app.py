"""Streamlit dashboard for the simtrade platform.

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
    page_title="Simtrade",
    page_icon=":bar_chart:",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    div[data-testid="stMetricValue"] { font-size: 2rem; }
    div[data-testid="stMetricDelta"] { font-size: 0.85rem; }
    [data-testid="stSidebar"] { background-color: #fafafa; }
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
    .stTabs [data-baseweb="tab"] { padding: 8px 4px; }
    </style>
    """,
    unsafe_allow_html=True,
)

if "GEMINI_API_KEY" in st.secrets and not os.environ.get("GEMINI_API_KEY"):
    os.environ["GEMINI_API_KEY"] = st.secrets["GEMINI_API_KEY"]

GREEN = "#16a34a"
RED = "#dc2626"
AMBER = "#f59e0b"
BLUE = "#2563eb"
GRAY = "#6b7280"


# ── Sidebar ───────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Simtrade")
    st.caption("Tier 4 simulated trading with active discovery")
    st.divider()
    db_path = st.text_input("Database", value="data/simtrade.db")
    st.divider()
    st.caption(
        "[L1-L7 architecture](https://github.com/lide015/Simulated-trading)  "
        "·  [Gemini key](https://aistudio.google.com/apikey)"
    )

db_file = Path(db_path)
if not db_file.exists():
    st.warning(f"Database `{db_path}` not found.")
    st.write(
        "First launch on a fresh deployment. Seed 60 synthetic decisions to "
        "see the dashboard in action."
    )
    if st.button("Generate demo data", type="primary"):
        db_file.parent.mkdir(parents=True, exist_ok=True)
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from scripts.demo import main as demo_main  # noqa: E402

        with st.spinner("Generating 60 synthetic trades (~30s)..."):
            demo_main(db_path=str(db_file))
        st.success("Done. Reloading...")
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
tab_overview, tab_patterns, tab_discovery, tab_coach, tab_trades = st.tabs(
    [":bar_chart: Overview", ":bookmark_tabs: Patterns",
     ":mag: Discovery", ":robot_face: AI Coach", ":memo: Trades"]
)


# ── Overview ──────────────────────────────────────────────────────────────
with tab_overview:
    st.subheader("Performance overview")

    cols = st.columns(4)
    cum_r = perf.cumulative_pnl_R
    cols[0].metric(
        "Cumulative R", f"{cum_r:+.2f}",
        delta=f"${perf.cumulative_pnl_usdt:+,.0f} USDT", delta_color="off"
    )
    cols[1].metric(
        "Win rate",
        f"{(perf.win_rate or 0):.0%}",
        delta=f"{perf.n_trades} trades", delta_color="off",
    )
    cols[2].metric(
        "Payoff ratio",
        f"{(perf.payoff_ratio or 0):.2f}",
        delta=f"avg win {(perf.avg_win_R or 0):+.2f}R / loss {(perf.avg_loss_R or 0):+.2f}R",
        delta_color="off",
    )
    cols[3].metric(
        "Max DD",
        f"-{perf.max_drawdown_R:.2f}R",
        delta=f"streak {perf.longest_losing_streak}",
        delta_color="off",
    )

    st.markdown("####")
    c1, c2 = st.columns([2, 1])
    with c1:
        st.markdown("**Equity curve (cumulative R)**")
        if not df_completed.empty:
            fig = px.area(
                df_completed, x="ts", y="cum_R",
                color_discrete_sequence=[BLUE],
            )
            fig.update_layout(
                height=320, margin=dict(l=0, r=0, t=10, b=0),
                xaxis_title=None, yaxis_title="R-multiple",
                showlegend=False,
            )
            fig.update_traces(fillcolor="rgba(37,99,235,0.15)")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No completed trades yet.")

    with c2:
        st.markdown("**R-multiple distribution**")
        if not df_completed.empty:
            colors = [GREEN if r > 0 else RED for r in df_completed["pnl_R"]]
            fig = go.Figure(go.Histogram(
                x=df_completed["pnl_R"], nbinsx=20,
                marker_color=BLUE, opacity=0.85,
            ))
            fig.add_vline(x=0, line_dash="dash", line_color=GRAY)
            fig.update_layout(
                height=320, margin=dict(l=0, r=0, t=10, b=0),
                xaxis_title="pnl_R", yaxis_title="count",
                bargap=0.05,
            )
            st.plotly_chart(fig, use_container_width=True)


# ── Patterns ──────────────────────────────────────────────────────────────
with tab_patterns:
    st.subheader("Attribution by tag")
    st.caption(
        "EV (expected value per trade) in R-multiples. Rows with n<20 are "
        "shown but flagged — small samples mislead."
    )

    c1, c2 = st.columns(2)
    with c1:
        dim = st.selectbox(
            "Tag dimension",
            ["setup_type", "key_level", "market_regime", "trader_emotion"],
            key="patterns_dim",
        )
    with c2:
        cross = st.selectbox(
            "Cross with (optional)",
            ["(none)", "setup_type", "key_level", "market_regime", "trader_emotion"],
            key="patterns_cross",
        )

    if cross == "(none)":
        result = attribution_by_tag(completed, dim=dim)
        rows = [
            {
                "Tag": tag,
                "n": v["n"],
                "Win rate": v["win_rate"],
                "EV (R)": v["EV_R"],
                "Flag": v.get("warning") or "",
            }
            for tag, v in result.items()
            if tag is not None
        ]
        if rows:
            pdf = pd.DataFrame(rows).sort_values("EV (R)", ascending=False)
            st.dataframe(
                pdf,
                hide_index=True,
                use_container_width=True,
                column_config={
                    "Win rate": st.column_config.ProgressColumn(
                        "Win rate", min_value=0, max_value=1, format="%.0f%%"
                    ),
                    "EV (R)": st.column_config.NumberColumn(
                        "EV (R)", format="%+.2f"
                    ),
                    "n": st.column_config.NumberColumn("n", format="%d"),
                },
            )
        else:
            st.info("No data for that dimension.")
    else:
        result = attribution_cross(completed, dim_a=dim, dim_b=cross)
        rows = [
            {
                "A": k[0],
                "B": k[1],
                "n": v["n"],
                "EV (R)": v["EV_R"],
                "Win rate": v["win_rate"],
                "Flag": v.get("warning") or "",
            }
            for k, v in result.items()
            if k[0] is not None and k[1] is not None
        ]
        if rows:
            pdf = pd.DataFrame(rows)
            pivot = pdf.pivot(index="A", columns="B", values="EV (R)")
            st.markdown("**Heatmap: EV by (A, B)**")
            fig = px.imshow(
                pivot, color_continuous_scale="RdYlGn",
                color_continuous_midpoint=0,
                aspect="auto", text_auto=".2f",
            )
            fig.update_layout(height=380, margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("**Raw table**")
            st.dataframe(
                pdf.sort_values("EV (R)", ascending=False),
                hide_index=True, use_container_width=True,
                column_config={
                    "Win rate": st.column_config.ProgressColumn(
                        "Win rate", min_value=0, max_value=1, format="%.0f%%"
                    ),
                    "EV (R)": st.column_config.NumberColumn(format="%+.2f"),
                },
            )


# ── Discovery ─────────────────────────────────────────────────────────────
with tab_discovery:
    st.subheader("L7 weekly discovery")

    if "discovery_report" not in st.session_state:
        st.session_state.discovery_report = None

    c1, c2 = st.columns([3, 1])
    with c1:
        st.caption(
            "Hidden correlations (Benjamini-Hochberg corrected) + regime "
            "decay alerts + meta-skill radar. Refreshes the cache each run."
        )
    with c2:
        if st.button("Run scan", type="primary", use_container_width=True):
            with st.spinner("Scanning..."):
                st.session_state.discovery_report = weekly_discovery_report(ctx)

    report = st.session_state.discovery_report
    if report is None:
        st.info("Click **Run scan** to generate this week's discovery report.")
    else:
        # Radar chart
        radar = report["meta_skills"]["radar"]
        axes = list(radar.keys())
        values = list(radar.values())
        fig = go.Figure(go.Scatterpolar(
            r=values + [values[0]],
            theta=axes + [axes[0]],
            fill="toself",
            line_color=BLUE,
            fillcolor="rgba(37,99,235,0.2)",
        ))
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            showlegend=False, height=380,
            margin=dict(l=40, r=40, t=20, b=20),
            title="Meta-skill radar (0-100)",
        )

        c1, c2 = st.columns([1, 1])
        c1.plotly_chart(fig, use_container_width=True)

        with c2:
            st.markdown("**Findings**")
            findings = report["findings"]
            if findings:
                for f in findings:
                    st.markdown(f"- {f}")
            else:
                st.write(":heavy_check_mark: No significant correlations this week.")

            st.markdown("**Decay alerts**")
            alerts = report["decay_alerts"]
            if alerts:
                for a in alerts:
                    st.warning(a, icon=":warning:")
            else:
                st.write(":heavy_check_mark: No decaying setups.")

        st.markdown("**Suggested experiments (next week)**")
        for i, s in enumerate(report["suggested_experiments"], 1):
            with st.container(border=True):
                st.markdown(f"**Experiment {i}.** {s['hypothesis']}")
                if s.get("conditions"):
                    st.caption(f"Conditions: {s['conditions']}")
                if s.get("target_n"):
                    st.caption(f"Target sample: n={s['target_n']}")


# ── AI Coach ──────────────────────────────────────────────────────────────
with tab_coach:
    st.subheader("AI coaching report")
    st.caption(
        "L7 findings + meta-skill diagnosis + experiment suggestions, "
        "turned into a concrete weekly coaching note by Gemini."
    )

    has_key = bool(
        os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    )

    if not has_key:
        st.error(
            ":no_entry_sign: **GEMINI_API_KEY not configured.**  \n"
            "On Streamlit Cloud: app menu (⋯) → Settings → Secrets, paste:  \n"
            "`GEMINI_API_KEY = \"AIzaSy...\"`  \n"
            "Free key (no credit card): https://aistudio.google.com/apikey"
        )
    else:
        c1, c2 = st.columns([2, 1])
        with c1:
            model = st.selectbox(
                "Model",
                ["gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.0-flash"],
                help="Flash: fast, free 1500/day. Pro: smarter, lower quota.",
            )
        with c2:
            st.write("")
            st.write("")
            run = st.button("Generate report", type="primary", use_container_width=True)

        if run:
            from simtrade.l7_discovery import L7AgentExplainer

            with st.spinner(f"Calling {model}..."):
                report = weekly_discovery_report(ctx)
                explainer = L7AgentExplainer(model=model)
                result = explainer.explain(report)

            with st.container(border=True):
                st.markdown(result.markdown)

            c1, c2 = st.columns([1, 4])
            c1.download_button(
                ":arrow_down: Download .md",
                data=result.markdown,
                file_name=f"coaching_{datetime.now():%Y%m%d}.md",
                mime="text/markdown",
                use_container_width=True,
            )
            c2.caption(result.usage_summary())


# ── Trades ────────────────────────────────────────────────────────────────
with tab_trades:
    st.subheader("Decision records")

    if df.empty:
        st.info("No trades recorded yet.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        side_filter = c1.multiselect(
            "Side", options=sorted(df["side"].dropna().unique()),
        )
        setup_filter = c2.multiselect(
            "Setup", options=sorted(df["setup_type"].dropna().unique()),
        )
        regime_filter = c3.multiselect(
            "Regime", options=sorted(df["market_regime"].dropna().unique()),
        )
        outcome_filter = c4.selectbox(
            "Outcome", ["all", "wins only", "losses only", "pending only"]
        )

        view = df.copy()
        if side_filter:
            view = view[view["side"].isin(side_filter)]
        if setup_filter:
            view = view[view["setup_type"].isin(setup_filter)]
        if regime_filter:
            view = view[view["market_regime"].isin(regime_filter)]
        if outcome_filter == "wins only":
            view = view[view["pnl_R"] > 0]
        elif outcome_filter == "losses only":
            view = view[view["pnl_R"] < 0]
        elif outcome_filter == "pending only":
            view = view[view["pnl_R"].isna()]

        st.caption(f"{len(view)} of {len(df)} trades")
        st.dataframe(
            view[["ts", "symbol", "side", "setup_type", "market_regime",
                  "confidence", "entry", "pnl_R", "exit_reason", "id"]],
            hide_index=True, use_container_width=True,
            column_config={
                "ts": st.column_config.DatetimeColumn(
                    "Time", format="YYYY-MM-DD HH:mm"
                ),
                "pnl_R": st.column_config.NumberColumn(
                    "pnl R", format="%+.2f"
                ),
                "confidence": st.column_config.NumberColumn(
                    "Conf", format="%d"
                ),
                "entry": st.column_config.NumberColumn(format="%.2f"),
            },
        )
