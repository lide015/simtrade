"""Streamlit dashboard for the simtrade platform.

Runs locally (streamlit run ui/streamlit_app.py) and on Streamlit Community
Cloud. On Cloud, set GEMINI_API_KEY in App Secrets; locally either set the
env var or run python scripts/demo.py first then `streamlit run`.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from simtrade.l5_feedback import compute_performance  # noqa: E402
from simtrade.l6_learning import attribution_by_tag, attribution_cross  # noqa: E402
from simtrade.platform import boot, weekly_discovery_report  # noqa: E402

st.set_page_config(page_title="Simtrade", layout="wide")

# Read GEMINI_API_KEY from Streamlit Secrets first (Cloud), env var as fallback.
if "GEMINI_API_KEY" in st.secrets and not os.environ.get("GEMINI_API_KEY"):
    os.environ["GEMINI_API_KEY"] = st.secrets["GEMINI_API_KEY"]

DEFAULT_DB = "data/simtrade.db"
db_path = st.sidebar.text_input("Database path", value=DEFAULT_DB)

# If DB doesn't exist (e.g. first launch on Cloud), offer to seed demo data
# so the dashboard isn't empty.
db_file = Path(db_path)
if not db_file.exists():
    st.warning(f"Database `{db_path}` not found.")
    st.write(
        "First launch on a fresh deployment. Click below to seed 60 synthetic "
        "decisions so the dashboard has something to show."
    )
    if st.button("Generate demo data"):
        db_file.parent.mkdir(parents=True, exist_ok=True)
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from scripts.demo import main as demo_main  # noqa: E402

        with st.spinner("Generating 60 synthetic trades..."):
            demo_main(db_path=str(db_file))
        st.success("Demo data generated. Refresh the page.")
    st.stop()

ctx = boot(db_path=db_path, with_market=False)

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Performance", "Attribution", "Discovery", "LLM Explain", "Decisions"]
)

with tab1:
    st.header("L5 Performance")
    perf = compute_performance(ctx.conn)
    cols = st.columns(4)
    cols[0].metric("Trades", perf.n_trades)
    cols[1].metric("Cum R", f"{perf.cumulative_pnl_R:+.2f}")
    cols[2].metric("Win rate", f"{(perf.win_rate or 0):.0%}")
    cols[3].metric("Max DD (R)", f"{perf.max_drawdown_R:.2f}")
    st.json(perf.to_dict())

with tab2:
    st.header("L6 Attribution")
    dim = st.selectbox(
        "Tag dimension",
        ["setup_type", "key_level", "market_regime", "trader_emotion"],
    )
    cross = st.selectbox(
        "Cross with (optional)",
        ["", "setup_type", "key_level", "market_regime", "trader_emotion"],
    )
    records = ctx.decisions.completed()
    if cross:
        result = attribution_cross(records, dim_a=dim, dim_b=cross)
        st.write({f"{k[0]} | {k[1]}": v for k, v in result.items()})
    else:
        st.write(attribution_by_tag(records, dim=dim))

with tab3:
    st.header("L7 Discovery Report")
    if st.button("Run discovery scan"):
        report = weekly_discovery_report(ctx)
        st.subheader("Radar")
        st.write(report["meta_skills"]["radar"])
        st.subheader("Hidden correlations")
        for f in report["findings"]:
            st.write(f"- {f}")
        st.subheader("Decay alerts")
        for a in report["decay_alerts"]:
            st.warning(a)
        st.subheader("Suggested experiments")
        for s in report["suggested_experiments"]:
            st.info(s["hypothesis"])

with tab4:
    st.header("LLM Coaching Report")
    has_key = bool(
        os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    )
    if not has_key:
        st.error(
            "GEMINI_API_KEY not configured. On Streamlit Cloud: App settings "
            "-> Secrets, add `GEMINI_API_KEY = \"AIzaSy...\"`. Free key at "
            "https://aistudio.google.com/apikey."
        )
    else:
        model = st.selectbox(
            "Model",
            ["gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.0-flash"],
            index=0,
            help="Flash is fast + free 1500/day. Pro is smarter but lower quota.",
        )
        if st.button("Generate coaching report"):
            from simtrade.l7_discovery import L7AgentExplainer

            with st.spinner(f"Calling {model}..."):
                report = weekly_discovery_report(ctx)
                explainer = L7AgentExplainer(model=model)
                result = explainer.explain(report)
            st.markdown(result.markdown)
            st.caption(result.usage_summary())
            st.download_button(
                "Download .md",
                data=result.markdown,
                file_name="coaching_report.md",
                mime="text/markdown",
            )

with tab5:
    st.header("DecisionRecord browser")
    records = ctx.decisions.all()
    st.write(f"Total: {len(records)}")
    for r in records[-20:][::-1]:
        with st.expander(f"{r.ts}  {r.symbol}  {r.action.get('side')}  id={r.id[:8]}"):
            st.json({
                "trader_state": r.trader_state,
                "action": r.action,
                "post_outcome": r.post_outcome,
            })
