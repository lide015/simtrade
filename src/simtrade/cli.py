from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from simtrade.l5_feedback import compute_performance
from simtrade.l6_learning import attribution_by_tag, attribution_cross
from simtrade.platform import boot, weekly_discovery_report


def _cmd_init(args: argparse.Namespace) -> int:
    ctx = boot(db_path=args.db, with_market=False)
    print(f"Initialized database at {args.db}")
    return 0


def _cmd_perf(args: argparse.Namespace) -> int:
    ctx = boot(db_path=args.db, with_market=False)
    stats = compute_performance(ctx.conn)
    print(json.dumps(stats.to_dict(), indent=2))
    return 0


def _cmd_attribution(args: argparse.Namespace) -> int:
    ctx = boot(db_path=args.db, with_market=False)
    records = ctx.decisions.completed()
    if args.cross:
        result = attribution_cross(records, dim_a=args.dim, dim_b=args.cross)
        result = {f"{k[0]}|{k[1]}": v for k, v in result.items()}
    else:
        result = attribution_by_tag(records, dim=args.dim)
    print(json.dumps(result, indent=2))
    return 0


def _cmd_reconcile(args: argparse.Namespace) -> int:
    ctx = boot(db_path=args.db, with_market=False)
    n = ctx.reconciler.run_once()
    print(f"Filled {n} pending outcomes.")
    return 0


def _cmd_discover(args: argparse.Namespace) -> int:
    ctx = boot(db_path=args.db, with_market=False)
    report = weekly_discovery_report(ctx)
    print(json.dumps(report, indent=2))
    return 0


def _cmd_explain(args: argparse.Namespace) -> int:
    import os
    from simtrade.l7_discovery import L7AgentExplainer

    if not os.environ.get("GEMINI_API_KEY") and not os.environ.get("GOOGLE_API_KEY"):
        if sys.stdin.isatty():
            print(
                "GEMINI_API_KEY not set. Get a free key (no credit card) at "
                "https://aistudio.google.com/apikey",
                file=sys.stderr,
            )
            try:
                key = input("Paste your Gemini API key here: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nAborted.", file=sys.stderr)
                return 1
            if not key:
                print("No key provided.", file=sys.stderr)
                return 1
            os.environ["GEMINI_API_KEY"] = key
            print(
                "Key accepted for this session. To make it permanent run "
                "[Environment]::SetEnvironmentVariable('GEMINI_API_KEY', "
                "'<your_key>', 'User') in PowerShell.",
                file=sys.stderr,
            )
        else:
            print(
                "ERROR: GEMINI_API_KEY env var not set. Get a free key at "
                "https://aistudio.google.com/apikey, then in PowerShell run: "
                "$env:GEMINI_API_KEY = 'AIzaSy...'",
                file=sys.stderr,
            )
            return 1

    ctx = boot(db_path=args.db, with_market=False)
    report = weekly_discovery_report(ctx)
    explainer = L7AgentExplainer(model=args.model)
    result = explainer.explain(report)
    print(result.markdown)
    print(f"\n---\n[{result.usage_summary()}]", file=sys.stderr)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="simtrade")
    parser.add_argument("--db", default="data/simtrade.db", help="SQLite database path")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_init = sub.add_parser("init", help="initialize the database")
    p_init.set_defaults(func=_cmd_init)

    p_perf = sub.add_parser("perf", help="show L5 performance stats")
    p_perf.set_defaults(func=_cmd_perf)

    p_attr = sub.add_parser("attribution", help="L6 tag attribution")
    p_attr.add_argument("--dim", default="setup_type")
    p_attr.add_argument("--cross", help="second dimension for cross attribution")
    p_attr.set_defaults(func=_cmd_attribution)

    p_rec = sub.add_parser("reconcile", help="run 24h outcome fill")
    p_rec.set_defaults(func=_cmd_reconcile)

    p_disc = sub.add_parser("discover", help="L7 weekly discovery report")
    p_disc.set_defaults(func=_cmd_discover)

    p_exp = sub.add_parser(
        "explain", help="L7 discovery report + Gemini coaching analysis"
    )
    p_exp.add_argument("--model", default="gemini-2.5-flash")
    p_exp.set_defaults(func=_cmd_explain)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
