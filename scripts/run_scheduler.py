"""APScheduler entrypoint per README §6.4.

Hourly: reconcile pending 24h outcomes.
Weekly (Sunday 08:00 UTC): full Discovery report.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from apscheduler.schedulers.blocking import BlockingScheduler  # noqa: E402
from apscheduler.triggers.cron import CronTrigger  # noqa: E402
from apscheduler.triggers.interval import IntervalTrigger  # noqa: E402

from simtrade.platform import boot, weekly_discovery_report  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("simtrade.scheduler")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="data/simtrade.db")
    args = parser.parse_args()
    ctx = boot(db_path=args.db, with_market=False)

    sched = BlockingScheduler(timezone="UTC")

    def _reconcile_job() -> None:
        n = ctx.reconciler.run_once()
        log.info("reconcile filled %d outcomes", n)

    def _discovery_job() -> None:
        report = weekly_discovery_report(ctx)
        log.info("discovery report:\n%s", json.dumps(report, indent=2))

    sched.add_job(_reconcile_job, IntervalTrigger(hours=1), id="reconcile")
    sched.add_job(_discovery_job, CronTrigger(day_of_week="sun", hour=8), id="discovery")
    log.info("scheduler started: reconcile=hourly, discovery=sun 08:00 UTC")
    sched.start()
    return 0


if __name__ == "__main__":
    sys.exit(main())
