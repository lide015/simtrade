from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone

from simtrade.l6_learning.decision import DecisionRecord
from simtrade.l7_discovery.scanner import Finding


@dataclass
class SuggestedExperiment:
    id: str
    proposed_at: str
    hypothesis: str
    conditions: dict
    target_n: int
    status: str = "proposed"
    conclusion: str | None = None
    related_decision_ids: list[str] = field(default_factory=list)

    @classmethod
    def from_finding(cls, finding: Finding, target_n: int = 5) -> "SuggestedExperiment":
        a_dim, a_val = finding.label_a
        b_dim, b_val = finding.label_b
        hypothesis = (
            f"You perform unusually well when {a_dim}={a_val} AND {b_dim}={b_val} "
            f"(observed win rate {finding.win_rate_subset:.0%} vs baseline "
            f"{finding.win_rate_baseline:.0%}, n={finding.n_subset})."
        )
        return cls(
            id=str(uuid.uuid4()),
            proposed_at=datetime.now(tz=timezone.utc).isoformat(),
            hypothesis=hypothesis,
            conditions={a_dim: a_val, b_dim: b_val},
            target_n=target_n,
        )


class ExperimentStore:
    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn

    def propose(self, exp: SuggestedExperiment) -> None:
        self.conn.execute(
            """
            INSERT INTO experiments
            (id, proposed_at, hypothesis, conditions, target_n, status,
             conclusion, related_decision_ids)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                exp.id,
                exp.proposed_at,
                exp.hypothesis,
                json.dumps(exp.conditions),
                exp.target_n,
                exp.status,
                exp.conclusion,
                json.dumps(exp.related_decision_ids),
            ),
        )
        self.conn.commit()

    def list_active(self) -> list[SuggestedExperiment]:
        cur = self.conn.execute(
            "SELECT * FROM experiments WHERE status IN ('proposed', 'active')"
        )
        return [_row_to_exp(r) for r in cur.fetchall()]

    def evaluate(
        self, exp: SuggestedExperiment, records: list[DecisionRecord]
    ) -> SuggestedExperiment:
        matching = [
            r for r in records
            if self._matches(r, exp.conditions) and r.post_outcome is not None
            and _parse_iso(r.ts) >= _parse_iso(exp.proposed_at)
        ]
        if len(matching) < exp.target_n:
            return exp
        wins = sum(1 for r in matching if (r.post_outcome or {}).get("pnl_R", 0) > 0)
        win_rate = wins / len(matching)
        avg_R = sum((r.post_outcome or {}).get("pnl_R", 0) for r in matching) / len(matching)
        exp.status = "concluded"
        exp.related_decision_ids = [r.id for r in matching]
        exp.conclusion = (
            f"Tested on {len(matching)} new trades: "
            f"win {win_rate:.0%}, avg {avg_R:+.2f}R"
        )
        self.conn.execute(
            "UPDATE experiments SET status=?, conclusion=?, related_decision_ids=? WHERE id=?",
            (
                exp.status,
                exp.conclusion,
                json.dumps(exp.related_decision_ids),
                exp.id,
            ),
        )
        self.conn.commit()
        return exp

    @staticmethod
    def _matches(rec: DecisionRecord, conditions: dict) -> bool:
        tags = rec.trader_state.get("reasoning_tags", {})
        for dim, val in conditions.items():
            t = tags.get(dim)
            if isinstance(t, list):
                if val not in (str(x) for x in t):
                    return False
            elif str(t) != str(val):
                return False
        return True


def _row_to_exp(row: sqlite3.Row) -> SuggestedExperiment:
    return SuggestedExperiment(
        id=row["id"],
        proposed_at=row["proposed_at"],
        hypothesis=row["hypothesis"],
        conditions=json.loads(row["conditions"]),
        target_n=row["target_n"],
        status=row["status"],
        conclusion=row["conclusion"],
        related_decision_ids=(
            json.loads(row["related_decision_ids"]) if row["related_decision_ids"] else []
        ),
    )


def _parse_iso(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))
