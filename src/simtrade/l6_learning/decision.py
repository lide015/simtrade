from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

from simtrade.l6_learning.tags import CONFIDENCE_RANGE, validate_reasoning_tags


@dataclass
class DecisionRecord:
    id: str
    ts: str
    symbol: str
    market_snapshot: dict[str, Any]
    trader_state: dict[str, Any]
    action: dict[str, Any]
    post_outcome: dict[str, Any] | None = None
    is_locked: bool = True

    @classmethod
    def new(
        cls,
        symbol: str,
        market_snapshot: dict,
        trader_state: dict,
        action: dict,
        ts: datetime | None = None,
    ) -> "DecisionRecord":
        return cls(
            id=str(uuid.uuid4()),
            ts=(ts or datetime.now(tz=timezone.utc)).isoformat(),
            symbol=symbol,
            market_snapshot=market_snapshot,
            trader_state=trader_state,
            action=action,
        )


class DecisionStore:
    """Append-only store with locking (README §3.2.2 rule 3).

    Pre-trade fields freeze on insert; only post_outcome can be filled
    later by the reconciler. No retroactive edits — that's the whole point.
    """

    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn

    def insert(self, rec: DecisionRecord) -> None:
        self._validate_pretrade(rec)
        self.conn.execute(
            """
            INSERT INTO decisions
            (id, ts, symbol, market_snapshot, trader_state, action, post_outcome, is_locked)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                rec.id,
                rec.ts,
                rec.symbol,
                json.dumps(rec.market_snapshot),
                json.dumps(rec.trader_state),
                json.dumps(rec.action),
                json.dumps(rec.post_outcome) if rec.post_outcome else None,
                1 if rec.is_locked else 0,
            ),
        )
        self.conn.commit()

    def fill_outcome(self, decision_id: str, outcome: dict) -> None:
        existing = self.get(decision_id)
        if existing is None:
            raise KeyError(decision_id)
        if existing.post_outcome is not None:
            raise ValueError(
                f"decision {decision_id} already has post_outcome — locked (rule §3.2.2 #3)"
            )
        self.conn.execute(
            "UPDATE decisions SET post_outcome=? WHERE id=?",
            (json.dumps(outcome), decision_id),
        )
        self.conn.commit()

    def get(self, decision_id: str) -> DecisionRecord | None:
        cur = self.conn.execute("SELECT * FROM decisions WHERE id=?", (decision_id,))
        row = cur.fetchone()
        return _row_to_record(row) if row else None

    def pending_outcomes(self) -> list[DecisionRecord]:
        cur = self.conn.execute("SELECT * FROM decisions WHERE post_outcome IS NULL")
        return [_row_to_record(r) for r in cur.fetchall()]

    def all(self) -> list[DecisionRecord]:
        cur = self.conn.execute("SELECT * FROM decisions ORDER BY ts ASC")
        return [_row_to_record(r) for r in cur.fetchall()]

    def completed(self) -> list[DecisionRecord]:
        cur = self.conn.execute(
            "SELECT * FROM decisions WHERE post_outcome IS NOT NULL ORDER BY ts ASC"
        )
        return [_row_to_record(r) for r in cur.fetchall()]

    @staticmethod
    def _validate_pretrade(rec: DecisionRecord) -> None:
        issues = validate_reasoning_tags(rec.trader_state.get("reasoning_tags", {}))
        if issues:
            raise ValueError(f"invalid reasoning_tags: {issues}")
        confidence = rec.trader_state.get("confidence")
        if confidence not in CONFIDENCE_RANGE:
            raise ValueError(f"confidence must be 1..5, got {confidence!r}")
        for field_name in ("risk_amount", "sl_distance_R", "tp_distance_R"):
            if field_name not in rec.action:
                raise ValueError(
                    f"action.{field_name} is required (R-multiple discipline §3.3.2)"
                )


def _row_to_record(row: sqlite3.Row) -> DecisionRecord:
    return DecisionRecord(
        id=row["id"],
        ts=row["ts"],
        symbol=row["symbol"],
        market_snapshot=json.loads(row["market_snapshot"]),
        trader_state=json.loads(row["trader_state"]),
        action=json.loads(row["action"]),
        post_outcome=json.loads(row["post_outcome"]) if row["post_outcome"] else None,
        is_locked=bool(row["is_locked"]),
    )
