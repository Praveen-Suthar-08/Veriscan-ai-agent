"""SQLite persistence and audit logging layer for VeriScan.

Manages cases, findings, human reviewer decisions, and an immutable audit trail.
"""

from __future__ import annotations
import sqlite3
import json
from pathlib import Path
from typing import List, Dict, Optional, Any
from datetime import datetime, timezone

from veriscan.schemas import CaseReport, Finding, ReviewerDecision

DB_PATH = Path(__file__).parent.parent / "veriscan.db"


def init_db(db_path: Path = DB_PATH) -> None:
    """Initialize database tables if they do not exist."""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS cases (
        case_id TEXT PRIMARY KEY,
        timestamp TEXT NOT NULL,
        triage TEXT NOT NULL,
        risk_score REAL NOT NULL,
        ocr_engine TEXT NOT NULL,
        narrative TEXT,
        report_json TEXT NOT NULL
    );
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS reviewer_decisions (
        finding_id TEXT PRIMARY KEY,
        case_id TEXT NOT NULL,
        status TEXT NOT NULL,
        notes TEXT,
        reviewer_id TEXT,
        updated_at TEXT NOT NULL,
        FOREIGN KEY (case_id) REFERENCES cases(case_id) ON DELETE CASCADE
    );
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS audit_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        case_id TEXT NOT NULL,
        finding_id TEXT,
        action TEXT NOT NULL,
        details TEXT,
        timestamp TEXT NOT NULL
    );
    """)

    conn.commit()
    conn.close()


class CaseStore:
    """Data access object for case history and reviewer interactions."""

    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        init_db(self.db_path)

    def _get_conn(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def save_case(self, report: CaseReport) -> None:
        """Save or update a case screening report."""
        conn = self._get_conn()
        cur = conn.cursor()
        report_json = report.model_dump_json()

        cur.execute("""
        INSERT OR REPLACE INTO cases (case_id, timestamp, triage, risk_score, ocr_engine, narrative, report_json)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            report.case_id,
            report.timestamp,
            report.triage,
            report.risk_score,
            report.ocr_engine,
            report.narrative,
            report_json
        ))

        # Log case creation
        cur.execute("""
        INSERT INTO audit_log (case_id, action, details, timestamp)
        VALUES (?, ?, ?, ?)
        """, (
            report.case_id,
            "CASE_SCREENED",
            f"Triage={report.triage}, Risk={report.risk_score:.2f}, Findings={len(report.findings)}",
            datetime.now(timezone.utc).isoformat()
        ))

        conn.commit()
        conn.close()

    def get_case(self, case_id: str) -> Optional[CaseReport]:
        """Retrieve CaseReport by case_id."""
        conn = self._get_conn()
        cur = conn.cursor()
        cur.execute("SELECT report_json FROM cases WHERE case_id = ?", (case_id,))
        row = cur.fetchone()
        conn.close()

        if not row:
            return None
        data = json.loads(row[0])
        return CaseReport(**data)

    def list_cases(self) -> List[Dict[str, Any]]:
        """List all processed cases with summary info."""
        conn = self._get_conn()
        cur = conn.cursor()
        cur.execute("""
        SELECT case_id, timestamp, triage, risk_score, ocr_engine FROM cases
        ORDER BY timestamp DESC
        """)
        rows = cur.fetchall()
        conn.close()

        cases = []
        for r in rows:
            raw_time = r[1]
            try:
                dt = datetime.fromisoformat(raw_time)
                formatted_time = dt.strftime("%d %b %Y, %H:%M")
            except Exception:
                formatted_time = str(raw_time)[:16]

            cases.append({
                "case_id": r[0],
                "timestamp": formatted_time,
                "triage": r[2],
                "risk_score": f"{float(r[3] or 0.0):.2f}",
                "ocr_engine": r[4]
            })
        return cases

    def record_decision(
        self,
        case_id: str,
        finding_id: str,
        status: str,
        notes: str = "",
        reviewer_id: str = "reviewer_1"
    ) -> ReviewerDecision:
        """Record human reviewer decision (CONFIRMED / DISMISSED / NEEDS_REUPLOAD)."""
        now = datetime.now(timezone.utc).isoformat()
        conn = self._get_conn()
        cur = conn.cursor()

        cur.execute("""
        INSERT OR REPLACE INTO reviewer_decisions (finding_id, case_id, status, notes, reviewer_id, updated_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """, (finding_id, case_id, status, notes, reviewer_id, now))

        cur.execute("""
        INSERT INTO audit_log (case_id, finding_id, action, details, timestamp)
        VALUES (?, ?, ?, ?, ?)
        """, (
            case_id,
            finding_id,
            f"DECISION_{status.upper()}",
            f"Notes: {notes} | Reviewer: {reviewer_id}",
            now
        ))

        conn.commit()
        conn.close()

        return ReviewerDecision(
            finding_id=finding_id,
            case_id=case_id,
            status=status,
            notes=notes,
            reviewer_id=reviewer_id,
            updated_at=now
        )

    def get_decisions(self, case_id: str) -> Dict[str, ReviewerDecision]:
        """Fetch all reviewer decisions for a case."""
        conn = self._get_conn()
        cur = conn.cursor()
        cur.execute("""
        SELECT finding_id, case_id, status, notes, reviewer_id, updated_at
        FROM reviewer_decisions WHERE case_id = ?
        """, (case_id,))
        rows = cur.fetchall()
        conn.close()

        decisions = {}
        for r in rows:
            decisions[r[0]] = ReviewerDecision(
                finding_id=r[0],
                case_id=r[1],
                status=r[2],
                notes=r[3],
                reviewer_id=r[4],
                updated_at=r[5]
            )
        return decisions

    def get_audit_trail(self, case_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Retrieve audit log entries."""
        conn = self._get_conn()
        cur = conn.cursor()
        if case_id:
            cur.execute("""
            SELECT id, case_id, finding_id, action, details, timestamp
            FROM audit_log WHERE case_id = ? ORDER BY id DESC
            """, (case_id,))
        else:
            cur.execute("""
            SELECT id, case_id, finding_id, action, details, timestamp
            FROM audit_log ORDER BY id DESC LIMIT 100
            """)
        rows = cur.fetchall()
        conn.close()

        entries = []
        for r in rows:
            entries.append({
                "id": r[0],
                "case_id": r[1],
                "finding_id": r[2],
                "action": r[3],
                "details": r[4],
                "timestamp": r[5]
            })
        return entries

    def delete_case(self, case_id: str) -> bool:
        """Delete case and associated decisions for privacy compliance."""
        conn = self._get_conn()
        cur = conn.cursor()
        cur.execute("DELETE FROM reviewer_decisions WHERE case_id = ?", (case_id,))
        cur.execute("DELETE FROM cases WHERE case_id = ?", (case_id,))
        cur.execute("""
        INSERT INTO audit_log (case_id, action, details, timestamp)
        VALUES (?, ?, ?, ?)
        """, (case_id, "CASE_DELETED", "Case permanently purged for privacy compliance.", datetime.now(timezone.utc).isoformat()))
        conn.commit()
        conn.close()
        return True
