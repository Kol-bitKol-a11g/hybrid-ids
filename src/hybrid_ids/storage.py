"""Сховище оповіщень: таблиця alert_records (PostgreSQL у розгортанні, SQLite для
локального запуску й тестів), вердикти аналітика та службова статистика.

Рядок підключення: postgresql://user:pass@host:5432/hybrid_ids або sqlite:///шлях.
"""
from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

from .schemas import AlertRecord

COLUMNS = ("flow_id", "timestamp", "src_ip", "dst_ip", "src_port", "dst_port", "alert_type",
           "detector", "attack_class", "mitre_technique", "p_xgb", "l_ae", "severity", "verified")

SCHEMA_POSTGRES = """
CREATE TABLE IF NOT EXISTS alert_records (
    id              BIGSERIAL PRIMARY KEY,
    flow_id         UUID NOT NULL,
    "timestamp"     TIMESTAMPTZ NOT NULL DEFAULT now(),
    src_ip          INET NOT NULL,
    dst_ip          INET NOT NULL,
    src_port        INTEGER,
    dst_port        INTEGER,
    alert_type      TEXT NOT NULL,       -- signature | ml_confident | reconciled | review
    detector        TEXT NOT NULL,
    attack_class    TEXT,
    mitre_technique TEXT,
    p_xgb           REAL,
    l_ae            REAL,
    severity        SMALLINT NOT NULL DEFAULT 1,
    verified        BOOLEAN NOT NULL DEFAULT false
);
CREATE INDEX IF NOT EXISTS alert_records_timestamp_idx ON alert_records ("timestamp" DESC);
CREATE TABLE IF NOT EXISTS alert_verdicts (
    flow_id UUID NOT NULL, confirmed BOOLEAN NOT NULL, comment TEXT,
    verified_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS runtime_stats (name TEXT PRIMARY KEY, value TEXT NOT NULL);
"""
SCHEMA_SQLITE = (
    SCHEMA_POSTGRES.replace("BIGSERIAL PRIMARY KEY", "INTEGER PRIMARY KEY AUTOINCREMENT")
    .replace("TIMESTAMPTZ NOT NULL DEFAULT now()", "TEXT NOT NULL")
    .replace("UUID", "TEXT").replace("INET", "TEXT").replace("DEFAULT false", "DEFAULT 0")
)


def _iso(value: datetime) -> str:
    """Час у UTC у форматі ISO 8601 (рядки такого формату коректно порівнюються)."""
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat()


class AlertStore:
    """Доступ до сховища оповіщень через DB-API 2.0."""

    def __init__(self, url: str = "sqlite:///data/alerts.db") -> None:
        self._lock = threading.Lock()
        self.is_postgres = url.startswith(("postgresql://", "postgres://"))
        if self.is_postgres:
            import psycopg2

            self.conn = psycopg2.connect(url)
            self.mark = "%s"
        else:
            target = url.removeprefix("sqlite:///") or ":memory:"
            if target != ":memory:":
                Path(target).parent.mkdir(parents=True, exist_ok=True)
            self.conn = sqlite3.connect(target, check_same_thread=False)
            self.mark = "?"
        self.init_schema()

    def _execute(self, sql: str, params: tuple = (), fetch: bool = False):
        with self._lock:
            cur = self.conn.cursor()
            try:
                cur.execute(sql.replace("?", self.mark), params)
            except Exception:
                self.conn.rollback()
                raise
            rows = cur.fetchall() if fetch else None
            names = [d[0] for d in cur.description] if fetch else None
            count, last_id = cur.rowcount, getattr(cur, "lastrowid", None)
            self.conn.commit()
        return [dict(zip(names, row)) for row in rows] if fetch else (count, last_id)

    def init_schema(self) -> None:
        for statement in (SCHEMA_POSTGRES if self.is_postgres else SCHEMA_SQLITE).split(";"):
            if statement.strip():
                self._execute(statement)

    def insert(self, alert: AlertRecord) -> int | None:
        """Зберігає оповіщення; повертає ідентифікатор запису."""
        values = [getattr(alert, c) for c in COLUMNS]
        values[1] = _iso(alert.timestamp)
        names = ", ".join(f'"{c}"' for c in COLUMNS)
        sql = f"INSERT INTO alert_records ({names}) VALUES ({', '.join('?' * len(COLUMNS))})"
        if self.is_postgres:
            alert.id = self._execute(sql + " RETURNING id", tuple(values), fetch=True)[0]["id"]
        else:
            alert.id = self._execute(sql, tuple(values))[1]
        return alert.id

    def query(self, time_from: datetime | None = None, time_to: datetime | None = None,
              attack_class: str | None = None, min_confidence: float = 0.0,
              limit: int = 100) -> list[dict]:
        """Оповіщення за часовим діапазоном, класом атаки та рівнем достовірності.
        Сигнатурні оповіщення без оцінки ML мають достовірність 1."""
        where, params = ["COALESCE(p_xgb, 1.0) >= ?"], [min_confidence]
        for clause, value in (('"timestamp" >= ?', time_from), ('"timestamp" <= ?', time_to)):
            if value is not None:
                where.append(clause)
                params.append(_iso(value))
        if attack_class:
            where.append("attack_class = ?")
            params.append(attack_class)
        sql = (f'SELECT * FROM alert_records WHERE {" AND ".join(where)} '
               f'ORDER BY "timestamp" DESC LIMIT ?')
        rows = self._execute(sql, (*params, int(limit)), fetch=True)
        for row in rows:
            row["verified"] = bool(row["verified"])
            row["timestamp"] = str(row["timestamp"])
            row["flow_id"] = str(row["flow_id"])
        return rows

    def verify(self, flow_id: str, confirmed: bool, comment: str | None = None) -> bool:
        """Фіксує вердикт аналітика SOC; False - якщо оповіщення з таким flow_id немає."""
        updated, _ = self._execute(
            "UPDATE alert_records SET verified = ? WHERE flow_id = ?", (True, flow_id))
        if updated:
            self._execute(
                "INSERT INTO alert_verdicts (flow_id, confirmed, comment, verified_at) VALUES (?, ?, ?, ?)",
                (flow_id, bool(confirmed), comment, _iso(datetime.now(timezone.utc))))
        return bool(updated)

    def verdict_counts(self) -> tuple[int, int]:
        """(підтверджено, спростовано) - для оцінки частки хибних оповіщень."""
        rows = self._execute(
            "SELECT confirmed, COUNT(*) AS n FROM alert_verdicts GROUP BY confirmed", fetch=True)
        counts = {bool(row["confirmed"]): row["n"] for row in rows}
        return counts.get(True, 0), counts.get(False, 0)

    def alerts_by_type(self, since: datetime) -> dict[str, int]:
        rows = self._execute(
            'SELECT alert_type, COUNT(*) AS n FROM alert_records WHERE "timestamp" >= ? '
            "GROUP BY alert_type", (_iso(since),), fetch=True)
        return {row["alert_type"]: row["n"] for row in rows}

    def save_stats(self, stats: dict) -> None:
        """Зберігає знімок службової статистики конвеєра (читає REST API)."""
        self._execute("DELETE FROM runtime_stats WHERE name = ?", ("snapshot",))
        self._execute("INSERT INTO runtime_stats (name, value) VALUES (?, ?)",
                      ("snapshot", json.dumps(stats)))

    def load_stats(self) -> dict:
        rows = self._execute("SELECT value FROM runtime_stats WHERE name = ?", ("snapshot",), fetch=True)
        return json.loads(rows[0]["value"]) if rows else {}
