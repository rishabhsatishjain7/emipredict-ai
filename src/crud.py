"""SQLite-backed CRUD layer for applicant records (admin page)."""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone

import pandas as pd

from config import RAW_CATEGORICAL, RAW_FEATURES, ROOT

DB_PATH = ROOT / "data" / "applicants.db"
PRED_COLS = ["pred_eligibility", "pred_max_emi", "confidence"]
EDITABLE = RAW_FEATURES + PRED_COLS


@contextmanager
def connect():
    DB_PATH.parent.mkdir(exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    try:
        yield con
        con.commit()
    finally:
        con.close()


def init_db():
    cols = ", ".join(f"{c} {'TEXT' if c in RAW_CATEGORICAL else 'REAL'}" for c in RAW_FEATURES)
    with connect() as con:
        con.execute(
            f"""CREATE TABLE IF NOT EXISTS applicants (
                id INTEGER PRIMARY KEY AUTOINCREMENT, {cols},
                pred_eligibility TEXT, pred_max_emi REAL, confidence REAL,
                created_at TEXT, updated_at TEXT)"""
        )


def count() -> int:
    with connect() as con:
        return con.execute("SELECT COUNT(*) FROM applicants").fetchone()[0]


def _now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


# ---- Create
def create_many(df: pd.DataFrame) -> int:
    df = df.copy()
    keep = [c for c in EDITABLE if c in df.columns]
    df = df[keep]
    df["created_at"] = df["updated_at"] = _now()
    with connect() as con:
        df.to_sql("applicants", con, if_exists="append", index=False)
    return len(df)


def create(record: dict) -> int:
    return create_many(pd.DataFrame([record]))


# ---- Read
def read(search: dict | None = None, limit: int = 500, offset: int = 0) -> pd.DataFrame:
    where, params = [], []
    for col, val in (search or {}).items():
        if val in (None, "", []):
            continue
        if col in RAW_CATEGORICAL + ["pred_eligibility"]:
            where.append(f"{col} IN ({','.join('?' * len(val))})"); params += list(val)
        elif isinstance(val, tuple):
            where.append(f"{col} BETWEEN ? AND ?"); params += [val[0], val[1]]
    sql = "SELECT * FROM applicants" + (" WHERE " + " AND ".join(where) if where else "")
    sql += " ORDER BY id DESC LIMIT ? OFFSET ?"
    with connect() as con:
        return pd.read_sql_query(sql, con, params=params + [limit, offset])


# ---- Update
def update(record_id: int, fields: dict) -> None:
    fields = {k: v for k, v in fields.items() if k in EDITABLE}
    if not fields:
        return
    sets = ", ".join(f"{k}=?" for k in fields) + ", updated_at=?"
    with connect() as con:
        con.execute(f"UPDATE applicants SET {sets} WHERE id=?", [*fields.values(), _now(), int(record_id)])


# ---- Delete
def delete(ids: list[int]) -> int:
    if not ids:
        return 0
    with connect() as con:
        cur = con.execute(f"DELETE FROM applicants WHERE id IN ({','.join('?' * len(ids))})", [int(i) for i in ids])
        return cur.rowcount
