"""
FIntelDB.py — F-Intel Unified Database Layer
===============================================
Core database connection pool and schema management for the F-Intel
historical data store. Supports both TimescaleDB (PostgreSQL) for
production and DuckDB for embedded/zero-install usage.

Schema Migration:
    Tables are created on first connection if they don't exist.
    Uses a `_schema_version` metadata table to track migrations.

Usage:
    from FIntelDB import get_db

    db = get_db()                      # returns singleton
    db.execute("INSERT INTO spot_ticks ...")
    rows = db.fetchall("SELECT * FROM spot_ticks WHERE ...")
"""

from __future__ import annotations

import io
import os
import sys
import logging
import threading
from pathlib import Path
from datetime import datetime
from typing import Any, Optional, List, Tuple

if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')  # type: ignore
    except (AttributeError, Exception):
        pass

_log = logging.getLogger("fintel.db")

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

try:
    import config as _cfg
    _DB_ENGINE = _cfg.get("db_engine", "duckdb")                # 'duckdb' or 'timescaledb'
    _PG_DSN    = _cfg.get("db_dsn", "postgresql://fintel:fintel@localhost:5432/fintel")
    _DB_PATH   = _cfg.get("db_path", str(Path(__file__).parent / "data" / "fintel_history.duckdb"))
except Exception:
    _DB_ENGINE = "duckdb"
    _PG_DSN    = "postgresql://fintel:fintel@localhost:5432/fintel"
    _DB_PATH   = str(Path(__file__).parent / "data" / "fintel_history.duckdb")

SCHEMA_VERSION = 1

# ──────────────────────────────────────────────────────────────────────────────
# SQL Schema Definitions (portable across DuckDB and PostgreSQL)
# ──────────────────────────────────────────────────────────────────────────────

# DuckDB-compatible schema (no hypertable calls — those are TimescaleDB-only)
_DUCKDB_SCHEMA = """
-- Schema metadata
CREATE TABLE IF NOT EXISTS _schema_version (
    version     INTEGER NOT NULL,
    applied_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 1. SPOT TICKS (from WebSocket, throttled to 1/sec)
CREATE TABLE IF NOT EXISTS spot_ticks (
    ts          TIMESTAMP NOT NULL,
    spot        REAL      NOT NULL,
    tick_count  INTEGER
);

-- 2. OPTION CHAIN SNAPSHOTS (per-strike, per-snapshot)
CREATE TABLE IF NOT EXISTS chain_snapshots (
    ts          TIMESTAMP NOT NULL,
    expiry      DATE      NOT NULL,
    strike      REAL      NOT NULL,
    opt_type    VARCHAR(2) NOT NULL,
    ltp         REAL,
    iv          REAL,
    oi          BIGINT,
    volume      BIGINT,
    delta       REAL,
    gamma       REAL,
    theta       REAL,
    vega        REAL
);

-- 3. GEX SNAPSHOTS (from _gex_refresh_loop, ~3s interval)
CREATE TABLE IF NOT EXISTS gex_snapshots (
    ts                TIMESTAMP NOT NULL,
    net_gex           REAL,
    gex_flip_point    REAL,
    regime            VARCHAR(32),
    atm_concentration REAL,
    score             REAL,
    direction         VARCHAR(16),
    oi_surge_bias     REAL
);

-- 4. VOLATILITY METRICS (from RealizedVolEngine, ~1/min)
CREATE TABLE IF NOT EXISTS vol_snapshots (
    ts              TIMESTAMP NOT NULL,
    spot            REAL,
    atm_iv          REAL,
    rv_5d           REAL,
    rv_20d          REAL,
    rv_60d          REAL,
    consensus_rv    REAL,
    vrp             REAL,
    ivp             REAL,
    ivr             REAL,
    ver             REAL,
    vov             REAL,
    regime          VARCHAR(32),
    har_forecast_1d REAL,
    har_forecast_5d REAL,
    jump_ratio      REAL,
    forward_vrp     REAL,
    skew_ratio      REAL
);

-- 5. OI VELOCITY SNAPSHOTS
CREATE TABLE IF NOT EXISTS oi_velocity (
    ts              TIMESTAMP NOT NULL,
    strike          REAL      NOT NULL,
    opt_type        VARCHAR(2) NOT NULL,
    oi_abs          BIGINT,
    vel_15m         REAL,
    vel_5m          REAL,
    accel           REAL,
    pct_change      REAL,
    classification  VARCHAR(32)
);

-- 6. SIGNAL LOG (from SignalMemory / MasterSignalEngine)
CREATE TABLE IF NOT EXISTS signals (
    ts              TIMESTAMP NOT NULL,
    signal_id       VARCHAR(64),
    source          VARCHAR(64) NOT NULL,
    direction       VARCHAR(16),
    action          VARCHAR(64),
    strike          REAL,
    entry_price     REAL,
    sl_spot         REAL,
    t1_spot         REAL,
    t2_spot         REAL,
    score           REAL,
    status          VARCHAR(16) DEFAULT 'ACTIVE',
    outcome_pnl     REAL,
    context         VARCHAR
);

-- 7. ALERTS (replaces alerts.jsonl)
CREATE TABLE IF NOT EXISTS alerts (
    ts          TIMESTAMP NOT NULL,
    source      VARCHAR(64) NOT NULL,
    level       VARCHAR(16) NOT NULL,
    title       VARCHAR(256),
    body        VARCHAR,
    data        VARCHAR
);

-- 8. HESTON CALIBRATION SNAPSHOTS
CREATE TABLE IF NOT EXISTS heston_params (
    ts          TIMESTAMP NOT NULL,
    v0          REAL,
    kappa       REAL,
    theta       REAL,
    sigma       REAL,
    rho         REAL,
    fit_error   REAL
);

-- Indexes for common query patterns (DuckDB auto-indexes but explicit helps)
CREATE INDEX IF NOT EXISTS idx_spot_ts ON spot_ticks(ts);
CREATE INDEX IF NOT EXISTS idx_chain_ts ON chain_snapshots(ts);
CREATE INDEX IF NOT EXISTS idx_chain_ts_strike ON chain_snapshots(ts, strike, opt_type);
CREATE INDEX IF NOT EXISTS idx_gex_ts ON gex_snapshots(ts);
CREATE INDEX IF NOT EXISTS idx_vol_ts ON vol_snapshots(ts);
CREATE INDEX IF NOT EXISTS idx_oi_ts ON oi_velocity(ts);
CREATE INDEX IF NOT EXISTS idx_oi_ts_strike ON oi_velocity(ts, strike, opt_type);
CREATE INDEX IF NOT EXISTS idx_signals_ts ON signals(ts);
CREATE INDEX IF NOT EXISTS idx_alerts_ts ON alerts(ts);
CREATE INDEX IF NOT EXISTS idx_heston_ts ON heston_params(ts);
"""


# ──────────────────────────────────────────────────────────────────────────────
# Database Adapter Interface
# ──────────────────────────────────────────────────────────────────────────────

class FIntelDB:
    """
    Unified database interface. Wraps either DuckDB or PostgreSQL/TimescaleDB.
    Thread-safe: uses a lock for write operations.
    """

    def __init__(self, engine: str = _DB_ENGINE):
        self._engine = engine
        self._conn: Any = None
        self._pool: Any = None
        self._lock = threading.Lock()
        self._closed = False
        self._connect()
        self._ensure_schema()

    def _connect(self):
        """Establish database connection."""
        if self._engine == "duckdb":
            try:
                import duckdb
                db_dir = os.path.dirname(_DB_PATH)
                os.makedirs(db_dir, exist_ok=True)
                self._conn = duckdb.connect(_DB_PATH)
                _log.info(f"Connected to DuckDB: {_DB_PATH}")
            except ImportError:
                _log.warning("DuckDB not installed, falling back to SQLite")
                self._engine = "sqlite"
                self._connect()

        elif self._engine == "sqlite":
            import sqlite3
            sqlite_path = _DB_PATH.replace(".duckdb", ".sqlite")
            db_dir = os.path.dirname(sqlite_path)
            os.makedirs(db_dir, exist_ok=True)
            self._conn = sqlite3.connect(sqlite_path, check_same_thread=False)
            _log.info(f"Connected to SQLite: {sqlite_path}")

        elif self._engine == "timescaledb":
            try:
                import psycopg2
                import psycopg2.pool
            except ImportError:
                _log.error("psycopg2 not installed. Run: pip install psycopg2-binary")
                raise

            self._pool = psycopg2.pool.ThreadedConnectionPool(1, 5, _PG_DSN)
            self._conn = self._pool.getconn()
            self._conn.autocommit = True
            _log.info(f"Connected to TimescaleDB: {_PG_DSN.split('@')[1]}")

        else:
            raise ValueError(f"Unsupported DB engine: {self._engine}")

    def _ensure_schema(self):
        """Create tables if they don't exist."""
        with self._lock:
            if self._engine in ("duckdb", "sqlite"):
                cursor = self._conn.cursor() if self._engine == "sqlite" else self._conn
                for stmt in _DUCKDB_SCHEMA.split(";"):
                    stmt = stmt.strip()
                    if stmt:
                        cursor.execute(stmt)
                if self._engine == "sqlite":
                    self._conn.commit()

                # Check if schema version is set
                result = self._conn.execute(
                    "SELECT COUNT(*) FROM _schema_version"
                ).fetchone()
                if result[0] == 0:
                    self._conn.execute(
                        "INSERT INTO _schema_version (version) VALUES (?)",
                        [SCHEMA_VERSION]
                    )
                    if self._engine == "sqlite":
                        self._conn.commit()
                _log.info(f"Schema v{SCHEMA_VERSION} ready ({self._engine})")

            elif self._engine == "timescaledb":
                cursor = self._conn.cursor()
                # Use PostgreSQL-compatible schema with TimescaleDB hypertables
                pg_schema = _DUCKDB_SCHEMA.replace("VARCHAR", "TEXT")
                for stmt in pg_schema.split(";"):
                    stmt = stmt.strip()
                    if stmt:
                        cursor.execute(stmt)

                # Create hypertables (TimescaleDB-specific)
                hypertables = [
                    "spot_ticks", "chain_snapshots", "gex_snapshots",
                    "vol_snapshots", "oi_velocity", "signals",
                    "alerts", "heston_params"
                ]
                for table in hypertables:
                    try:
                        cursor.execute(
                            f"SELECT create_hypertable('{table}', 'ts', "
                            f"if_not_exists => TRUE)"
                        )
                    except Exception as e:
                        _log.debug(f"Hypertable {table}: {e}")

                # Compression policies
                for table in ["chain_snapshots", "oi_velocity", "spot_ticks", "gex_snapshots"]:
                    try:
                        cursor.execute(
                            f"ALTER TABLE {table} SET (timescaledb.compress)"
                        )
                        cursor.execute(
                            f"SELECT add_compression_policy('{table}', "
                            f"INTERVAL '2 days', if_not_exists => TRUE)"
                        )
                    except Exception as e:
                        _log.debug(f"Compression {table}: {e}")

                cursor.close()
                _log.info(f"Schema v{SCHEMA_VERSION} ready (TimescaleDB)")

    # ── Query Methods ────────────────────────────────────────────────────────

    def execute(self, sql: str, params: tuple = ()) -> Any:
        """Execute a single SQL statement."""
        with self._lock:
            if self._engine == "duckdb":
                return self._conn.execute(sql, list(params) if params else [])
            elif self._engine == "sqlite":
                cursor = self._conn.cursor()
                cursor.execute(sql, params)
                self._conn.commit()
                return cursor
            else:
                cursor = self._conn.cursor()
                cursor.execute(sql, params)
                return cursor

    def executemany(self, sql: str, params_list: List[tuple]) -> None:
        """Execute a SQL statement with many parameter sets (batch insert)."""
        if not params_list:
            return
        with self._lock:
            if self._engine == "duckdb":
                self._conn.executemany(sql, [list(p) for p in params_list])
            elif self._engine == "sqlite":
                cursor = self._conn.cursor()
                cursor.executemany(sql, params_list)
                self._conn.commit()
                cursor.close()
            else:
                cursor = self._conn.cursor()
                cursor.executemany(sql, params_list)
                cursor.close()

    def fetchall(self, sql: str, params: tuple = ()) -> List[tuple]:
        """Execute and fetch all rows."""
        with self._lock:
            if self._engine == "duckdb":
                result = self._conn.execute(sql, list(params) if params else [])
                return result.fetchall()
            elif self._engine == "sqlite":
                cursor = self._conn.cursor()
                cursor.execute(sql, params)
                rows = cursor.fetchall()
                cursor.close()
                return rows
            else:
                cursor = self._conn.cursor()
                cursor.execute(sql, params)
                rows = cursor.fetchall()
                cursor.close()
                return rows

    def fetchone(self, sql: str, params: tuple = ()) -> Optional[tuple]:
        """Execute and fetch one row."""
        with self._lock:
            if self._engine == "duckdb":
                result = self._conn.execute(sql, list(params) if params else [])
                return result.fetchone()
            elif self._engine == "sqlite":
                cursor = self._conn.cursor()
                cursor.execute(sql, params)
                row = cursor.fetchone()
                cursor.close()
                return row
            else:
                cursor = self._conn.cursor()
                cursor.execute(sql, params)
                row = cursor.fetchone()
                cursor.close()
                return row

    def fetchdf(self, sql: str, params: tuple = ()):
        """Execute and return a pandas DataFrame."""
        import pandas as pd
        with self._lock:
            if self._engine == "duckdb":
                result = self._conn.execute(sql, list(params) if params else [])
                return result.fetchdf()
            else:
                return pd.read_sql_query(sql, self._conn, params=params)

    def close(self):
        """Close the database connection."""
        if self._closed:
            return
        self._closed = True
        with self._lock:
            if self._engine in ("duckdb", "sqlite"):
                self._conn.close()
            elif self._engine == "timescaledb":
                self._pool.putconn(self._conn)
                self._pool.closeall()
        _log.info("Database connection closed")

    @property
    def engine(self) -> str:
        return self._engine


# ──────────────────────────────────────────────────────────────────────────────
# Singleton access
# ──────────────────────────────────────────────────────────────────────────────

_instance: Optional[FIntelDB] = None
_instance_lock = threading.Lock()


def get_db(engine: Optional[str] = None) -> FIntelDB:
    """
    Return the singleton FIntelDB instance.
    Call once at startup; subsequent calls return the same instance.
    """
    global _instance
    with _instance_lock:
        if _instance is None or _instance._closed:
            _instance = FIntelDB(engine or _DB_ENGINE)
        return _instance


# ──────────────────────────────────────────────────────────────────────────────
# CLI: Quick test / schema creation
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    db = get_db()
    print(f"[OK] FIntelDB initialized (engine={db.engine})")

    # Quick smoke test
    db.execute(
        "INSERT INTO spot_ticks (ts, spot, tick_count) VALUES (?, ?, ?)",
        (datetime.now(), 23456.7, 1)
    )
    row = db.fetchone("SELECT * FROM spot_ticks ORDER BY ts DESC LIMIT 1")
    print(f"[OK] Test row: {row}")

    # Table summary
    tables = [
        "spot_ticks", "chain_snapshots", "gex_snapshots",
        "vol_snapshots", "oi_velocity", "signals", "alerts", "heston_params"
    ]
    for t in tables:
        row = db.fetchone(f"SELECT COUNT(*) FROM {t}")
        count = row[0] if row else 0
        print(f"  {t}: {count} rows")

    db.close()
    print("[OK] Done")
