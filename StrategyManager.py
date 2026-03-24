"""
StrategyManager.py
==================
SQLite-backed tracking system for live option strategies.
Extended to support full position lifecycle: entry, exit, P&L, stop-loss,
and backtest result persistence.
"""

import sqlite3
import json
import time
from datetime import datetime


class StrategyManager:
    """
    Manages the lifecycle, legs, and P&L of active and closed strategies.
    Persists backtest results for historical reference.
    """

    def __init__(self, db_path='strategies.db'):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Main strategies table (extended)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS strategies (
                id                 TEXT PRIMARY KEY,
                name               TEXT NOT NULL,
                type               TEXT NOT NULL,          -- 'CREDIT' or 'DEBIT'
                status             TEXT DEFAULT 'ACTIVE',  -- ACTIVE, CLOSED, STOPPED
                entry_premium      REAL NOT NULL,
                entry_spot         REAL DEFAULT 0,
                entry_time_str     TEXT NOT NULL,
                entry_timestamp    INTEGER NOT NULL,
                exit_premium       REAL,
                exit_spot          REAL,
                exit_time_str      TEXT,
                exit_timestamp     INTEGER,
                pnl                REAL,
                stop_loss_level    REAL,
                target_level       REAL,
                reasoning          TEXT,
                score              INTEGER,
                notes              TEXT
            )
        ''')

        # Add new columns to existing DB if upgrading from old schema
        _new_cols = [
            ('entry_spot',      'REAL DEFAULT 0'),
            ('exit_premium',    'REAL'),
            ('exit_spot',       'REAL'),
            ('exit_time_str',   'TEXT'),
            ('exit_timestamp',  'INTEGER'),
            ('pnl',             'REAL'),
            ('stop_loss_level', 'REAL'),
            ('target_level',    'REAL'),
            ('notes',           'TEXT'),
        ]
        for col, col_type in _new_cols:
            try:
                cursor.execute(f'ALTER TABLE strategies ADD COLUMN {col} {col_type}')
            except sqlite3.OperationalError:
                pass   # column already exists

        # Legs table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS strategy_legs (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                strategy_id  TEXT NOT NULL,
                opt_type     TEXT NOT NULL,   -- 'CE' or 'PE'
                action       TEXT NOT NULL,   -- 'BUY' or 'SELL'
                strike       REAL NOT NULL,
                entry_price  REAL NOT NULL,
                iv           REAL DEFAULT 0,
                lots         INTEGER DEFAULT 1,
                expiry       TEXT DEFAULT '',
                FOREIGN KEY (strategy_id) REFERENCES strategies(id) ON DELETE CASCADE
            )
        ''')

        # Add legs columns for existing DBs
        for col, col_type in [('iv', 'REAL DEFAULT 0'), ('lots', 'INTEGER DEFAULT 1'),
                               ('expiry', 'TEXT DEFAULT ""')]:
            try:
                cursor.execute(f'ALTER TABLE strategy_legs ADD COLUMN {col} {col_type}')
            except sqlite3.OperationalError:
                pass

        # Backtest results table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS backtest_results (
                id             TEXT PRIMARY KEY,
                strategy_type  TEXT NOT NULL,
                run_date       TEXT NOT NULL,
                start_date     TEXT NOT NULL,
                end_date       TEXT NOT NULL,
                total_trades   INTEGER,
                win_rate       REAL,
                avg_pnl        REAL,
                total_pnl      REAL,
                max_drawdown   REAL,
                sharpe         REAL,
                result_json    TEXT
            )
        ''')

        conn.commit()
        conn.close()

    # ─────────────────────────────────────────────────────
    # Create / Track
    # ─────────────────────────────────────────────────────
    def track_strategy(self, strategy_data: dict, entry_spot: float = 0) -> str:
        """
        Saves a newly generated strategy into the database for live tracking.
        Returns the generated strategy_id.
        """
        strat_id = str(int(time.time() * 1000))
        now = datetime.now()

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO strategies (
                id, name, type, status, entry_premium, entry_spot,
                entry_time_str, entry_timestamp, reasoning, score
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            strat_id,
            strategy_data.get('name', 'Custom Strategy'),
            strategy_data.get('type', 'UNKNOWN'),
            'ACTIVE',
            float(strategy_data.get('premium', 0)),
            float(entry_spot),
            now.strftime('%Y-%m-%d %H:%M:%S'),
            int(now.timestamp()),
            strategy_data.get('reasoning', ''),
            int(strategy_data.get('score', 0))
        ))

        legs = strategy_data.get('legs', [])
        leg_records = [(
            strat_id,
            leg.get('type'),
            leg.get('action'),
            float(leg.get('strike', 0)),
            float(leg.get('price', 0)),
            float(leg.get('iv', 0)),
            int(leg.get('lots', 1)),
            str(leg.get('expiry', ''))
        ) for leg in legs]

        if leg_records:
            cursor.executemany('''
                INSERT INTO strategy_legs
                    (strategy_id, opt_type, action, strike, entry_price, iv, lots, expiry)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', leg_records)

        conn.commit()
        conn.close()
        return strat_id

    # ─────────────────────────────────────────────────────
    # Update
    # ─────────────────────────────────────────────────────
    def set_stop_loss(self, strategy_id: str, level: float):
        """Record stop-loss level for a strategy."""
        conn = sqlite3.connect(self.db_path)
        conn.execute('UPDATE strategies SET stop_loss_level = ? WHERE id = ?',
                     (level, strategy_id))
        conn.commit()
        conn.close()

    def set_target(self, strategy_id: str, level: float):
        conn = sqlite3.connect(self.db_path)
        conn.execute('UPDATE strategies SET target_level = ? WHERE id = ?',
                     (level, strategy_id))
        conn.commit()
        conn.close()

    def add_note(self, strategy_id: str, note: str):
        conn = sqlite3.connect(self.db_path)
        conn.execute('UPDATE strategies SET notes = ? WHERE id = ?',
                     (note, strategy_id))
        conn.commit()
        conn.close()

    # ─────────────────────────────────────────────────────
    # Close / Exit
    # ─────────────────────────────────────────────────────
    def close_strategy(self, strategy_id: str, exit_premium: float,
                       exit_spot: float = 0, reason: str = 'MANUAL') -> float:
        """
        Marks a strategy as CLOSED, computes and records P&L.
        Returns realized P&L per lot.
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute('SELECT * FROM strategies WHERE id = ?', (strategy_id,))
        row = cursor.fetchone()
        if not row:
            conn.close()
            return 0.0

        entry_premium = float(row['entry_premium'])
        strat_type    = row['type']
        pnl_per_lot   = (exit_premium - entry_premium) * 75

        now = datetime.now()
        cursor.execute('''
            UPDATE strategies SET
                status = 'CLOSED',
                exit_premium = ?,
                exit_spot = ?,
                exit_time_str = ?,
                exit_timestamp = ?,
                pnl = ?,
                notes = COALESCE(notes || ' | ', '') || ?
            WHERE id = ?
        ''', (
            float(exit_premium),
            float(exit_spot),
            now.strftime('%Y-%m-%d %H:%M:%S'),
            int(now.timestamp()),
            float(pnl_per_lot),
            f'Closed via {reason}',
            strategy_id
        ))

        conn.commit()
        conn.close()
        return pnl_per_lot

    # ─────────────────────────────────────────────────────
    # Query
    # ─────────────────────────────────────────────────────
    def get_all_active_strategies(self) -> list:
        """Returns active strategies in legacy-compatible format."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM strategies WHERE status = 'ACTIVE' ORDER BY entry_timestamp DESC")
        strategy_rows = cursor.fetchall()

        results = []
        for r in strategy_rows:
            strat_dict = dict(r)
            strat_dict['premium']    = strat_dict.pop('entry_premium')
            strat_dict['tracked_at'] = strat_dict.pop('entry_time_str')

            cursor.execute("SELECT * FROM strategy_legs WHERE strategy_id = ?", (strat_dict['id'],))
            leg_rows = cursor.fetchall()

            legs = []
            for lr in leg_rows:
                leg_dict = dict(lr)
                leg_dict['price'] = leg_dict.pop('entry_price')
                leg_dict['type']  = leg_dict.pop('opt_type')
                legs.append(leg_dict)

            strat_dict['legs'] = legs
            results.append(strat_dict)

        conn.close()
        return results

    def get_closed_strategies(self, limit: int = 30) -> list:
        """Returns recently closed strategies with P&L."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM strategies WHERE status = 'CLOSED' ORDER BY exit_timestamp DESC LIMIT ?",
            (limit,)
        )
        rows = [dict(r) for r in cursor.fetchall()]
        conn.close()
        return rows

    def get_summary(self) -> dict:
        """Returns aggregate P&L summary for the tracker panel."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*), SUM(pnl) FROM strategies WHERE status = 'CLOSED'")
        row = cursor.fetchone()
        conn.close()
        count, total_pnl = row
        return {
            'closed_count': count or 0,
            'total_realized_pnl': float(total_pnl or 0),
        }

    # ─────────────────────────────────────────────────────
    # Backtest persistence
    # ─────────────────────────────────────────────────────
    def save_backtest(self, strategy_type: str, report_json: str,
                      stats: dict, start_date: str, end_date: str):
        """Persist a BacktestReport for historical reference."""
        bt_id = f'bt_{strategy_type}_{int(time.time())}'
        conn  = sqlite3.connect(self.db_path)
        conn.execute('''
            INSERT OR REPLACE INTO backtest_results
            (id, strategy_type, run_date, start_date, end_date,
             total_trades, win_rate, avg_pnl, total_pnl, max_drawdown, sharpe, result_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            bt_id, strategy_type,
            datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            start_date, end_date,
            int(stats.get('total_trades', 0)),
            float(stats.get('win_rate', 0)),
            float(stats.get('avg_pnl', 0)),
            float(stats.get('total_pnl', 0)),
            float(stats.get('max_drawdown', 0)),
            float(stats.get('sharpe', 0)),
            report_json
        ))
        conn.commit()
        conn.close()
        return bt_id

    def get_backtest_history(self, strategy_type: str = '', limit: int = 5) -> list:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        if strategy_type:
            cursor.execute(
                "SELECT * FROM backtest_results WHERE strategy_type = ? ORDER BY run_date DESC LIMIT ?",
                (strategy_type, limit)
            )
        else:
            cursor.execute(
                "SELECT * FROM backtest_results ORDER BY run_date DESC LIMIT ?",
                (limit,)
            )
        rows = [dict(r) for r in cursor.fetchall()]
        conn.close()
        return rows

    def delete_strategy(self, strategy_id: str):
        """Hard delete a strategy and its legs."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("PRAGMA foreign_keys = ON")
        cursor.execute("DELETE FROM strategies WHERE id = ?", (strategy_id,))
        conn.commit()
        conn.close()
