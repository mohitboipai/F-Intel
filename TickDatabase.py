import sqlite3
import pandas as pd
from datetime import datetime
import threading
import time

class IntradayTickDB:
    """
    Records snapshots of the Option Chain (df_chain) to a local SQLite database.
    This enables historical time-scrubbing for strategy P&L tracking throughout the day.
    """
    def __init__(self, db_path='intraday_chain.db'):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Initializes the database schema if it doesn't exist."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # We record the exact tick minute, strike, option type (CE/PE), and live price
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS option_ticks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp_str TEXT NOT NULL,       -- e.g. '2026-03-18 10:15:00'
                timestamp_epoch INTEGER NOT NULL,  -- for fast exact time bounding
                symbol TEXT NOT NULL,
                strike REAL NOT NULL,
                opt_type TEXT NOT NULL,
                price REAL NOT NULL,
                iv REAL,
                delta REAL
            )
        ''')
        
        # Create an index for faster time-based queries
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_time_strike 
            ON option_ticks (timestamp_epoch, strike, opt_type)
        ''')
        
        conn.commit()
        conn.close()

    def record_snapshot(self, symbol, df_chain):
        """
        Takes the current df_chain and saves every strike to the DB
        with the current system timestamp. Call this gracefully every 1-3 mins.
        """
        if df_chain.empty:
            return
            
        now = datetime.now()
        now_str = now.strftime('%Y-%m-%d %H:%M:00') # Floor to minute
        now_epoch = int(now.timestamp())
        
        # To avoid recording the exact same minute twice if called too fast
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT 1 FROM option_ticks WHERE timestamp_str = ? LIMIT 1", (now_str,))
        if cursor.fetchone():
            conn.close()
            return  # Already recorded this minute
            
        # Prepare batch insert natively
        records = []
        for _, row in df_chain.iterrows():
            # df_chain has columns: strike, type, symbol, price, id
            records.append((
                now_str,
                now_epoch,
                symbol,
                float(row.get('strike', 0)),
                str(row.get('type', '')),
                float(row.get('price', 0)),
                None, # Placeholder for IV if we want to save it later
                None  # Placeholder for Delta if we want to save it later
            ))
            
        cursor.executemany('''
            INSERT INTO option_ticks (timestamp_str, timestamp_epoch, symbol, strike, opt_type, price, iv, delta)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', records)
        
        conn.commit()
        conn.close()
        
    def get_chain_at_time(self, symbol, target_time_str):
        """
        Fetches the option chain closest to the requested time (YYYY-MM-DD HH:MM).
        Returns a DataFrame identical in structure to df_chain.
        """
        try:
            target_dt = datetime.strptime(target_time_str, '%Y-%m-%d %H:%M')
        except ValueError:
            # If they just pass "10:15", assume today
            today_str = datetime.now().strftime('%Y-%m-%d')
            target_dt = datetime.strptime(f"{today_str} {target_time_str}", '%Y-%m-%d %H:%M')
            
        target_epoch = int(target_dt.timestamp())
        
        conn = sqlite3.connect(self.db_path)
        # Find the absolute closest recorded epoch to the requested time
        query = f'''
            SELECT timestamp_epoch 
            FROM option_ticks 
            ORDER BY ABS(timestamp_epoch - {target_epoch}) ASC 
            LIMIT 1
        '''
        cursor = conn.cursor()
        cursor.execute(query)
        res = cursor.fetchone()
        
        if not res:
            conn.close()
            return pd.DataFrame()
            
        closest_epoch = res[0]
        
        # Fetch all strikes for that exact closest snapshot
        df = pd.read_sql_query(f'''
            SELECT strike, opt_type as type, price, symbol 
            FROM option_ticks 
            WHERE timestamp_epoch = {closest_epoch}
        ''', conn)
        
        conn.close()
        return df

class TickRecorderThread(threading.Thread):
    """
    Background thread that asks VolatilityAnalyzer for the chain
    every 2 minutes and saves it.
    """
    def __init__(self, analyzer_instance):
        super().__init__(daemon=True)
        self.analyzer = analyzer_instance
        self.db = IntradayTickDB()
        
    def run(self):
        while True:
            try:
                # Give the main loop time to populate the shared cache
                time.sleep(120) 
                
                # Fetch chain natively via the cache to avoid hitting Fyers limits
                chain_raw = self.analyzer.get_option_chain_data()
                if chain_raw:
                    df = self.analyzer.parse_and_filter(chain_raw)
                    if not df.empty:
                        self.db.record_snapshot(self.analyzer.symbol, df)
            except Exception as e:
                print(f"[TickRecorder] Error saving snapshot: {e}")
