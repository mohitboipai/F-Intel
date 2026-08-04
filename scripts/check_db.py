import sqlite3

def check_db(db_name):
    print(f"\n--- {db_name} ---")
    try:
        db = sqlite3.connect(db_name)
        tables = db.execute("SELECT name FROM sqlite_master WHERE type='table';").fetchall()
        print("Tables:", [t[0] for t in tables])
        for t in tables:
            count = db.execute(f"SELECT COUNT(*) FROM {t[0]}").fetchone()[0]
            print(f"Table {t[0]} has {count} rows.")
            # print first row or schema
            schema = db.execute(f"PRAGMA table_info({t[0]});").fetchall()
            print(f"  Schema: {[c[1] for c in schema]}")
    except Exception as e:
        print("Error:", e)

check_db('intraday_chain.db')
check_db('strategies.db')
