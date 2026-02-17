# database.py
import sqlite3
import json
from datetime import datetime
import streamlit as st

@st.cache_resource
def init_database():
    """Initialize SQLite database for caching analysis results with schema migration"""
    conn = sqlite3.connect('analysis_cache.db', check_same_thread=False)
    c = conn.cursor()
    
    # Check if table exists
    c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='analysis_cache'")
    table_exists = c.fetchone() is not None
    
    if not table_exists:
        # Create new table with all columns
        c.execute('''
            CREATE TABLE analysis_cache (
                ticker TEXT PRIMARY KEY,
                base_report TEXT,
                report TEXT,
                timestamp DATETIME,
                score INTEGER,
                signal TEXT,
                current_price REAL,
                change REAL,
                change_pct REAL,
                bid REAL,
                ask REAL,
                volume INTEGER,
                week_high_52 REAL,
                week_low_52 REAL,
                market_cap TEXT,
                listed_exchange TEXT
            )
        ''')
    else:
        # Check which columns exist and add missing ones
        c.execute("PRAGMA table_info(analysis_cache)")
        existing_columns = [column[1] for column in c.fetchall()]
        
        # Define all new columns we want to add
        new_columns = [
            ('current_price', 'REAL'),
            ('change', 'REAL'),
            ('change_pct', 'REAL'),
            ('bid', 'REAL'),
            ('ask', 'REAL'),
            ('volume', 'INTEGER'),
            ('week_high_52', 'REAL'),
            ('week_low_52', 'REAL'),
            ('market_cap', 'TEXT'),
            ('listed_exchange', 'TEXT')
        ]
        
        # Add missing columns
        for col_name, col_type in new_columns:
            if col_name not in existing_columns:
                try:
                    c.execute(f'ALTER TABLE analysis_cache ADD COLUMN {col_name} {col_type}')
                except Exception:
                    pass
    
    # Create index on timestamp for cleanup
    c.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON analysis_cache(timestamp)')
    
    conn.commit()
    return conn

def cleanup_old_cache(conn, max_entries=20):
    """Keep only the most recent N entries in cache"""
    c = conn.cursor()
    c.execute('SELECT COUNT(*) FROM analysis_cache')
    count = c.fetchone()[0]
    
    if count > max_entries:
        c.execute('''
            DELETE FROM analysis_cache 
            WHERE timestamp IN (
                SELECT timestamp FROM analysis_cache 
                ORDER BY timestamp DESC 
                LIMIT -1 OFFSET ?
            )
        ''', (max_entries,))
        conn.commit()

def save_to_cache(conn, ticker, base_report, report, score, signal, market_data=None):
    """Save analysis results to cache with market data"""
    c = conn.cursor()
    
    base_report_json = json.dumps(base_report, default=str)
    report_json = json.dumps(report, default=str)
    timestamp = datetime.now()
    
    if market_data:
        current_price = market_data.get('current_price', report.get('price', 0))
        change = market_data.get('change', 0)
        change_pct = market_data.get('change_pct', 0)
        bid = market_data.get('bid', 0)
        ask = market_data.get('ask', 0)
        volume = market_data.get('volume', 0)
        week_high_52 = market_data.get('week_high_52', 0)
        week_low_52 = market_data.get('week_low_52', 0)
        market_cap = market_data.get('market_cap', 'N/A')
        listed_exchange = market_data.get('listed_exchange', 'NASDAQ')
    else:
        current_price = report.get('price', 0)
        change = 0
        change_pct = 0
        bid = report.get('csp', {}).get('bid', 0)
        ask = report.get('csp', {}).get('ask', 0)
        volume = 0
        week_high_52 = 0
        week_low_52 = 0
        market_cap = 'N/A'
        listed_exchange = 'NASDAQ'
    
    c.execute('''
        INSERT OR REPLACE INTO analysis_cache 
        (ticker, base_report, report, timestamp, score, signal, current_price, change, change_pct, 
         bid, ask, volume, week_high_52, week_low_52, market_cap, listed_exchange)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (ticker, base_report_json, report_json, timestamp, score, signal, current_price, change, change_pct,
          bid, ask, volume, week_high_52, week_low_52, market_cap, listed_exchange))
    
    conn.commit()
    cleanup_old_cache(conn, 20)

def load_from_cache(conn, ticker):
    """Load analysis results from cache"""
    c = conn.cursor()
    
    c.execute("PRAGMA table_info(analysis_cache)")
    columns = [col[1] for col in c.fetchall()]
    
    query = 'SELECT base_report, report, timestamp, score, signal'
    optional_cols = []
    for col in ['current_price', 'change', 'change_pct', 'bid', 'ask', 'volume', 
                'week_high_52', 'week_low_52', 'market_cap', 'listed_exchange']:
        if col in columns:
            query += f', {col}'
            optional_cols.append(col)
    
    query += ' FROM analysis_cache WHERE ticker = ?'
    
    c.execute(query, (ticker,))
    result = c.fetchone()
    
    if result:
        base_report = json.loads(result[0])
        report = json.loads(result[1])
        timestamp = result[2]
        score = result[3]
        signal = result[4]
        
        market_data = {}
        for i, col in enumerate(optional_cols, start=5):
            if i < len(result):
                market_data[col] = result[i]
        
        return {
            'base_report': base_report,
            'report': report,
            'timestamp': datetime.fromisoformat(timestamp) if isinstance(timestamp, str) else timestamp,
            'score': score,
            'signal': signal,
            'market_data': market_data
        }
    return None

def get_recent_tickers(conn, limit=10):
    """Get most recently analyzed tickers"""
    c = conn.cursor()
    
    c.execute("PRAGMA table_info(analysis_cache)")
    columns = [col[1] for col in c.fetchall()]
    
    select_cols = ['ticker', 'timestamp', 'score', 'signal']
    for col in ['current_price', 'change', 'change_pct']:
        if col in columns:
            select_cols.append(col)
    
    query = f"SELECT {', '.join(select_cols)} FROM analysis_cache ORDER BY timestamp DESC LIMIT ?"
    
    c.execute(query, (limit,))
    results = c.fetchall()
    
    recent = []
    for row in results:
        item = {
            'ticker': row[0],
            'timestamp': row[1],
            'score': row[2],
            'signal': row[3]
        }
        
        idx = 4
        if 'current_price' in select_cols:
            item['current_price'] = row[idx] if idx < len(row) else 0
            idx += 1
        if 'change' in select_cols:
            item['change'] = row[idx] if idx < len(row) else 0
            idx += 1
        if 'change_pct' in select_cols:
            item['change_pct'] = row[idx] if idx < len(row) else 0
        
        recent.append(item)
    
    return recent