# database.py
import sqlite3
from datetime import datetime
import streamlit as st

@st.cache_resource
def init_database():
    """Initialize SQLite database for ticker history only."""
    conn = sqlite3.connect('analysis_cache.db', check_same_thread=False)
    c = conn.cursor()
    
    # Check if table exists
    c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='analysis_cache'")
    table_exists = c.fetchone() is not None
    
    if not table_exists:
        c.execute('''
            CREATE TABLE analysis_cache (
                ticker TEXT PRIMARY KEY,
                timestamp DATETIME
            )
        ''')
    else:
        c.execute("PRAGMA table_info(analysis_cache)")
        existing_columns = [column[1] for column in c.fetchall()]
        if 'timestamp' not in existing_columns:
            c.execute('ALTER TABLE analysis_cache ADD COLUMN timestamp DATETIME')
    
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

def save_to_cache(conn, ticker, *_args, **_kwargs):
    """Save ticker timestamp to SQLite history."""
    c = conn.cursor()
    timestamp = datetime.now()

    c.execute('''
        INSERT OR REPLACE INTO analysis_cache (ticker, timestamp)
        VALUES (?, ?)
    ''', (ticker, timestamp))

    conn.commit()
    cleanup_old_cache(conn, 20)

def load_from_cache(conn, ticker):
    """Return ticker/timestamp only (analysis payloads are session-memory only)."""
    c = conn.cursor()
    c.execute('SELECT ticker, timestamp FROM analysis_cache WHERE ticker = ?', (ticker,))
    result = c.fetchone()
    if result:
        return {
            'ticker': result[0],
            'timestamp': datetime.fromisoformat(result[1]) if isinstance(result[1], str) else result[1]
        }
    return None

def get_recent_tickers(conn, limit=10):
    """Get most recently analyzed tickers."""
    c = conn.cursor()
    c.execute('SELECT ticker, timestamp FROM analysis_cache ORDER BY timestamp DESC LIMIT ?', (limit,))
    results = c.fetchall()
    return [{'ticker': row[0], 'timestamp': row[1]} for row in results]
