import sqlite3
import json

# Database
conn = sqlite3.connect(database='chatbot.db', check_same_thread=False)

# Create thread metadata table to store thread names
cursor = conn.cursor()
cursor.execute('''
    CREATE TABLE IF NOT EXISTS thread_metadata (
        thread_id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        named INTEGER DEFAULT 0,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
''')
conn.commit()

# ======================== Thread Metadata Functions ========================

def save_thread_metadata(thread_id: str, name: str, named: bool = False):
    """Save or update thread metadata (name and named status)"""
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO thread_metadata (thread_id, name, named)
        VALUES (?, ?, ?)
        ON CONFLICT(thread_id) 
        DO UPDATE SET name=excluded.name, named=excluded.named
    ''', (thread_id, name, 1 if named else 0))
    conn.commit()


def get_thread_metadata(thread_id: str):
    """Get thread metadata (name and named status)"""
    cursor = conn.cursor()
    cursor.execute('''
        SELECT name, named FROM thread_metadata WHERE thread_id = ?
    ''', (thread_id,))
    result = cursor.fetchone()
    
    if result:
        return {"name": result[0], "named": bool(result[1])}
    return None


def get_all_thread_metadata():
    """Get metadata for all threads"""
    cursor = conn.cursor()
    cursor.execute('''
        SELECT thread_id, name, named FROM thread_metadata
        ORDER BY created_at DESC
    ''')
    results = cursor.fetchall()
    
    return [
        {
            "id": row[0],
            "name": row[1],
            "named": bool(row[2])
        }
        for row in results
    ]
