import os
import sqlite3
import sqlite_vec
import json
import firebase_admin
from firebase_admin import credentials, firestore
from core.config import SQLITE_DB

# --- INICIALIZACIÓN DE FIREBASE ---
SERVICE_ACCOUNT_FILE = 'serviceAccountKey.json'
db = None

if os.getenv('FIREBASE_SERVICE_ACCOUNT'):
    try:
        cred_dict = json.loads(os.getenv('FIREBASE_SERVICE_ACCOUNT'))
        cred = credentials.Certificate(cred_dict)
        firebase_admin.initialize_app(cred)
        db = firestore.client()
        print("Firebase inicializado desde FIREBASE_SERVICE_ACCOUNT")
    except Exception as e:
        print(f"Error parseando FIREBASE_SERVICE_ACCOUNT: {e}")
elif os.path.exists(SERVICE_ACCOUNT_FILE):
    cred = credentials.Certificate(SERVICE_ACCOUNT_FILE)
    firebase_admin.initialize_app(cred)
    db = firestore.client()
else:
    print(f"ADVERTENCIA: No se encontró '{SERVICE_ACCOUNT_FILE}' ni la variable FIREBASE_SERVICE_ACCOUNT. El endpoint /api/sync fallará.")

# --- CONEXIONES A SQLITE CON SOPORTE VECTORIAL ---
def get_db_connection_raw():
    conn = sqlite3.connect(SQLITE_DB, timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    return conn

def get_db_connection():
    conn = get_db_connection_raw()
    try:
        yield conn
    finally:
        conn.close()

def init_db():
    conn = get_db_connection_raw()
    # Habilitar modo WAL (Write-Ahead Logging) para alta concurrencia
    conn.execute('PRAGMA journal_mode=WAL;')
    # Optimizar el rendimiento de escritura
    conn.execute('PRAGMA synchronous=NORMAL;')
    
    c = conn.cursor()
    try:
        c.execute("SELECT available FROM search_index LIMIT 1")
    except sqlite3.OperationalError:
        c.execute("DROP TABLE IF EXISTS search_index")
        
    c.execute('''
        CREATE VIRTUAL TABLE IF NOT EXISTS search_index USING fts5(
            id, type, storeId, name, category, description, price, icon, imageUrl UNINDEXED, onSale UNINDEXED, salePrice UNINDEXED, likes UNINDEXED, views UNINDEXED, purchases UNINDEXED, available UNINDEXED, isOpen UNINDEXED
        )
    ''')
    
    c.execute('''
        CREATE TABLE IF NOT EXISTS metadata (
            key TEXT PRIMARY KEY,
            value TEXT
        )
    ''')
    
    c.execute('''
        CREATE VIRTUAL TABLE IF NOT EXISTS product_vectors USING vec0(
            product_id TEXT PRIMARY KEY,
            embedding float[768]
        )
    ''')
    
    c.execute('''
        CREATE VIRTUAL TABLE IF NOT EXISTS store_vectors USING vec0(
            store_id TEXT PRIMARY KEY,
            embedding float[768]
        )
    ''')
    
    c.execute('''
        CREATE TABLE IF NOT EXISTS anchor_metadata (
            anchor_id TEXT PRIMARY KEY,
            title TEXT,
            subtitle TEXT,
            section_type TEXT
        )
    ''')
    try:
        c.execute("ALTER TABLE anchor_metadata ADD COLUMN exclude_rules TEXT")
    except sqlite3.OperationalError:
        pass
    try:
        c.execute("ALTER TABLE anchor_metadata ADD COLUMN allowed_categories TEXT")
    except sqlite3.OperationalError:
        pass
    try:
        c.execute("ALTER TABLE anchor_metadata ADD COLUMN titles TEXT")
    except sqlite3.OperationalError:
        pass
    try:
        c.execute("ALTER TABLE anchor_metadata ADD COLUMN is_manual INTEGER DEFAULT 0")
    except sqlite3.OperationalError:
        pass
    
    c.execute('''
        CREATE VIRTUAL TABLE IF NOT EXISTS anchor_vectors USING vec0(
            anchor_id TEXT PRIMARY KEY,
            embedding float[768]
        )
    ''')

    c.execute("DROP TABLE IF EXISTS promotions")
    c.execute('''
        CREATE TABLE IF NOT EXISTS promotions (
            id TEXT PRIMARY KEY,
            type TEXT,
            targetUrl TEXT,
            imageUrl TEXT,
            storeId TEXT,
            emoji TEXT,
            title TEXT,
            subtitle TEXT,
            bg TEXT,
            titleColor TEXT,
            subtitleColor TEXT
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS search_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            query TEXT NOT NULL,
            clicked_id TEXT,
            clicked_category TEXT,
            result_count INTEGER DEFAULT 0,
            timestamp TEXT DEFAULT (datetime('now'))
        )
    ''')
    c.execute('CREATE INDEX IF NOT EXISTS idx_search_logs_query ON search_logs(query)')
    c.execute('CREATE INDEX IF NOT EXISTS idx_search_logs_clicked ON search_logs(clicked_id)')
    conn.commit()
    conn.close()
