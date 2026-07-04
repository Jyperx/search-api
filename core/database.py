import sqlite3
import sqlite_vec
import threading
from core.config import SQLITE_DB

sqlite_lock = threading.Lock()

def get_db_connection() -> sqlite3.Connection:
    """Para uso directo en servicios y workers (no en routers)."""
    conn = sqlite3.connect(SQLITE_DB, timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    return conn

def get_db_dep():
    """Para inyección con FastAPI Depends() en los routers."""
    conn = get_db_connection()
    try:
        yield conn
    finally:
        conn.close()

def init_db():
    with sqlite_lock:
        conn = get_db_connection()
        conn.execute('PRAGMA journal_mode=WAL;')
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

        # Comercios destacados (paquete "Comercio Destacado"). Tabla aparte para no tocar el FTS5.
        # featured_until = epoch en ms; el comercio es destacado si featured_until > ahora.
        c.execute('''
            CREATE TABLE IF NOT EXISTS featured_stores (
                store_id TEXT PRIMARY KEY,
                featured_until INTEGER DEFAULT 0
            )
        ''')
        
        c.execute('''
            CREATE TABLE IF NOT EXISTS anchor_metadata (
                anchor_id TEXT PRIMARY KEY,
                title TEXT,
                subtitle TEXT,
                section_type TEXT,
                exclude_rules TEXT,
                allowed_categories TEXT,
                titles TEXT,
                is_manual INTEGER DEFAULT 0
            )
        ''')
        
        c.execute('''
            CREATE VIRTUAL TABLE IF NOT EXISTS anchor_vectors USING vec0(
                anchor_id TEXT PRIMARY KEY,
                embedding float[768]
            )
        ''')
        
        c.execute('''
            CREATE VIRTUAL TABLE IF NOT EXISTS concept_vectors USING vec0(
                id TEXT PRIMARY KEY,
                embedding float[768]
            )
        ''')
        
        c.execute('''
            CREATE VIRTUAL TABLE IF NOT EXISTS user_vectors USING vec0(
                user_id TEXT PRIMARY KEY,
                embedding float[768]
            )
        ''')
        
        c.execute('''
            CREATE TABLE IF NOT EXISTS user_vector_meta (
                user_id TEXT PRIMARY KEY,
                last_updated TEXT,
                event_count INTEGER DEFAULT 0,
                source TEXT DEFAULT 'calculated'
            )
        ''')

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

        c.execute('''
            CREATE TABLE IF NOT EXISTS user_activity_cache (
                user_id TEXT NOT NULL,
                product_id TEXT NOT NULL,
                activity_type TEXT NOT NULL,
                score REAL DEFAULT 1.0,
                timestamp TEXT DEFAULT (datetime('now')),
                PRIMARY KEY (user_id, product_id, activity_type)
            )
        ''')
        c.execute('CREATE INDEX IF NOT EXISTS idx_uac_user ON user_activity_cache(user_id)')
        c.execute('CREATE INDEX IF NOT EXISTS idx_uac_product ON user_activity_cache(product_id)')

        c.execute('''
            CREATE TABLE IF NOT EXISTS store_locations (
                store_id TEXT PRIMARY KEY,
                lat REAL,
                lng REAL
            )
        ''')

        c.execute('''
            CREATE TABLE IF NOT EXISTS section_stats (
                section_id TEXT PRIMARY KEY,
                impressions INTEGER DEFAULT 0,
                clicks INTEGER DEFAULT 0,
                updated_at TEXT DEFAULT (datetime('now'))
            )
        ''')

        c.execute('''
            CREATE TABLE IF NOT EXISTS item_stats (
                product_id TEXT PRIMARY KEY,
                impressions INTEGER DEFAULT 0,
                clicks INTEGER DEFAULT 0,
                purchases INTEGER DEFAULT 0,
                updated_at TEXT DEFAULT (datetime('now'))
            )
        ''')

        c.execute('''
            CREATE TABLE IF NOT EXISTS vector_queue (
                product_id TEXT PRIMARY KEY,
                name TEXT,
                category TEXT,
                description TEXT,
                attempts INTEGER DEFAULT 0,
                last_attempt TEXT,
                source_hint TEXT
            )
        ''')
        conn.commit()
        conn.close()
