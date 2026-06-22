import re

file_path = 'main.py'
with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Add import sqlite_vec
if 'import sqlite_vec' not in content:
    content = content.replace('import sqlite3', 'import sqlite3\nimport sqlite_vec')

# 2. Define get_db_connection
db_conn_code = """
def get_db_connection():
    conn = sqlite3.connect(SQLITE_DB, timeout=10.0)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    conn.execute('PRAGMA journal_mode=WAL;')
    conn.execute('PRAGMA synchronous=NORMAL;')
    conn.row_factory = sqlite3.Row
    return conn
"""

if 'def get_db_connection()' not in content:
    # Insert right before init_db()
    content = content.replace('def init_db():', db_conn_code + '\ndef init_db():')

# 3. Replace all sqlite3.connect calls (except the one inside get_db_connection)
# First temporarily rename the one inside get_db_connection
content = content.replace('conn = sqlite3.connect(SQLITE_DB, timeout=10.0)\n    conn.enable_load_extension(True)', 'conn = __TEMP_CONNECT__\n    conn.enable_load_extension(True)')

# Now replace the rest
content = re.sub(r'sqlite3\.connect\([^\)]+\)', 'get_db_connection()', content)

# Restore the one inside get_db_connection
content = content.replace('conn = __TEMP_CONNECT__', 'conn = sqlite3.connect(SQLITE_DB, timeout=10.0)')

# 4. Update init_db schema
init_db_replacement = """def init_db():
    conn = get_db_connection()
    try:
        conn.execute("SELECT likes FROM search_index LIMIT 1")
    except sqlite3.OperationalError:
        conn.execute("DROP TABLE IF EXISTS search_index")
        
    conn.execute('''
        CREATE VIRTUAL TABLE IF NOT EXISTS search_index USING fts5(
            id, type, storeId, name, category, description, price, icon, imageUrl UNINDEXED, onSale UNINDEXED, salePrice UNINDEXED, likes UNINDEXED, views UNINDEXED, purchases UNINDEXED
        )
    ''')
    
    conn.execute('''
        CREATE TABLE IF NOT EXISTS metadata (
            key TEXT PRIMARY KEY,
            value TEXT
        )
    ''')
    
    conn.execute("DROP TABLE IF EXISTS promotions")
    conn.execute('''
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

    # NUEVO: Tabla para Vectores de Productos
    conn.execute('''
        CREATE VIRTUAL TABLE IF NOT EXISTS product_vectors USING vec0(
            product_id TEXT PRIMARY KEY,
            embedding float[768]
        )
    ''')
    
    # NUEVO: Tabla para Vectores Ancla (Generative UI)
    conn.execute('''
        CREATE VIRTUAL TABLE IF NOT EXISTS anchor_vectors USING vec0(
            anchor_id TEXT PRIMARY KEY,
            embedding float[768]
        )
    ''')
    
    # Tabla auxiliar para metadata de los anclajes
    conn.execute('''
        CREATE TABLE IF NOT EXISTS anchor_metadata (
            anchor_id TEXT PRIMARY KEY,
            title TEXT,
            subtitle TEXT,
            section_type TEXT
        )
    ''')
    
    conn.commit()
    conn.close()"""

# We need to replace the old init_db entirely
content = re.sub(r'def init_db\(\):.*?(?=^# =+)', init_db_replacement + '\n\n', content, flags=re.DOTALL | re.MULTILINE)

with open(file_path, 'w', encoding='utf-8') as f:
    f.write(content)

print("Refactor completed successfully!")
