import os
import sqlite3
import json
import difflib
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import firebase_admin
from firebase_admin import credentials, firestore
from pydantic import BaseModel
from typing import List, Optional

app = FastAPI(title="Punto Search Engine (Mini-Algolia)")

# Habilitar CORS para que la app móvil o web pueda consultar sin bloqueos
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

import json

# Inicializar Firebase Admin SDK (solo si existe el archivo json o la variable de entorno)
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

# Configurar SQLite con FTS5 (Full-Text Search 5)
SQLITE_DB = 'search_index.db'

def init_db():
    conn = sqlite3.connect(SQLITE_DB)
    c = conn.cursor()
    # FTS5 crea una tabla virtual súper rápida para texto
    # type: 'store' o 'product'
    try:
        c.execute("SELECT salePrice FROM search_index LIMIT 1")
    except sqlite3.OperationalError:
        c.execute("DROP TABLE IF EXISTS search_index")
        
    c.execute('''
        CREATE VIRTUAL TABLE IF NOT EXISTS search_index USING fts5(
            id, type, storeId, name, category, description, price, icon, imageUrl UNINDEXED, onSale UNINDEXED, salePrice UNINDEXED
        )
    ''')
    
    # Tabla normal para las promociones (no necesitamos fts5 para esto)
    c.execute('''
        CREATE TABLE IF NOT EXISTS promotions (
            id TEXT PRIMARY KEY,
            emoji TEXT,
            title TEXT,
            subtitle TEXT,
            bg TEXT,
            titleColor TEXT,
            subtitleColor TEXT
        )
    ''')
    conn.commit()
    conn.close()

init_db()

# ==========================================
# MOTOR INTELIGENTE DE BÚSQUEDA
# ==========================================
SYNONYMS = {
    # Comida Rápida y Restaurantes
    "hamburguesa": ["hamburguesa", "burger", "burguer", "hanburguesa"],
    "gaseosa": ["gaseosa", "coca", "coca-cola", "coca cola", "pepsi", "soda", "sprite", "postobon", "refresco", "bebida"],
    "pizza": ["pizza", "piza", "pissa"],
    "perro": ["perro", "hot dog", "hotdog", "salchicha", "hot-dog", "chori", "chorizo"],
    "pollo": ["pollo", "broaster", "asado", "alitas", "wings", "nuggets", "pechuga"],
    "papas": ["papas", "fritas", "francesa", "cascos", "salchipapa", "papa"],
    "helado": ["helado", "postre", "cono", "paleta", "sundae", "mcflurry", "brownie"],
    "cerveza": ["cerveza", "pola", "biela", "chela", "club colombia", "aguila", "poker", "corona", "heineken"],
    "jugo": ["jugo", "zumo", "batido", "licuado", "limonada", "jugos", "avena"],
    "carne": ["carne", "res", "churrasco", "parrilla", "asado", "picada", "cerdo", "chuzo"],
    "empanada": ["empanada", "pastel", "arepa", "pastelito", "dedito", "tequeno", "tequeño", "pandebono", "buñuelo"],
    "sushi": ["sushi", "maki", "roll", "sashimi", "nigiri"],
    
    # Farmacia / Salud
    "pastillas": ["pastilla", "pildora", "tableta", "medicamento", "droga", "acetaminofen", "ibuprofeno", "aspirina", "dolex", "advil"],
    "jarabe": ["jarabe", "tos"],
    "preservativos": ["preservativo", "condon", "condones", "profilactico", "duo", "today"],
    "alcohol": ["alcohol", "antiseptico", "antibacterial", "desinfectante"],
    "panal": ["pañal", "panales", "pañales", "winny", "huggies", "pequeñin", "pañalitis"],
    "toallas": ["toalla", "toallas", "nosotras", "protectores", "tampones"],
    "crema": ["crema", "pomada", "unguento", "gel"],
    "suero": ["suero", "pedialyte", "electrolit"],
    
    # Ferretería / Hogar
    "taladro": ["taladro", "perforadora", "pulidora", "caladora"],
    "martillo": ["martillo", "mazo", "maceta", "alicate", "pinza", "hombre solo"],
    "destornillador": ["destornillador", "desatornillador", "estrella", "pala"],
    "bombillo": ["bombillo", "foco", "lampara", "luz", "bombilla", "led"],
    "pintura": ["pintura", "esmalte", "vinilo", "brocha", "rodillo", "aerosol", "thinner"],
    "clavos": ["clavo", "clavos", "puntilla", "tornillo", "chazo", "tuerca", "arandela"],
    "cinta": ["cinta", "pegante", "aislante", "enmascarar", "pegamento", "silicona", "boxer"],
    "tubo": ["tubo", "pvc", "tuberia", "codo", "accesorio", "soldadura"],
    "llave": ["llave", "candado", "cerradura", "cerrojo", "chapa"],
    "cable": ["cable", "alambre", "extension", "enchufe", "tomacorriente", "interruptor"],
    
    # Tecnología / Celulares
    "cargador": ["cargador", "cable", "adaptador", "fuente"],
    "audifonos": ["audifonos", "auriculares", "diadema", "airpods", "inpods", "earpods", "headset"],
    "celular": ["celular", "telefono", "smartphone", "iphone", "android", "movil", "xiaomi", "samsung", "motorola", "huawei"],
    "pantalla": ["pantalla", "display", "monitor", "tv", "televisor", "glass", "vidrio templado", "visor"],
    "bateria": ["bateria", "pila", "powerbank"],
    "computador": ["computador", "pc", "laptop", "portatil", "computadora", "teclado", "mouse", "raton", "impresora"],
    "memoria": ["memoria", "usb", "microsd", "pendrive", "disco duro", "ssd"],
    "funda": ["funda", "estuche", "carcasa", "forro", "case", "protector"]
}

REVERSE_SYNONYMS = {}
for root, alts in SYNONYMS.items():
    for alt in alts:
        REVERSE_SYNONYMS[alt] = root
# ==========================================

@app.post("/api/sync")
def sync_database():
    """Descarga todos los comercios y productos de Firestore y reconstruye el índice SQLite."""
    if not db:
        raise HTTPException(status_code=500, detail="Firebase no está inicializado. Falta serviceAccountKey.json o FIREBASE_SERVICE_ACCOUNT.")
    
    conn = sqlite3.connect(SQLITE_DB)
    c = conn.cursor()
    
    # Vaciar el índice actual
    c.execute("DELETE FROM search_index")
    c.execute("DELETE FROM promotions")
    
    # 1. Leer Promociones
    promos_ref = db.collection("promotions")
    promos = list(promos_ref.stream())
    
    if len(promos) == 0:
        # Insertar datos por defecto si no hay en firebase
        default_ads = [
            ("1", "local-offer", "50% OFF en Tacos", "Tacos El Rey", "#FFE4E1", "#DC143C", "#CD5C5C"),
            ("2", "local-pizza", "2x1 en Pizzas", "Pizza Nostra", "#E6E6FA", "#4B0082", "#6A5ACD"),
            ("3", "bakery-dining", "Envío Gratis", "Pan Artesano", "#FFFACD", "#B8860B", "#DAA520")
        ]
        for ad in default_ads:
            c.execute("INSERT INTO promotions VALUES (?, ?, ?, ?, ?, ?, ?)", ad)
    else:
        for promo in promos:
            p_data = promo.to_dict()
            c.execute("""
                INSERT INTO promotions (id, emoji, title, subtitle, bg, titleColor, subtitleColor)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                promo.id,
                p_data.get('emoji', 'star'),
                p_data.get('title', ''),
                p_data.get('subtitle', ''),
                p_data.get('bg', '#000'),
                p_data.get('titleColor', '#FFF'),
                p_data.get('subtitleColor', '#FFF')
            ))

    # 2. Leer Comercios
    stores_ref = db.collection("stores")
    stores = stores_ref.stream()
    
    count = 0
    for store in stores:
        s_data = store.to_dict()
        s_id = store.id
        
        # Insertar el comercio en el índice
        c.execute("""
            INSERT INTO search_index (id, type, storeId, name, category, description, price, icon, imageUrl, onSale, salePrice)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0, NULL)
        """, (
            s_id, 'store', s_id, 
            s_data.get('name', ''), 
            s_data.get('category', ''), 
            '', '', '', s_data.get('imageUrl', '')
        ))
        count += 1
        
        # Leer los productos de este comercio (Sub-colección)
        products_ref = stores_ref.document(s_id).collection("products")
        for product in products_ref.stream():
            p_data = product.to_dict()
            c.execute("""
                INSERT INTO search_index (id, type, storeId, name, category, description, price, icon, imageUrl, onSale, salePrice)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                product.id, 'product', s_id, 
                p_data.get('name', ''), 
                p_data.get('category', ''), 
                p_data.get('description', ''), 
                str(p_data.get('price', '')),
                p_data.get('icon', ''),
                p_data.get('imageUrl', ''),
                1 if p_data.get('onSale') else 0,
                p_data.get('salePrice', None)
            ))
            count += 1

    conn.commit()
    conn.close()
    
    return {"message": "Sincronización exitosa", "items_indexed": count}

@app.post("/api/sync/store/{store_id}")
def sync_store(store_id: str):
    """Sincroniza un solo comercio y sus productos (Más rápido)."""
    if not db:
        raise HTTPException(status_code=500, detail="Firebase no está inicializado.")
        
    conn = sqlite3.connect(SQLITE_DB)
    c = conn.cursor()
    
    # 1. Eliminar datos antiguos del comercio
    c.execute("DELETE FROM search_index WHERE storeId = ?", (store_id,))
    
    count = 0
    # 2. Leer Comercio
    store_ref = db.collection("stores").document(store_id)
    store_doc = store_ref.get()
    
    if store_doc.exists:
        s_data = store_doc.to_dict()
        c.execute("""
            INSERT INTO search_index (id, type, storeId, name, category, description, price, icon, imageUrl, onSale, salePrice)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0, NULL)
        """, (
            store_id, 'store', store_id, 
            s_data.get('name', ''), 
            s_data.get('category', ''), 
            '', '', '', s_data.get('imageUrl', '')
        ))
        count += 1
        
        # 3. Leer Productos
        products_ref = store_ref.collection("products")
        for product in products_ref.stream():
            p_data = product.to_dict()
            c.execute("""
                INSERT INTO search_index (id, type, storeId, name, category, description, price, icon, imageUrl, onSale, salePrice)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                product.id, 'product', store_id, 
                p_data.get('name', ''), 
                p_data.get('category', ''), 
                p_data.get('description', ''), 
                str(p_data.get('price', '')),
                p_data.get('icon', ''),
                p_data.get('imageUrl', ''),
                1 if p_data.get('onSale') else 0,
                p_data.get('salePrice', None)
            ))
            count += 1

    conn.commit()
    conn.close()
    return {"message": f"Comercio {store_id} sincronizado", "items_indexed": count}

@app.get("/api/search")
def search(q: str = ""):
    """Busca en milisegundos en el índice FTS5 usando sinónimos y Fuzzy Match."""
    if not q.strip():
        return {"results": []}
    
    conn = sqlite3.connect(SQLITE_DB)
    # Devolver filas como diccionarios
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    
    safe_q = q.replace('"', '').replace("'", "").lower().strip()
    words = safe_q.split()
    expanded_parts = []
    
    for word in words:
        if word in REVERSE_SYNONYMS:
            root = REVERSE_SYNONYMS[word]
            syns = SYNONYMS[root]
            # Formar grupo OR para FTS5
            # Ej: ("hamburguesa"* OR "burger"* OR "burguer"*)
            group = " OR ".join([f'"{s}"*' for s in syns])
            expanded_parts.append(f"({group})")
        else:
            expanded_parts.append(f'"{word}"*')
            
    fts_query = " ".join(expanded_parts)
    
    try:
        # Buscamos en todas las columnas y ordenamos por "rank" (relevancia automática de SQLite FTS5)
        c.execute("""
            SELECT id, type, storeId, name, category, description, price, icon, imageUrl, onSale, salePrice
            FROM search_index 
            WHERE search_index MATCH ?
            ORDER BY rank
            LIMIT 50
        """, (fts_query,))
        
        rows = c.fetchall()
        results = [dict(row) for row in rows]
        
        # FUZZY FALLBACK (Si no encontró nada y la query tiene al menos 3 letras)
        if len(results) == 0 and len(safe_q) >= 3:
            c.execute("SELECT id, type, storeId, name, category, description, price, icon, imageUrl, onSale, salePrice FROM search_index")
            all_items = c.fetchall()
            
            if all_items:
                # Extraer todos los nombres
                names = [item["name"] for item in all_items if item["name"]]
                # Encontrar coincidencias aproximadas (cutoff bajo para perdonar errores graves)
                matches = difflib.get_close_matches(safe_q, names, n=15, cutoff=0.45)
                
                if matches:
                    seen_ids = set()
                    for item in all_items:
                        # Si el nombre del item fue uno de los que matcheó
                        if item["name"] in matches and item["id"] not in seen_ids:
                            results.append(dict(item))
                            seen_ids.add(item["id"])
                    
                    # (Opcional) Ordenar para que el match más perfecto de difflib salga primero
                    # difflib.get_close_matches ya los devuelve en orden de mejor a peor,
                    # así que mapeamos ese orden a los resultados:
                    results.sort(key=lambda x: matches.index(x["name"]) if x["name"] in matches else 999)
                    
    except Exception as e:
        print("Error de búsqueda:", e)
        results = []
    finally:
        conn.close()
    
    return {"results": results}

@app.get("/api/popular")
def get_popular_products():
    """Devuelve productos recomendados o populares."""
    conn = sqlite3.connect(SQLITE_DB)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    # Tomamos algunos productos aleatorios como "populares" para el MVP
    c.execute("""
        SELECT id, type, storeId, name, category, description, price, icon, imageUrl, onSale, salePrice
        FROM search_index 
        WHERE type = 'product'
        ORDER BY RANDOM()
        LIMIT 6
    """)
    rows = c.fetchall()
    conn.close()
    return {"results": [dict(row) for row in rows]}

@app.get("/api/promotions")
def get_promotions():
    """Devuelve las promociones indexadas."""
    conn = sqlite3.connect(SQLITE_DB)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("SELECT * FROM promotions")
    rows = c.fetchall()
    conn.close()
    return {"results": [dict(row) for row in rows]}

# ==========================================
# FIREBASE REAL-TIME LISTENERS
# ==========================================
import threading

sqlite_lock = threading.Lock()

def on_snapshot_stores(col_snapshot, changes, read_time):
    with sqlite_lock:
        conn = sqlite3.connect(SQLITE_DB, timeout=10.0)
        c = conn.cursor()
        for change in changes:
            doc = change.document
            s_id = doc.id
            s_data = doc.to_dict()
            if change.type.name in ['ADDED', 'MODIFIED']:
                c.execute("""
                    INSERT OR REPLACE INTO search_index (id, type, storeId, name, category, description, price, icon, imageUrl, onSale, salePrice)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0, NULL)
                """, (
                    s_id, 'store', s_id, 
                    s_data.get('name', ''), 
                    s_data.get('category', ''), 
                    '', '', '', s_data.get('imageUrl', '')
                ))
            elif change.type.name == 'REMOVED':
                c.execute("DELETE FROM search_index WHERE id = ? AND type = 'store'", (s_id,))
        conn.commit()
        conn.close()

def on_snapshot_products(col_snapshot, changes, read_time):
    with sqlite_lock:
        conn = sqlite3.connect(SQLITE_DB, timeout=10.0)
        c = conn.cursor()
        for change in changes:
            doc = change.document
            p_id = doc.id
            p_data = doc.to_dict()
            s_id = doc.reference.parent.parent.id
            
            if change.type.name in ['ADDED', 'MODIFIED']:
                c.execute("""
                    INSERT OR REPLACE INTO search_index (id, type, storeId, name, category, description, price, icon, imageUrl, onSale, salePrice)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    p_id, 'product', s_id, 
                    p_data.get('name', ''), 
                    p_data.get('category', ''), 
                    p_data.get('description', ''), 
                    str(p_data.get('price', '')),
                    p_data.get('icon', ''),
                    p_data.get('imageUrl', ''),
                    1 if p_data.get('onSale') else 0,
                    p_data.get('salePrice', None)
                ))
            elif change.type.name == 'REMOVED':
                c.execute("DELETE FROM search_index WHERE id = ? AND type = 'product'", (p_id,))
        conn.commit()
        conn.close()

def start_firestore_listeners():
    if not db:
        print("Firebase no inicializado. No se iniciaron los listeners.")
        return
    print("Iniciando listeners de Firestore en segundo plano...")
    db.collection("stores").on_snapshot(on_snapshot_stores)
    db.collection_group("products").on_snapshot(on_snapshot_products)

@app.on_event("startup")
def startup_event():
    listener_thread = threading.Thread(target=start_firestore_listeners, daemon=True)
    listener_thread.start()
