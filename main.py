import os
import sqlite3
import json
import difflib
import random
from datetime import datetime, timezone
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from apscheduler.schedulers.background import BackgroundScheduler
import firebase_admin
from firebase_admin import credentials, firestore
from google.cloud.firestore_v1.base_query import FieldFilter
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
# Soportar Persistent Volume en Railway
VOLUME_PATH = os.getenv('RAILWAY_VOLUME_MOUNT_PATH', '')
if VOLUME_PATH:
    SQLITE_DB = os.path.join(VOLUME_PATH, 'search_index.db')
else:
    SQLITE_DB = 'search_index.db'

def init_db():
    conn = sqlite3.connect(SQLITE_DB)
    # Habilitar modo WAL (Write-Ahead Logging) para alta concurrencia
    conn.execute('PRAGMA journal_mode=WAL;')
    # Optimizar el rendimiento de escritura
    conn.execute('PRAGMA synchronous=NORMAL;')
    
    c = conn.cursor()
    # FTS5 crea una tabla virtual súper rápida para texto
    # type: 'store' o 'product'
    try:
        c.execute("SELECT likes FROM search_index LIMIT 1")
    except sqlite3.OperationalError:
        c.execute("DROP TABLE IF EXISTS search_index")
        
    c.execute('''
        CREATE VIRTUAL TABLE IF NOT EXISTS search_index USING fts5(
            id, type, storeId, name, category, description, price, icon, imageUrl UNINDEXED, onSale UNINDEXED, salePrice UNINDEXED, likes UNINDEXED, views UNINDEXED, purchases UNINDEXED
        )
    ''')
    
    c.execute('''
        CREATE TABLE IF NOT EXISTS metadata (
            key TEXT PRIMARY KEY,
            value TEXT
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
    "regalo": ["regalo", "mama", "mamá", "madre", "cumpleaños", "aniversario", "floristeria", "flores", "spa", "chocolates", "detalle", "regalos"],
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
    
    # 1. Leer Promociones desde marketing_campaigns
    import time
    now_ms = int(time.time() * 1000)
    
    camps_ref = db.collection("marketing_campaigns")
    # Filtramos solo activas y tipo banner, o filtramos localmente para simplificar
    camps = list(camps_ref.stream())
    
    count_banners = 0
    if len(camps) > 0:
        for promo in camps:
            p_data = promo.to_dict()
            if p_data.get('type') in ['simple', 'premium_product', 'premium_store']:
                # Validar estado y expiración
                if p_data.get('status') == 'active':
                    expires_at = p_data.get('expiresAt', 0)
                    if expires_at > now_ms:
                        c.execute("""
                            INSERT INTO promotions (id, type, targetUrl, imageUrl, storeId, emoji, title, subtitle, bg, titleColor, subtitleColor)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (
                            promo.id,
                            p_data.get('type', 'simple'),
                            p_data.get('targetUrl', ''),
                            p_data.get('imageUrl', ''),
                            p_data.get('commerceId', ''),
                            p_data.get('emoji', ''),
                            p_data.get('title', ''),
                            p_data.get('subtitle', ''),
                            p_data.get('bg', '#000'),
                            p_data.get('titleColor', '#FFF'),
                            p_data.get('subtitleColor', '#FFF')
                        ))
                        count_banners += 1

    if count_banners == 0:
        # Fallback si no hay banners
        default_ads = [
            ("1", "simple", "store", "", "", "local-offer", "Descubre Ofertas", "En los mejores comercios", "#FFE4E1", "#DC143C", "#CD5C5C")
        ]
        for ad in default_ads:
            c.execute("INSERT INTO promotions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", ad)

    # 2. Leer Comercios
    stores_ref = db.collection("stores")
    stores = stores_ref.stream()
    
    count = 0
    for store in stores:
        s_data = store.to_dict()
        s_id = store.id
        
        # Insertar el comercio en el índice
        c.execute("""
            INSERT INTO search_index (id, type, storeId, name, category, description, price, icon, imageUrl, onSale, salePrice, likes, views, purchases)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0, NULL, 0, 0, 0)
        """, (
            s_id, 'store', s_id, 
            s_data.get('name', ''), 
            s_data.get('category', ''), 
            '', '', '', s_data.get('logoUrl', s_data.get('imageUrl', ''))
        ))
        count += 1
        
        # Leer los productos de este comercio (Sub-colección)
        products_ref = stores_ref.document(s_id).collection("products")
        for product in products_ref.stream():
            p_data = product.to_dict()
            c.execute("""
                INSERT INTO search_index (id, type, storeId, name, category, description, price, icon, imageUrl, onSale, salePrice, likes, views, purchases)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                product.id, 'product', s_id, 
                p_data.get('name', ''), 
                p_data.get('category', ''), 
                p_data.get('description', ''), 
                str(p_data.get('price', '')),
                p_data.get('icon', ''),
                p_data.get('imageUrl', ''),
                1 if p_data.get('onSale') else 0,
                p_data.get('salePrice', None),
                p_data.get('likes', 0),
                p_data.get('views', 0),
                p_data.get('purchases', 0)
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
            INSERT INTO search_index (id, type, storeId, name, category, description, price, icon, imageUrl, onSale, salePrice, likes, views, purchases)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0, NULL, 0, 0, 0)
        """, (
            store_id, 'store', store_id, 
            s_data.get('name', ''), 
            s_data.get('category', ''), 
            '', '', '', s_data.get('logoUrl', s_data.get('imageUrl', ''))
        ))
        count += 1
        
        # 3. Leer Productos
        products_ref = store_ref.collection("products")
        for product in products_ref.stream():
            p_data = product.to_dict()
            c.execute("""
                INSERT INTO search_index (id, type, storeId, name, category, description, price, icon, imageUrl, onSale, salePrice, likes, views, purchases)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                product.id, 'product', store_id, 
                p_data.get('name', ''), 
                p_data.get('category', ''), 
                p_data.get('description', ''), 
                str(p_data.get('price', '')),
                p_data.get('icon', ''),
                p_data.get('imageUrl', ''),
                1 if p_data.get('onSale') else 0,
                p_data.get('salePrice', None),
                p_data.get('likes', 0),
                p_data.get('views', 0),
                p_data.get('purchases', 0)
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
            SELECT p.id, p.type, p.storeId, p.name, p.category, p.description,
                   p.price, p.icon, p.imageUrl, p.onSale, p.salePrice, p.likes, p.views, p.purchases,
                   s.name as storeName
            FROM search_index p
            LEFT JOIN (SELECT id, name FROM search_index WHERE type='store') s ON s.id = p.storeId
            WHERE search_index MATCH ?
            ORDER BY rank
            LIMIT 50
        """, (fts_query,))
        
        rows = c.fetchall()
        results = [dict(row) for row in rows]
        
        # FALLBACK 1: LIKE Substring match (ideal para fragmentos como "ur" en "burguer")
        if len(results) == 0 and len(safe_q) >= 2:
            like_q = f"%{safe_q}%"
            c.execute("""
                SELECT p.id, p.type, p.storeId, p.name, p.category, p.description,
                       p.price, p.icon, p.imageUrl, p.onSale, p.salePrice, p.likes, p.views, p.purchases,
                       s.name as storeName
                FROM search_index p
                LEFT JOIN (SELECT id, name FROM search_index WHERE type='store') s ON s.id = p.storeId
                WHERE p.name LIKE ? OR p.category LIKE ? OR p.description LIKE ?
                LIMIT 50
            """, (like_q, like_q, like_q))
            
            rows_like = c.fetchall()
            results = [dict(row) for row in rows_like]

        # FALLBACK 2: FUZZY (Si no encontró nada y la query tiene al menos 3 letras)
        if len(results) == 0 and len(safe_q) >= 3:
            c.execute("""
                SELECT p.id, p.type, p.storeId, p.name, p.category, p.description,
                       p.price, p.icon, p.imageUrl, p.onSale, p.salePrice, p.likes, p.views, p.purchases,
                       s.name as storeName
                FROM search_index p
                LEFT JOIN (SELECT id, name FROM search_index WHERE type='store') s ON s.id = p.storeId
            """)
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
    # Hacemos JOIN con el registro de la tienda para obtener su nombre
    c.execute("""
        SELECT p.id, p.type, p.storeId, p.name, p.category, p.description,
               p.price, p.icon, p.imageUrl, p.onSale, p.salePrice, p.likes, p.views, p.purchases,
               s.name as storeName
        FROM search_index p
        LEFT JOIN (SELECT id, name FROM search_index WHERE type='store') s ON s.id = p.storeId
        WHERE p.type = 'product'
        ORDER BY CAST(p.likes AS INTEGER) DESC, CAST(p.views AS INTEGER) DESC, RANDOM()
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

MACRO_CLUSTERS = {
    "desayuno": {
        "titles": ["Empieza el día con energía", "Mañanas deliciosas", "Despierta con sabor", "Para el desayuno"],
        "keywords": "desayuno OR arepa OR pan OR cafe OR huevos OR tamal OR calentao OR pastel"
    },
    "comida_rapida": {
        "titles": ["Antojos Rápidos", "Para calmar el hambre", "Pecados deliciosos", "Tus favoritos"],
        "keywords": "hamburguesa OR pizza OR perro caliente OR salchipapa OR frito OR pollo"
    },
    "saludable": {
        "titles": ["Cuida tu cuerpo", "Opciones Saludables", "Ligero y delicioso", "Para mantener la línea"],
        "keywords": "ensalada OR bowl OR saludable OR vegano OR vegetariano OR light OR dieta"
    },
    "regalos": {
        "titles": ["Para esa persona especial", "Detalles que enamoran", "Sorpresas únicas", "Regalos inolvidables"],
        "keywords": "regalo OR flor OR spa OR chocolate OR detalle OR aniversario OR peluche OR amor"
    },
    "licores": {
        "titles": ["Para la fiesta", "Salud y celebración", "Prende la noche", "Tus bebidas favoritas"],
        "keywords": "licor OR cerveza OR aguardiente OR ron OR vodka OR vino OR coctel OR fiesta OR hielo"
    },
    "farmacia": {
        "titles": ["Cuida de tu salud", "Farmacia en casa", "Lo que necesitas, rápido", "Alivio inmediato"],
        "keywords": "farmacia OR medicamento OR pastilla OR dolor OR salud OR cuidado OR resaca OR guayabo OR suero"
    },
    "hogar": {
        "titles": ["Mejora tu hogar", "Todo para tu casa", "Remodela tu espacio", "Cuidado del hogar"],
        "keywords": "mueble OR herramienta OR pintura OR decoracion OR limpieza OR aseo OR ferreteria OR destornillador"
    },
    "mercado": {
        "titles": ["Directo a tu nevera", "Mercado fresco", "Llena tu despensa", "Frutas y verduras"],
        "keywords": "mercado OR carne OR pollo OR verdura OR fruta OR lacteo OR viveres OR abarrotes"
    },
    "mascotas": {
        "titles": ["Para el rey de la casa", "Mimos para tu peludo", "Cuidado animal", "Amor de 4 patas"],
        "keywords": "mascota OR perro OR gato OR purina OR concentrado OR veterinaria OR pet"
    },
    "ropa": {
        "titles": ["Completa tu clóset", "Renueva tu estilo", "Moda recomendada", "Tendencias"],
        "keywords": "ropa OR camisa OR pantalon OR zapato OR tenis OR moda OR accesorio OR reloj OR gafas"
    }
}

@app.get("/api/home/{uid}")
def get_dynamic_home_feed(uid: str):
    """Devuelve el inicio completo (Home Feed) basado en el Algoritmo V2.0 (Time-Decay, Context, FTS5)."""
    feed_sections = []
    
    conn = sqlite3.connect(SQLITE_DB)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
        
    # 1. Leer Intenciones (Actividad) y aplicar Time-Decay
    cluster_scores = {k: 0.0 for k in MACRO_CLUSTERS.keys()}
    now = datetime.now(timezone.utc)
    current_hour = (now.hour - 5) % 24  # UTC-5 (Colombia)
    
    if db:
        try:
            # Leer ultimos 100 eventos de actividad del usuario
            activities = db.collection('users').document(uid).collection('activity').order_by('timestamp', direction=firestore.Query.DESCENDING).limit(100).stream()
            for act in activities:
                data = act.to_dict()
                cat = (data.get('category') or '').lower()
                ts = data.get('timestamp')
                
                # Time-Decay Weighting
                multiplier = 1.0
                if ts:
                    try:
                        days_ago = (now - ts).days
                        if days_ago == 0: multiplier = 3.0
                        elif days_ago > 7: multiplier = 0.2
                        elif days_ago > 30: multiplier = 0.0
                    except:
                        pass
                
                score = (2.0 if data.get('type') == 'search' else 1.0) * multiplier
                
                # Match activity category to our MACRO_CLUSTERS
                for c_key, c_val in MACRO_CLUSTERS.items():
                    if cat in c_val['keywords'].lower() or cat == c_key:
                        cluster_scores[c_key] += score
        except Exception as e:
            print("Error leyendo actividad:", e)
            
    # 2. Conciencia Temporal (Context-Awareness)
    if 6 <= current_hour <= 10:
        cluster_scores["desayuno"] += 15.0
        cluster_scores["farmacia"] += 5.0
    elif 11 <= current_hour <= 15:
        cluster_scores["comida_rapida"] += 10.0
        cluster_scores["saludable"] += 8.0
    elif 18 <= current_hour <= 23 or current_hour < 4:
        cluster_scores["comida_rapida"] += 15.0
        cluster_scores["licores"] += 12.0
        
    # 3. Selección 80/20 (Explotación vs Exploración)
    sorted_clusters = sorted([k for k, v in cluster_scores.items() if v > 0], key=lambda k: cluster_scores[k], reverse=True)
    top_clusters = sorted_clusters[:2]
    
    unvisited = [k for k in MACRO_CLUSTERS.keys() if k not in top_clusters]
    exploration_cluster = random.choice(unvisited) if unvisited else None
    
    selected_clusters = top_clusters.copy()
    if exploration_cluster:
        selected_clusters.append(exploration_cluster)
        
    # Fallback si no hay clusters seleccionados (ej: cuenta nueva y hora neutra)
    if not selected_clusters:
        selected_clusters = ["comida_rapida", "mercado", random.choice(list(MACRO_CLUSTERS.keys()))]
        
    # 4. Construir Secciones Dinámicas con FTS5 MATCH
    for cluster in selected_clusters:
        keywords = MACRO_CLUSTERS[cluster]["keywords"]
        title = random.choice(MACRO_CLUSTERS[cluster]["titles"])
        subtitle = "Descubre algo nuevo" if cluster == exploration_cluster else "Basado en tus intereses"
        
        c.execute("""
            SELECT p.id, p.type, p.storeId, p.name, p.category, p.description,
                   p.price, p.icon, p.imageUrl, p.onSale, p.salePrice, p.likes, p.views, p.purchases,
                   s.name as storeName
            FROM search_index p
            LEFT JOIN (SELECT id, name FROM search_index WHERE type='store') s ON s.id = p.storeId
            WHERE p.type = 'product' AND search_index MATCH ?
            ORDER BY CAST(p.likes AS INTEGER) DESC, CAST(p.views AS INTEGER) DESC, RANDOM()
            LIMIT 5
        """, (keywords,))
        items = c.fetchall()
        if items:
            feed_sections.append({
                "id": f"dyn_{cluster}",
                "type": "products",
                "title": title,
                "subtitle": subtitle,
                "items": [dict(row) for row in items]
            })

    # 5. Los más Baratos
    c.execute("""
        SELECT p.id, p.type, p.storeId, p.name, p.category, p.description,
               p.price, p.icon, p.imageUrl, p.onSale, p.salePrice, p.likes, p.views, p.purchases,
               s.name as storeName
        FROM search_index p
        LEFT JOIN (SELECT id, name FROM search_index WHERE type='store') s ON s.id = p.storeId
        WHERE p.type = 'product' AND CAST(p.price AS INTEGER) > 0
        ORDER BY CAST(p.price AS INTEGER) ASC
        LIMIT 5
    """)
    cheap_items = c.fetchall()
    if cheap_items:
        feed_sections.append({
            "id": "cheap_deals",
            "type": "products",
            "title": "¡Ahorra dinero!",
            "subtitle": "Los más baratos del sistema",
            "items": [dict(row) for row in cheap_items]
        })

    # 6. Comercios Destacados (Limitado)
    c.execute("""
        SELECT id, type, storeId, name, category, description, price, icon, imageUrl as logoUrl
        FROM search_index
        WHERE type = 'store'
        LIMIT 15
    """)
    stores = c.fetchall()
    if stores:
        store_list = []
        for s in stores:
            s_dict = dict(s)
            s_dict['open'] = True # Mocking open state since search_index doesn't track it yet, or default to True
            s_dict['time'] = '15-30 min'
            s_dict['rating'] = 4.5
            s_dict['deliveryFee'] = 0
            store_list.append(s_dict)
            
        feed_sections.append({
            "id": "stores",
            "type": "stores",
            "title": "Cerca de ti",
            "subtitle": "Lugares recomendados en tu zona",
            "items": store_list
        })
        
    conn.close()
    return {"sections": feed_sections}

@app.get("/api/recommendations/{uid}")
def get_user_recommendations(uid: str):
    """Obtiene recomendaciones on-demand calculadas en base a la actividad del usuario."""
    if not db:
        return get_popular_products()
        
    try:
        # 1. Analizar actividad reciente del usuario en Firestore (clics, busquedas, vistas)
        activities = db.collection('users').document(uid).collection('activity').order_by('timestamp', direction=firestore.Query.DESCENDING).limit(50).stream()
        
        category_scores = {}
        for act in activities:
            data = act.to_dict()
            cat = data.get('category')
            if cat and cat != 'General':
                score = 2 if data.get('type') == 'search' else 1
                category_scores[cat] = category_scores.get(cat, 0) + score
                
        if not category_scores:
            return get_popular_products() # Fallback si es un usuario nuevo sin clics
            
        top_categories = sorted(category_scores.keys(), key=lambda k: category_scores[k], reverse=True)[:2]
        
        # 2. Consultar nuestra base local ultrarrápida (SQLite) para buscar productos de esas categorías
        conn = sqlite3.connect(SQLITE_DB)
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        
        placeholders = ', '.join('?' for _ in top_categories)
        query_sql = f"""
            SELECT p.id, p.type, p.storeId, p.name, p.category, p.description,
                   p.price, p.icon, p.imageUrl, p.onSale, p.salePrice, p.likes, p.views, p.purchases,
                   s.name as storeName
            FROM search_index p
            LEFT JOIN (SELECT id, name FROM search_index WHERE type='store') s ON s.id = p.storeId
            WHERE p.type = 'product' AND p.category IN ({placeholders})
            ORDER BY CAST(p.likes AS INTEGER) DESC, CAST(p.views AS INTEGER) DESC, RANDOM()
            LIMIT 6
        """
        c.execute(query_sql, top_categories)
        rows = c.fetchall()
        conn.close()
        
        results = [dict(row) for row in rows]
        
        if len(results) < 3:
            # Si no hay suficientes productos, completamos con los más populares
            return get_popular_products()
            
        return {"results": results}
    except Exception as e:
        print(f"Error generando recomendaciones para {uid}:", e)
        return get_popular_products()

# ==========================================
# ==========================================
# WEBHOOKS PUSH PARA ACTUALIZAR ÍNDICE (MINI-ALGOLIA)
# ==========================================
import threading

sqlite_lock = threading.Lock()

class ProductPayload(BaseModel):
    id: str
    storeId: str
    name: str
    category: Optional[str] = ""
    description: Optional[str] = ""
    price: Optional[float] = 0
    icon: Optional[str] = ""
    imageUrl: Optional[str] = ""
    onSale: Optional[bool] = False
    salePrice: Optional[float] = None
    likes: Optional[int] = 0
    views: Optional[int] = 0
    purchases: Optional[int] = 0

class StorePayload(BaseModel):
    id: str
    name: str
    category: Optional[str] = ""
    imageUrl: Optional[str] = ""

@app.post("/api/index/product")
def index_product(payload: ProductPayload):
    with sqlite_lock:
        conn = sqlite3.connect(SQLITE_DB, timeout=10.0)
        c = conn.cursor()
        c.execute("DELETE FROM search_index WHERE id = ? AND type = 'product'", (payload.id,))
        c.execute("""
            INSERT INTO search_index (id, type, storeId, name, category, description, price, icon, imageUrl, onSale, salePrice, likes, views, purchases)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            payload.id, 'product', payload.storeId, 
            payload.name, payload.category, payload.description, 
            str(payload.price), payload.icon, payload.imageUrl,
            1 if payload.onSale else 0, payload.salePrice,
            payload.likes, payload.views, payload.purchases
        ))
        conn.commit()
        conn.close()
    return {"status": "indexed", "id": payload.id}

@app.delete("/api/index/product/{product_id}")
def delete_product_index(product_id: str):
    with sqlite_lock:
        conn = sqlite3.connect(SQLITE_DB, timeout=10.0)
        c = conn.cursor()
        c.execute("DELETE FROM search_index WHERE id = ? AND type = 'product'", (product_id,))
        conn.commit()
        conn.close()
    return {"status": "deleted", "id": product_id}

@app.post("/api/index/store")
def index_store(payload: StorePayload):
    with sqlite_lock:
        conn = sqlite3.connect(SQLITE_DB, timeout=10.0)
        c = conn.cursor()
        c.execute("DELETE FROM search_index WHERE id = ? AND type = 'store'", (payload.id,))
        c.execute("""
            INSERT INTO search_index (id, type, storeId, name, category, description, price, icon, imageUrl, onSale, salePrice, likes, views, purchases)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0, NULL, 0, 0, 0)
        """, (
            payload.id, 'store', payload.id, 
            payload.name, payload.category, '', '', '', payload.imageUrl
        )) 
        conn.commit()
        conn.close()
    return {"status": "indexed", "id": payload.id}

def on_stores_snapshot(col_snapshot, changes, read_time):
    with sqlite_lock:
        try:
            conn = sqlite3.connect(SQLITE_DB, timeout=10.0)
            c = conn.cursor()
            for change in changes:
                doc = change.document
                s_id = doc.id
                if change.type.name in ['ADDED', 'MODIFIED']:
                    s_data = doc.to_dict() or {}
                    c.execute("DELETE FROM search_index WHERE id = ? AND type = 'store'", (s_id,))
                    c.execute("""
                        INSERT INTO search_index (id, type, storeId, name, category, description, price, icon, imageUrl, onSale, salePrice, likes, views, purchases)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0, NULL, 0, 0, 0)
                    """, (
                        s_id, 'store', s_id, 
                        s_data.get('name', ''), 
                        s_data.get('category', ''), 
                        '', '', '', s_data.get('logoUrl', s_data.get('imageUrl', ''))
                    ))
                elif change.type.name == 'REMOVED':
                    c.execute("DELETE FROM search_index WHERE id = ? AND type = 'store'", (s_id,))
            conn.commit()
            conn.close()
            print(f"[Realtime Sync] Procesados {len(changes)} cambios en Stores")
        except Exception as e:
            print(f"Error en on_stores_snapshot: {e}")

import time
import threading
from datetime import datetime, timezone, timedelta

def delta_sync_loop():
    if not db:
        print("Firebase no inicializado. No se puede iniciar delta sync.")
        return

    while True:
        try:
            with sqlite_lock:
                conn = sqlite3.connect(SQLITE_DB, timeout=10.0)
                c = conn.cursor()
                c.execute("SELECT value FROM metadata WHERE key = 'last_sync_time'")
                row = c.fetchone()
                last_sync_str = row[0] if row else None
            
            if not last_sync_str:
                print("[Delta Sync] Primer arranque o SQLite vacío. Sincronizando todo el catálogo...")
                try:
                    sync_database() # Llamamos a la sincronización completa
                    current_time = datetime.now(timezone.utc).isoformat()
                    with sqlite_lock:
                        conn = sqlite3.connect(SQLITE_DB, timeout=10.0)
                        c = conn.cursor()
                        c.execute("INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)", ('last_sync_time', current_time))
                        conn.commit()
                        conn.close()
                    print("[Delta Sync] Sincronización inicial completada.")
                except Exception as ex:
                    print(f"Error en full sync inicial: {ex}")
            else:
                last_sync_dt = datetime.fromisoformat(last_sync_str)
                stores_ref = db.collection("stores").where(filter=FieldFilter("updatedAt", ">", last_sync_dt))
                changed_stores = list(stores_ref.stream())
                
                products_ref = db.collection_group("products").where(filter=FieldFilter("updatedAt", ">", last_sync_dt))
                changed_products = list(products_ref.stream())
                
                if changed_stores or changed_products:
                    print(f"[Delta Sync] Cambios detectados: {len(changed_stores)} comercios, {len(changed_products)} productos.")
                    with sqlite_lock:
                        conn = sqlite3.connect(SQLITE_DB, timeout=10.0)
                        c = conn.cursor()
                        
                        for store in changed_stores:
                            s_data = store.to_dict()
                            s_id = store.id
                            c.execute("DELETE FROM search_index WHERE id = ? AND type = 'store'", (s_id,))
                            c.execute("""
                                INSERT INTO search_index (id, type, storeId, name, category, description, price, icon, imageUrl, onSale, salePrice, likes, views, purchases)
                                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0, NULL, 0, 0, 0)
                            """, (
                                s_id, 'store', s_id, 
                                s_data.get('name', ''), 
                                s_data.get('category', ''), 
                                '', '', '', s_data.get('logoUrl', s_data.get('imageUrl', ''))
                            ))
                            
                        for prod in changed_products:
                            p_data = prod.to_dict()
                            p_id = prod.id
                            path_parts = prod.reference.path.split('/')
                            store_id = path_parts[1] if len(path_parts) >= 2 else ""
                            c.execute("DELETE FROM search_index WHERE id = ? AND type = 'product'", (p_id,))
                            if p_data.get('available', True):
                                c.execute("""
                                    INSERT INTO search_index (id, type, storeId, name, category, description, price, icon, imageUrl, onSale, salePrice, likes, views, purchases)
                                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                                """, (
                                    p_id, 'product', store_id, 
                                    p_data.get('name', ''), 
                                    p_data.get('category', ''), 
                                    p_data.get('description', ''), 
                                    str(p_data.get('price', '')),
                                    p_data.get('icon', ''),
                                    p_data.get('imageUrl', ''),
                                    1 if p_data.get('onSale') else 0,
                                    p_data.get('salePrice', None),
                                    p_data.get('likes', 0),
                                    p_data.get('views', 0),
                                    p_data.get('purchases', 0)
                                ))
                        
                        current_time = datetime.now(timezone.utc).isoformat()
                        c.execute("INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)", ('last_sync_time', current_time))
                        conn.commit()
                        conn.close()
                        print(f"[Delta Sync] Sincronización exitosa. Siguiente chequeo desde {current_time}.")

        except Exception as e:
            # Firebase gRPC error usually happens here due to missing Composite Index
            print(f"[Delta Sync Error]: {e}")

        # Dormir 60 segundos antes de volver a verificar
        time.sleep(60)

def cleanup_activity_loop():
    if not db:
        return
        
    # Esperamos 1 minuto antes del primer barrido para no saturar el arranque
    time.sleep(60)
    
    while True:
        try:
            # Eliminar actividad de más de 30 días
            thirty_days_ago = datetime.now(timezone.utc) - timedelta(days=30)
            
            old_activities = db.collection_group('activity').where(filter=FieldFilter("timestamp", "<", thirty_days_ago)).limit(500).stream()
            
            deleted_count = 0
            for doc in old_activities:
                doc.reference.delete()
                deleted_count += 1
                
            if deleted_count > 0:
                print(f"[Cleanup] Eliminados {deleted_count} registros de actividad antiguos.")
                
        except Exception as e:
            print(f"[Cleanup Error]: Requiere índice. {e}")
            
        # Esperar 24 horas (86400 segundos)
        time.sleep(86400)

@app.on_event("startup")
def startup_event():
    print("Search backend is ready. Iniciando procesos en segundo plano...")
    threading.Thread(target=delta_sync_loop, daemon=True).start()
    threading.Thread(target=cleanup_activity_loop, daemon=True).start()
