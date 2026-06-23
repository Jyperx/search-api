import os
import sqlite3
import sqlite_vec
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import google.generativeai as genai
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import json
import difflib
import random
from datetime import datetime, timezone
import threading
sqlite_lock = threading.Lock()
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from apscheduler.schedulers.background import BackgroundScheduler
import firebase_admin
from firebase_admin import credentials, firestore
from google.cloud.firestore_v1.base_query import FieldFilter
from pydantic import BaseModel
from typing import List, Optional

app = FastAPI(title="Punto Search Engine (Mini-Algolia)")
# Habilitar CORS para que la app m├│vil o web pueda consultar sin bloqueos
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
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
    print(f"ADVERTENCIA: No se encontr├│ '{SERVICE_ACCOUNT_FILE}' ni la variable FIREBASE_SERVICE_ACCOUNT. El endpoint /api/sync fallar├í.")

# --- ACTIVE CACHE PARA EL ALGORITMO V3.2 ---
TIME_RULES_CACHE = [
    {"startHour": 5, "endHour": 10, "cluster": "desayuno", "scoreBoost": 5.0},
    {"startHour": 5, "endHour": 10, "cluster": "comida_rapida", "scoreBoost": -3.0},
    {"startHour": 11, "endHour": 14, "cluster": "almuerzo", "scoreBoost": 5.0},
    {"startHour": 18, "endHour": 23, "cluster": "comida_rapida", "scoreBoost": 3.0},
    {"startHour": 18, "endHour": 23, "cluster": "licores", "scoreBoost": 2.0},
]
MACRO_CLUSTERS_CACHE = {
    "desayuno": {
        "titles": ["Empieza el d├¡a con energ├¡a", "Ma├▒anas deliciosas", "Despierta con sabor", "Para el desayuno"],
        "keywords": "desayuno OR arepa OR pan OR cafe OR huevos OR tamal OR calentao OR jugo OR tostada OR pandebono OR almojabana OR empanada OR bu├▒uelo",
        "storeCategories": "Cafeter├¡a, Panaderia, Restaurante de desayunos, Desayunos",
        "negativeKeywords": "",
        "relatedClusters": "comida_rapida"
    },
    "almuerzo": {
        "titles": ["Hora de almorzar", "Almuerzos Ejecutivos", "Para el medio d├¡a", "Almuerzo Casero"],
        "keywords": "almuerzo OR corrientazo OR sopa OR arroz OR carne OR pollo OR principio OR bandeja OR menu OR ejecutivo",
        "storeCategories": "Restaurante, Comida Casera, Asadero, Almuerzos",
        "negativeKeywords": "desayuno OR pan OR cafe",
        "relatedClusters": "saludable"
    },
    "calor_dia": {
        "titles": ["Para este calorcito ÔÿÇ´©Å", "D├¡as soleados", "Refresca tu tarde"],
        "keywords": "helado OR jugo OR paleta OR granizado OR frappe OR ensalada OR fruta OR refresco OR gafas OR pantaloneta OR camiseta OR bermuda OR sandalias",
        "storeCategories": "Helader├¡a, Jugos, Ropa, Boutique",
        "negativeKeywords": "sopa OR tinto OR cafe OR caliente OR caldo OR chaqueta OR abrigo",
        "relatedClusters": "postres"
    },
    "calor_noche": {
        "titles": ["Noches c├ílidas", "Refr├®scate esta noche", "El calor no para", "Para compartir hoy"],
        "keywords": "helado OR cerveza OR licor OR coctel OR refresco OR frio OR hielo OR bebida",
        "storeCategories": "Helader├¡a, Bar, Licorer├¡a",
        "negativeKeywords": "sopa OR tinto OR cafe OR caliente",
        "relatedClusters": "licores"
    },
    "frio_dia": {
        "titles": ["D├¡as fr├¡os ­ƒîº´©Å", "Acomp├í├▒alo con caf├®", "Entra en calor"],
        "keywords": "cafe OR tinto OR sopa OR caldo OR chocolate OR empanada OR pan OR postre OR tamal OR changua OR chaqueta OR sueter OR bufanda",
        "storeCategories": "Cafeter├¡a, Panaderia, Restaurante, Ropa",
        "negativeKeywords": "helado OR hielo OR cerveza OR pantaloneta",
        "relatedClusters": "desayuno"
    },
    "frio_noche": {
        "titles": ["Noches fr├¡as ­ƒîº´©Å", "No salgas de casa", "Pide a domicilio", "Para el fr├¡o de hoy"],
        "keywords": "sopa OR caldo OR cobija OR saco OR chaqueta OR domicilio OR pizza OR hamburguesa OR comida",
        "storeCategories": "Restaurante, Hogar, Comida R├ípida",
        "negativeKeywords": "helado OR jugo OR hielo",
        "relatedClusters": "comida_rapida"
    },
    "comida_rapida": {
        "titles": ["Antojos R├ípidos", "Para calmar el hambre", "Pecados deliciosos", "Tus favoritos"],
        "keywords": "hamburguesa OR pizza OR salchipapa OR frito OR alitas OR nuggets OR shawarma OR wrap OR combo",
        "storeCategories": "Restaurante, Comida R├ípida, Hamburgueseria, Pizzeria",
        "negativeKeywords": "",
        "relatedClusters": "licores, saludable"
    },
    "saludable": {
        "titles": ["Cuida tu cuerpo", "Opciones Saludables", "Ligero y delicioso", "Para mantener la l├¡nea"],
        "keywords": "ensalada OR bowl OR saludable OR vegano OR vegetariano OR light OR dieta OR acai OR proteina OR organico",
        "storeCategories": "Restaurante Saludable, Jugos, Comida Saludable, Vegano",
        "negativeKeywords": "",
        "relatedClusters": "mercado"
    },
    "regalos": {
        "titles": ["Para esa persona especial", "Detalles que enamoran", "Sorpresas ├║nicas", "Regalos inolvidables"],
        "keywords": "regalo OR flor OR spa OR detalle OR aniversario OR peluche OR amor OR flores OR arreglo OR canasta OR bouquet",
        "storeCategories": "Regaler├¡a, Florister├¡a, Spa, Detalles, Perfumeria",
        "negativeKeywords": "chocolate OR torta OR pastel OR cake OR pan",
        "relatedClusters": ""
    },
    "licores": {
        "titles": ["Para la fiesta", "Salud y celebraci├│n", "Prende la noche", "Tus bebidas favoritas"],
        "keywords": "licor OR cerveza OR aguardiente OR ron OR vodka OR vino OR coctel OR fiesta OR hielo OR tequila OR whisky",
        "storeCategories": "Licorer├¡a, Bar, Distribuidora de Licores",
        "negativeKeywords": "",
        "relatedClusters": "comida_rapida, snacks"
    },
    "farmacia": {
        "titles": ["Cuida de tu salud", "Farmacia en casa", "Lo que necesitas, r├ípido", "Alivio inmediato"],
        "keywords": "farmacia OR medicamento OR pastilla OR dolor OR vitamina OR shampoo OR pa├▒al OR crema OR jabon OR desodorante OR curitas OR antiseptico OR alcohol OR suero OR droga",
        "storeCategories": "Farmacia, Drogue├¡a, Cuidado Personal, Salud, Supermercado",
        "negativeKeywords": "",
        "relatedClusters": ""
    },
    "hogar": {
        "titles": ["Mejora tu hogar", "Todo para tu casa", "Remodela tu espacio", "Cuidado del hogar"],
        "keywords": "mueble OR herramienta OR pintura OR decoracion OR ferreteria OR destornillador OR bombillo OR taladro OR llave OR tornillo OR cable OR electricidad",
        "storeCategories": "Ferreteri├¡a, Hogar, Materiales, Decoraci├│n",
        "negativeKeywords": "jabon OR shampoo OR crema OR pa├▒al OR medicamento",
        "relatedClusters": ""
    },
    "mercado": {
        "titles": ["Directo a tu nevera", "Mercado fresco", "Llena tu despensa", "Frutas y verduras"],
        "keywords": "mercado OR carne OR verdura OR fruta OR lacteo OR viveres OR abarrotes OR huevo OR arroz OR aceite OR sal OR papa OR platano",
        "storeCategories": "Supermercado, Minimarket, Mercado, Carnicer├¡a, Fruver, Tienda",
        "negativeKeywords": "pollo asado OR asadero OR restaurante",
        "relatedClusters": "desayuno"
    },
    "mascotas": {
        "titles": ["Para el rey de la casa", "Mimos para tu peludo", "Cuidado animal", "Amor de 4 patas"],
        "keywords": "mascota OR concentrado OR veterinaria OR pet OR pulgas OR collar OR juguete OR arena OR raza OR canino OR felino",
        "storeCategories": "Veterinaria, Tienda de Mascotas, Pet Shop",
        "negativeKeywords": "perro caliente OR hot dog OR salchicha",
        "relatedClusters": ""
    },
    "ropa": {
        "titles": ["Completa tu cl├│set", "Renueva tu estilo", "Moda recomendada", "Tendencias"],
        "keywords": "ropa OR camisa OR pantalon OR zapato OR tenis OR moda OR accesorio OR reloj OR gafas OR vestido OR falda OR chaqueta OR sudadera",
        "storeCategories": "Ropa, Moda, Calzado, Boutique, Accesorios",
        "negativeKeywords": "",
        "relatedClusters": ""
    },
    "tecnologia": {
        "titles": ["Gadgets para tu vida", "Tecnolog├¡a al instante", "Lo ├║ltimo en tech", "Accesorios para tu celular"],
        "keywords": "audifonos OR cargador OR cable OR funda OR celular OR tablet OR powerbank OR bluetooth OR usb OR memoria OR teclado OR mouse",
        "storeCategories": "Tecnolog├¡a, Electr├│nicos, Celulares, Accesorios Tech",
        "negativeKeywords": "",
        "relatedClusters": ""
    },
    "postres": {
        "titles": ["Dulce tentaci├│n", "Antojos dulces", "El postre que mereces", "Algo dulce hoy"],
        "keywords": "postre OR helado OR torta OR brownie OR cono OR malteada OR muffin OR cheesecake OR tiramis├║ OR flan OR crepe OR waffle",
        "storeCategories": "Helader├¡a, Pasteler├¡a, Caf├®, Postres",
        "negativeKeywords": "",
        "relatedClusters": "comida_rapida, licores"
    }
}

# --- WEATHER CACHE (IN-MEMORY) ---
# Almacena el clima por ubicaci├│n redondeada (aprox 10km) para no quemar la API.
# Formato: {"lat_lng": {"temp": 20, "code": 0, "time": timestamp}}
WEATHER_CACHE_STORE = {}

def on_algorithm_config_snapshot(doc_snapshot, changes, read_time):
    global MACRO_CLUSTERS_CACHE
    global TIME_RULES_CACHE
    for doc in doc_snapshot:
        data = doc.to_dict()
        if data:
            if "clusters" in data:
                MACRO_CLUSTERS_CACHE = data["clusters"]
            if "time_rules" in data:
                TIME_RULES_CACHE = data["time_rules"]
            print(f"­ƒöÑ Cerebro H├¡brido V4.0 RAM Actualizado. Cl├║steres: {len(MACRO_CLUSTERS_CACHE)} | Reglas: {len(TIME_RULES_CACHE)}")

if db:
    doc_ref = db.collection('config').document('algorithm')
    # Inicializar datos si no existen
    doc_snap = doc_ref.get()
    if not doc_snap.exists:
        doc_ref.set({"clusters": MACRO_CLUSTERS_CACHE})
    # Conectar el Listener en tiempo real
    doc_watch = doc_ref.on_snapshot(on_algorithm_config_snapshot)

# Configurar SQLite con FTS5 (Full-Text Search 5)
# Soportar Persistent Volume en Railway
VOLUME_PATH = os.getenv('RAILWAY_VOLUME_MOUNT_PATH', '')
if VOLUME_PATH:
    SQLITE_DB = os.path.join(VOLUME_PATH, 'search_index.db')
else:
    SQLITE_DB = 'search_index.db'

genai.configure(api_key=os.getenv("VITE_GEMINI_API_KEY", os.getenv("GOOGLE_API_KEY", "")))
EMBEDDING_MODEL = "models/gemini-embedding-001"
vector_worker_pool = ThreadPoolExecutor(max_workers=3)

def get_db_connection():
    conn = sqlite3.connect(SQLITE_DB, timeout=10.0)
    conn.row_factory = sqlite3.Row
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    return conn


def init_db():
    conn = get_db_connection()
    # Habilitar modo WAL (Write-Ahead Logging) para alta concurrencia
    conn.execute('PRAGMA journal_mode=WAL;')
    # Optimizar el rendimiento de escritura
    conn.execute('PRAGMA synchronous=NORMAL;')
    
    c = conn.cursor()
    # FTS5 crea una tabla virtual s├║per r├ípida para texto
    # type: 'store' o 'product'
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
            embedding float[3072]
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
    try:
        c.execute("ALTER TABLE anchor_metadata ADD COLUMN rule_type TEXT DEFAULT 'general'")
    except sqlite3.OperationalError:
        pass
    
    c.execute('''
        CREATE VIRTUAL TABLE IF NOT EXISTS anchor_vectors USING vec0(
            anchor_id TEXT PRIMARY KEY,
            embedding float[3072]
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

def generate_product_embedding(name, category, description):
    text = f"Producto a la venta: {name}. Categor├¡a principal del comercio o producto: {category}. Descripci├│n: {description}. (NOTA: Si es comida, pertenece a restaurante/cafeter├¡a, NO a mascotas)."
    import time
    for attempt in range(3):
        try:
            res = genai.embed_content(model=EMBEDDING_MODEL, content=text, task_type="retrieval_document")
            return sqlite_vec.serialize_float32(res['embedding'])
        except Exception as e:
            time.sleep(2 ** attempt)
    return None

def async_index_product_vector(p_id, name, category, description):
    vector_bytes = generate_product_embedding(name, category, description)
    if vector_bytes:
        try:
            with sqlite_lock:
                conn = get_db_connection()
                c = conn.cursor()
                c.execute("DELETE FROM product_vectors WHERE product_id = ?", (p_id,))
                c.execute("INSERT INTO product_vectors (product_id, embedding) VALUES (?, ?)", (p_id, vector_bytes))
                conn.commit()
                conn.close()
        except Exception as e:
            print(f"Error guardando vector: {e}")

def calculate_user_vector(activity_docs, calculate_time_decay_func, current_hour=None):
    product_ids = []
    decay_weights = {}
    
    for doc in activity_docs:
        data = doc.to_dict() if hasattr(doc, 'to_dict') else doc
        p_id = data.get('productId')
        act_type = data.get('type', 'view')
        ts = data.get('timestamp')
        
        # Action Weighting Multiplier
        act_multiplier = 1.0
        if act_type == 'purchase': act_multiplier = 5.0
        elif act_type == 'cart': act_multiplier = 3.0
        elif act_type == 'search': act_multiplier = 2.0
        elif act_type == 'view' or act_type == 'click': act_multiplier = 1.0
        elif act_type == 'ignored': act_multiplier = -0.5
        
        # Circadian Boost (Memoria Horaria)
        if current_hour is not None and ts:
            try:
                from datetime import datetime, timezone
                act_dt = None
                if isinstance(ts, str):
                    act_dt = datetime.fromisoformat(ts.replace('Z', '+00:00'))
                elif isinstance(ts, (int, float)):
                    act_dt = datetime.fromtimestamp(ts/1000 if ts > 10000000000 else ts, tz=timezone.utc)
                    
                if act_dt:
                    act_hour = (act_dt.hour - 5) % 24
                    diff = abs(act_hour - current_hour)
                    if diff > 12: diff = 24 - diff
                    # Si ocurri├│ en la misma ventana horaria (+- 3 horas), boost masivo 3x
                    if diff <= 3:
                        act_multiplier *= 3.0
                    # Si ocurri├│ en un momento totalmente opuesto del d├¡a (+- 8 a 12h), penalizamos 0.3x
                    elif diff >= 8:
                        act_multiplier *= 0.3
            except: pass
        
        if p_id:
            weight = calculate_time_decay_func(ts) * act_multiplier
            if p_id not in decay_weights:
                product_ids.append(p_id)
            decay_weights[p_id] = decay_weights.get(p_id, 0.0) + weight
            
    if not product_ids:
        return None
        
    conn = get_db_connection()
    c = conn.cursor()
    placeholders = ','.join(['?'] * len(product_ids))
    c.execute(f"SELECT product_id, embedding FROM product_vectors WHERE product_id IN ({placeholders})", tuple(product_ids))
    rows = c.fetchall()
    conn.close()
    
    vectors_map = {}
    for row in rows:
        if row['embedding']:
            vectors_map[row['product_id']] = np.frombuffer(row['embedding'], dtype=np.float32)
            
    user_vector = np.zeros(3072, dtype=np.float32)
    total_weight = 0.0
    
    for p_id in product_ids:
        if p_id in vectors_map:
            vec = vectors_map[p_id]
            w = decay_weights[p_id]
            user_vector += (vec * w)
            total_weight += w
            
    if total_weight > 0:
        user_vector = user_vector / total_weight
        return sqlite_vec.serialize_float32(user_vector.tolist())
    return None

# ==========================================
# MOTOR INTELIGENTE DE B├ÜSQUEDA
# ==========================================
SYNONYMS = {
    # Comida R├ípida y Restaurantes
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
    "empanada": ["empanada", "pastel", "arepa", "pastelito", "dedito", "tequeno", "teque├▒o", "pandebono", "bu├▒uelo"],
    "sushi": ["sushi", "maki", "roll", "sashimi", "nigiri"],
    
    # Farmacia / Salud
    "pastillas": ["pastilla", "pildora", "tableta", "medicamento", "droga", "acetaminofen", "ibuprofeno", "aspirina", "dolex", "advil"],
    "jarabe": ["jarabe", "tos"],
    "preservativos": ["preservativo", "condon", "condones", "profilactico", "duo", "today"],
    "alcohol": ["alcohol", "antiseptico", "antibacterial", "desinfectante"],
    "panal": ["pa├▒al", "panales", "pa├▒ales", "winny", "huggies", "peque├▒in", "pa├▒alitis"],
    "toallas": ["toalla", "toallas", "nosotras", "protectores", "tampones"],
    "crema": ["crema", "pomada", "unguento", "gel"],
    "suero": ["suero", "pedialyte", "electrolit"],
    
    # Ferreter├¡a / Hogar
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
    
    # Tecnolog├¡a / Celulares
    "cargador": ["cargador", "cable", "adaptador", "fuente"],
    "audifonos": ["audifonos", "auriculares", "diadema", "airpods", "inpods", "earpods", "headset"],
    "celular": ["celular", "telefono", "smartphone", "iphone", "android", "movil", "xiaomi", "samsung", "motorola", "huawei"],
    "pantalla": ["pantalla", "display", "monitor", "tv", "televisor", "glass", "vidrio templado", "visor"],
    "bateria": ["bateria", "pila", "powerbank"],
    "regalo": ["regalo", "mama", "mam├í", "madre", "cumplea├▒os", "aniversario", "floristeria", "flores", "spa", "chocolates", "detalle", "regalos"],
    "computador": ["computador", "pc", "laptop", "portatil", "computadora", "teclado", "mouse", "raton", "impresora"],
    "memoria": ["memoria", "usb", "microsd", "pendrive", "disco duro", "ssd"],
    "funda": ["funda", "estuche", "carcasa", "forro", "case", "protector"]
}

REVERSE_SYNONYMS = {}
for root, alts in SYNONYMS.items():
    for alt in alts:
        REVERSE_SYNONYMS[alt] = root
# ==========================================

ANCHORS = [
    {"id": "A1", "title": "Gustos Culposos", "subtitle": "Para pecar sin remordimiento", "desc": "Comida r├ípida para humanos, hamburguesas, hot dogs, perros calientes, postres dulces, frituras, pizza, donas."},
    {"id": "A2", "title": "Cena R├ípida", "subtitle": "Sin complicaciones", "desc": "Comida f├ícil de preparar o lista para comer en la noche, s├índwiches, ensaladas ligeras, sushi, wraps."},
    {"id": "A3", "title": "Desayuno Energ├®tico", "subtitle": "Empieza el d├¡a con todo", "desc": "Caf├®, huevos, pan, arepas, jugo de naranja, tostadas, tocino."},
    {"id": "A4", "title": "Mercado Fresco", "subtitle": "Para la alacena", "desc": "Frutas frescas, verduras, l├ícteos, carnes, abarrotes, despensa."},
    {"id": "A5", "title": "Farmacia y Cuidado", "subtitle": "Salud y bienestar", "desc": "Medicamentos, vitaminas, cuidado personal, aseo, primeros auxilios."},
    {"id": "A6", "title": "Mascotas Felices", "subtitle": "Para tu peludo", "desc": "Alimento y accesorios exclusivos para animales. Croquetas para caninos y felinos, arena, juguetes, snacks para mascotas. (EXCLUYE y rechaza comida r├ípida humana)."},
    {"id": "A7", "title": "Tecnolog├¡a", "subtitle": "Gadgets y repuestos", "desc": "Celulares, cargadores, aud├¡fonos, pantallas, cables, accesorios electr├│nicos."},
    {"id": "A8", "title": "Hogar y Ferreter├¡a", "subtitle": "Arregla tu casa", "desc": "Herramientas, bombillos, cintas, plomer├¡a, tornillos, pinturas."}
]

from fastapi import BackgroundTasks

def do_seed_anchors():
    """L├│gica interna para sembrar anclas en segundo plano."""
    try:
        with sqlite_lock:
            conn = get_db_connection()
            c = conn.cursor()
            c.execute("DELETE FROM anchor_vectors")
            c.execute("DELETE FROM anchor_metadata")
            conn.commit()
            conn.close()
        
        for a in ANCHORS:
            text = f"{a['title']} - {a['desc']}"
            import time
            res = None
            for attempt in range(3):
                try:
                    res = genai.embed_content(model=EMBEDDING_MODEL, content=text, task_type="retrieval_document")
                    break
                except Exception as e:
                    print(f"Error in embed_content (attempt {attempt}):", e)
                    time.sleep(2 ** attempt)
            
            if res and 'embedding' in res:
                vector_blob = sqlite_vec.serialize_float32(res['embedding'])
                with sqlite_lock:
                    conn = get_db_connection()
                    c = conn.cursor()
                    c.execute(
                        "INSERT INTO anchor_metadata (anchor_id, title, subtitle, section_type) VALUES (?, ?, ?, 'products')",
                        (a['id'], a['title'], a['subtitle'])
                    )
                    c.execute(
                        "INSERT INTO anchor_vectors (anchor_id, embedding) VALUES (?, ?)",
                        (a['id'], vector_blob)
                    )
                    conn.commit()
                    conn.close()
        print("Vectores ancla sembrados en segundo plano exitosamente.")
    except Exception as e:
        print("Error seeding anchors en bg:", e)

@app.post("/api/seed-anchors")
def seed_anchors_endpoint(background_tasks: BackgroundTasks):
    """Siembra los vectores ancla base en SQLite en bg para evitar Timeout."""
    background_tasks.add_task(do_seed_anchors)
    return {"status": "processing", "message": "Vectores ancla sembr├índose en segundo plano."}

@app.post("/api/sync")
def sync_database():
    """Descarga todos los comercios y productos de Firestore y reconstruye el ├¡ndice SQLite."""
    try:
        if not db:
            raise HTTPException(status_code=500, detail="Firebase no est├í inicializado.")
        
        with sqlite_lock:
            conn = get_db_connection()
            c = conn.cursor()
            
            # Vaciar el ├¡ndice actual
            c.execute("DELETE FROM search_index")
            c.execute("DELETE FROM promotions")
            conn.commit()
            conn.close()
    
        # 1. Leer Promociones desde marketing_campaigns
        import time
        now_ms = int(time.time() * 1000)
        
        camps_ref = db.collection("marketing_campaigns")
        # Filtramos solo activas y tipo banner, o filtramos localmente para simplificar
        camps = list(camps_ref.stream())
        
        count_banners = 0
        if len(camps) > 0:
            with sqlite_lock:
                conn = get_db_connection()
                c = conn.cursor()
                for promo in camps:
                    p_data = promo.to_dict()
                    if p_data.get('type') in ['simple', 'premium_product', 'premium_store']:
                        # Validar estado y expiraci├│n
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
                conn.commit()
                conn.close()

        if count_banners == 0:
            # Fallback si no hay banners
            default_ads = [
                ("1", "simple", "store", "", "", "local-offer", "Descubre Ofertas", "En los mejores comercios", "#FFE4E1", "#DC143C", "#CD5C5C")
            ]
            with sqlite_lock:
                conn = get_db_connection()
                c = conn.cursor()
                for ad in default_ads:
                    c.execute("INSERT INTO promotions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", ad)
                conn.commit()
                conn.close()

        # 2. Leer Comercios
        stores_ref = db.collection("stores")
        stores = stores_ref.stream()
        
        count = 0
        for store in stores:
            s_data = store.to_dict()
            s_id = store.id
            
            with sqlite_lock:
                conn = get_db_connection()
                c = conn.cursor()
                # Insertar el comercio en el ├¡ndice
                c.execute("""
                    INSERT INTO search_index (id, type, storeId, name, category, description, price, icon, imageUrl, onSale, salePrice, likes, views, purchases, available, isOpen)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0, NULL, 0, 0, 0, 1, ?)
                """, (
                    s_id, 'store', s_id, 
                    s_data.get('name', ''), 
                    s_data.get('category', ''), 
                    '', '', '', s_data.get('logoUrl', s_data.get('imageUrl', '')),
                                1 if s_data.get('isOpen', True) else 0
                            ))
                count += 1
                conn.commit()
                conn.close()
            
            # Leer los productos de este comercio (Sub-colecci├│n)
            products_ref = stores_ref.document(s_id).collection("products")
            products = products_ref.stream()
            
            with sqlite_lock:
                conn = get_db_connection()
                c = conn.cursor()
                for product in products:
                    p_data = product.to_dict()
                    c.execute("""
                        INSERT INTO search_index (id, type, storeId, name, category, description, price, icon, imageUrl, onSale, salePrice, likes, views, purchases, available, isOpen)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
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
                        p_data.get('purchases', 0),
                                1 if p_data.get('available', True) else 0
                            ))
                    count += 1
                    if p_data.get('available', True):
                        vector_worker_pool.submit(
                            async_index_product_vector, 
                            product.id, 
                            p_data.get('name', ''), 
                            p_data.get('category', ''), 
                            p_data.get('description', '')
                        )
                conn.commit()
                conn.close()
        
        return {"message": "Sincronizaci├│n exitosa", "items_indexed": count}
    except Exception as e:
        print("Sync Error:", e)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/sync/store/{store_id}")
def sync_store(store_id: str):
    """Sincroniza un solo comercio y sus productos (M├ís r├ípido)."""
    if not db:
        raise HTTPException(status_code=500, detail="Firebase no est├í inicializado.")
        
    with sqlite_lock:
        conn = get_db_connection()
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
                INSERT INTO search_index (id, type, storeId, name, category, description, price, icon, imageUrl, onSale, salePrice, likes, views, purchases, available, isOpen)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0, NULL, 0, 0, 0, 1, ?)
            """, (
                store_id, 'store', store_id, 
                s_data.get('name', ''), 
                s_data.get('category', ''), 
                '', '', '', s_data.get('logoUrl', s_data.get('imageUrl', '')),
                            1 if s_data.get('isOpen', True) else 0
                        ))
            count += 1
            
            # 3. Leer Productos
            products_ref = store_ref.collection("products")
            for product in products_ref.stream():
                p_data = product.to_dict()
                c.execute("""
                    INSERT INTO search_index (id, type, storeId, name, category, description, price, icon, imageUrl, onSale, salePrice, likes, views, purchases, available, isOpen)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
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
                    p_data.get('purchases', 0),
                            1 if p_data.get('available', True) else 0
                        ))
                count += 1

        conn.commit()
        conn.close()
        return {"message": f"Comercio {store_id} sincronizado", "items_indexed": count}

def build_cluster_fts_query(cluster_name, c_val, include_cluster_name=True):
    cluster_match = c_val.get("keywords", "")
    cluster_words = [w.strip() for w in cluster_match.split(" OR ") if w.strip()]
    
    if include_cluster_name and cluster_name not in cluster_words:
        cluster_words.append(cluster_name)
        
    parts = [f'"{w}"*' for w in cluster_words]
    if not parts:
        return ""
    base_fts = " OR ".join(parts)
    
    neg_keywords_str = c_val.get("negativeKeywords", "")
    neg_parts = []
    if neg_keywords_str:
        neg_words = [w.strip() for w in neg_keywords_str.split(" OR ") if w.strip()]
        if neg_words:
            neg_group = " OR ".join([f'"{w}"*' for w in neg_words])
            neg_parts.append(f"NOT ({neg_group})")
            
    store_cats_str = c_val.get("storeCategories", "")
    cat_parts = []
    if store_cats_str:
        cats = [cat.strip() for cat in store_cats_str.split(",") if cat.strip()]
        if cats:
            # FTS5 column filter syntax: {column} : "term"*
            cat_terms = " OR ".join([f'"{cat}"*' for cat in cats])
            cat_parts.append(f"{{category}} : ({cat_terms})")
            
    fts_query_parts = []
    if cat_parts:
        fts_query_parts.append(f"({cat_parts[0]}) AND ({base_fts})")
    else:
        fts_query_parts.append(f"({base_fts})")
        
    if neg_parts:
        fts_query_parts.append(neg_parts[0])
        
    return " ".join(fts_query_parts)

@app.get("/api/search")
def search(q: str = ""):
    """Busca en milisegundos en el ├¡ndice FTS5 usando sin├│nimos y Fuzzy Match."""
    if not q.strip():
        return {"results": []}
    
    conn = sqlite3.connect(SQLITE_DB)
    # Devolver filas como diccionarios
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    
    safe_q = q.replace('"', '').replace("'", "").lower().strip()
    
    # === CEREBRO V2 INJECTION ===
    cluster_match = None
    cluster_name = None
    for c_key, c_val in MACRO_CLUSTERS_CACHE.items():
        if safe_q == c_key or safe_q in [k.strip().lower() for k in c_val.get("keywords", "").split(" OR ")]:
            cluster_match = True
            cluster_name = c_key
            break
            
    if cluster_match:
        fts_query = build_cluster_fts_query(cluster_name, MACRO_CLUSTERS_CACHE[cluster_name], True)
    else:
        words = safe_q.split()
        expanded_parts = []
        for word in words:
            if word in REVERSE_SYNONYMS:
                root = REVERSE_SYNONYMS[word]
                syns = SYNONYMS[root]
                group = " OR ".join([f'"{s}"*' for s in syns])
                expanded_parts.append(f"({group})")
            else:
                expanded_parts.append(f'"{word}"*')
                
        fts_query = " ".join(expanded_parts)
    
    try:
        # Buscamos en todas las columnas y ordenamos por "rank" (relevancia autom├ítica de SQLite FTS5)
        c.execute("""
            SELECT p.id, p.type, p.storeId, p.name, p.category, p.description,
                   p.price, p.icon, p.imageUrl, p.onSale, p.salePrice, p.likes, p.views, p.purchases,
                   s.name as storeName
            FROM search_index p
            LEFT JOIN (SELECT id, name FROM search_index WHERE type='store') s ON s.id = p.storeId
            WHERE search_index MATCH ?
            ORDER BY 
                rank - (
                    (COALESCE(CAST(p.likes AS REAL), 0) * 0.1) + 
                    (COALESCE(CAST(p.purchases AS REAL), 0) * 0.2) + 
                    (CASE WHEN COALESCE(CAST(p.views AS INTEGER), 0) < 50 THEN ABS(RANDOM() % 10) / 10.0 ELSE 0 END)
                )
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
                ORDER BY 
                    (COALESCE(CAST(p.likes AS REAL), 0) * 10.0) +
                    (COALESCE(CAST(p.purchases AS REAL), 0) * 15.0) +
                    (COALESCE(CAST(p.views AS REAL), 0) * 0.5) +
                    (CASE WHEN COALESCE(CAST(p.views AS INTEGER), 0) < 50 THEN ABS(RANDOM() % 100) ELSE 0 END) +
                    ABS(RANDOM() % 20) DESC
                LIMIT 50
            """, (like_q, like_q, like_q))
            
            rows_like = c.fetchall()
            results = [dict(row) for row in rows_like]

        # FALLBACK 2: FUZZY (Si no encontr├│ nada y la query tiene al menos 3 letras)
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
                        # Si el nombre del item fue uno de los que matche├│
                        if item["name"] in matches and item["id"] not in seen_ids:
                            results.append(dict(item))
                            seen_ids.add(item["id"])
                    
                    results.sort(key=lambda x: matches.index(x["name"]) if x["name"] in matches else 999)
                    
    except Exception as e:
        print("Error de b├║squeda:", e)
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

class HomeFeedRequest(BaseModel):
    activities: List[dict] = []
    lat: float = None
    lng: float = None

class ManualAnchorRequest(BaseModel):
    title: str
    subtitle: str
    desc: str
    allowed_categories: List[str] = []
    exclude_rules: List[str] = []
    titles: List[str] = []

class SimulateRequest(BaseModel):
    prompt: str

@app.post("/api/simulate")
def simulate_home_feed(req: SimulateRequest):
    """Simulador para probar el Cerebro Vectorial en el panel de Admin"""
    try:
        import google.generativeai as genai
        
        # 1. Generar vector para el prompt del admin
        res = genai.embed_content(
            model=EMBEDDING_MODEL,
            content=req.prompt,
            task_type="retrieval_query"
        )
        if not res or 'embedding' not in res:
            return {"status": "error", "error": "No se pudo generar el embedding."}
            
        sim_vector = sqlite_vec.serialize_float32(res['embedding'])
        
        conn = get_db_connection()
        c = conn.cursor()
        
        # 2. Buscar las 3 anclas m├ís afines al prompt
        c.execute("""
            SELECT a.anchor_id, m.title, m.subtitle, m.allowed_categories, m.exclude_rules, m.titles, vec_distance_cosine(a.embedding, ?) AS distance
            FROM anchor_vectors a
            JOIN anchor_metadata m ON a.anchor_id = m.anchor_id
            ORDER BY distance ASC
            LIMIT 3
        """, (sim_vector,))
        anchors = [dict(row) for row in c.fetchall()]
        
        feed_sections = []
        
        # 3. Para cada ancla, traer 30 productos y filtrar los malos
        for anchor in anchors:
            c.execute("""
                SELECT p.product_id, vec_distance_cosine(p.embedding, a.embedding) AS distance,
                       s.name, s.category, s.price, s.imageUrl, s.storeId
                FROM product_vectors p
                JOIN anchor_vectors a ON a.anchor_id = ?
                JOIN search_index s ON s.id = p.product_id
                ORDER BY distance ASC
                LIMIT 30
            """, (anchor['anchor_id'],))
            
            raw_items = c.fetchall()
            
            import json
            allowed_categories = []
            if anchor.get("allowed_categories"):
                try: allowed_categories = [cat.lower() for cat in json.loads(anchor["allowed_categories"])]
                except: pass
                
            exclude_rules = []
            if anchor.get("exclude_rules"):
                try: exclude_rules = json.loads(anchor["exclude_rules"])
                except: pass
                
            items = []
            for raw_row in raw_items:
                row = dict(raw_row)
                if row["distance"] > 0.8: # Umbral de similitud
                    continue
                
                cat = str(row.get("category", "")).lower()
                # Positive Mapping Filter
                if allowed_categories and cat not in allowed_categories:
                    continue
                
                # FILTRO INFALIBLE: Negativas
                name_cat = (str(row.get("name", "")) + " " + cat).lower()
                is_excluded = False
                for rule in exclude_rules:
                    if rule and rule.lower() in name_cat:
                        is_excluded = True
                        break
                if is_excluded: continue
                
                items.append({
                    "id": row["product_id"],
                    "name": row["name"],
                    "category": row["category"],
                    "price": row["price"],
                    "imageUrl": row["imageUrl"],
                    "storeId": row["storeId"],
                    "storeName": "Tienda Simulada"
                })
                if len(items) >= 10: break
                
            if items:
                import random
                import json
                anchor_title = anchor.get("title", "Explorar")
                if anchor.get("titles"):
                    try:
                        titles_list = json.loads(anchor["titles"])
                        if titles_list:
                            anchor_title = random.choice(titles_list)
                    except: pass
                    
                feed_sections.append({
                    "id": f"sim_anchor_{anchor['anchor_id']}",
                    "title": anchor_title,
                    "subtitle": anchor['subtitle'],
                    "section_type": "interest",
                    "items": items
                })
                
        conn.close()
        return {"results": feed_sections}
    except Exception as e:
        import traceback
        traceback.print_exc()
        return []

@app.post("/api/home/{uid}")
def get_dynamic_home_feed(uid: str, req: HomeFeedRequest):
    """V6: Cerebro Vectorial Unificado. Todo pasa por KNN con Gravedad Dinámica."""
    feed_sections = []
    
    conn = get_db_connection()
    c = conn.cursor()
        
    user_vector = None
    activities = req.activities
    
    if activities:
        try:
            from datetime import datetime, timezone
            now = datetime.now(timezone.utc)
            def calc_decay(ts):
                if not ts: return 1.0
                try:
                    if hasattr(ts, 'timestamp'): pass
                    elif isinstance(ts, str):
                        try:
                            ts = datetime.fromisoformat(ts.replace('Z', '+00:00'))
                        except:
                            try:
                                ts = datetime.fromtimestamp(float(ts)/1000, tz=timezone.utc)
                            except: return 1.0
                    elif isinstance(ts, (int, float)):
                        ts = datetime.fromtimestamp(ts/1000, tz=timezone.utc)
                    else:
                        return 1.0
                        
                    days_ago = (now - ts).days
                    if days_ago == 0: return 3.0
                    elif days_ago > 7: return 0.2
                    elif days_ago > 30: return 0.0
                    return 1.0
                except: return 1.0
                
            user_vector = calculate_user_vector(activities, calc_decay, current_hour=datetime.now().hour)
        except Exception as e:
            print("Error generando vector de usuario desde local:", e)
            
    global_seen_ids = set()
    
    # ============================================
    # PASO 1: DETECTAR CONTEXTO AMBIENTAL (Clima + Hora)
    # ============================================
    from datetime import datetime, timezone
    current_hour = (datetime.now(timezone.utc).hour - 5) % 24
    
    # Mapa de gravedad: rule_type → bias (se resta a la distancia coseno)
    gravity_map = {}
    
    # 1a. Gravedad Horaria
    if 5 <= current_hour <= 10:
        gravity_map["hora_desayuno"] = 0.12
    elif 11 <= current_hour <= 14:
        gravity_map["hora_almuerzo"] = 0.12
    elif 18 <= current_hour <= 23 or current_hour < 2:
        gravity_map["hora_noche"] = 0.12
        gravity_map["hora_cena"] = 0.12
    elif 14 < current_hour < 18:
        gravity_map["hora_tarde"] = 0.08
    
    # 1b. Gravedad Climática
    weather_temp = None
    if req.lat is not None and req.lng is not None:
        try:
            import requests
            import time as _time
            lat_key = round(req.lat, 1)
            lng_key = round(req.lng, 1)
            loc_key = f"{lat_key}_{lng_key}"
            
            now_ts = _time.time()
            if loc_key in WEATHER_CACHE_STORE and (now_ts - WEATHER_CACHE_STORE[loc_key]["time"] < 3600):
                weather_temp = WEATHER_CACHE_STORE[loc_key]["temp"]
                weather_code = WEATHER_CACHE_STORE[loc_key]["code"]
            else:
                w_res = requests.get(f"https://api.open-meteo.com/v1/forecast?latitude={lat_key}&longitude={lng_key}&current_weather=true", timeout=2).json()
                if "current_weather" in w_res:
                    weather_temp = w_res["current_weather"].get("temperature", 20)
                    weather_code = w_res["current_weather"].get("weathercode", 0)
                    WEATHER_CACHE_STORE[loc_key] = {"temp": weather_temp, "code": weather_code, "time": now_ts}
                    
            if weather_temp is not None:
                if weather_temp >= 24:
                    gravity_map["clima_calor"] = 0.15
                elif weather_temp <= 16 or weather_code >= 50:
                    gravity_map["clima_frio"] = 0.15
        except Exception as e:
            print(f"[Weather] Error: {e}")
    
    # ============================================
    # PASO 2: OBTENER TODAS LAS ANCLAS CON DISTANCIA + GRAVEDAD
    # ============================================
    anchors = []
    try:
        if user_vector:
            c.execute("""
                SELECT a.anchor_id, m.title, m.subtitle, m.allowed_categories, 
                       m.exclude_rules, m.titles, m.rule_type,
                       vec_distance_cosine(a.embedding, ?) AS distance
                FROM anchor_vectors a
                JOIN anchor_metadata m ON a.anchor_id = m.anchor_id
                ORDER BY distance ASC
            """, (user_vector,))
        else:
            # Usuarios nuevos: traer todas las anclas con distancia neutra
            c.execute("""
                SELECT a.anchor_id, m.title, m.subtitle, m.allowed_categories, 
                       m.exclude_rules, m.titles, m.rule_type,
                       0.5 AS distance
                FROM anchor_vectors a
                JOIN anchor_metadata m ON a.anchor_id = m.anchor_id
            """)
        
        all_anchors = [dict(row) for row in c.fetchall()]
        
        # Aplicar Gravedad Dinámica
        for anchor in all_anchors:
            rule = anchor.get("rule_type") or "general"
            gravity_bias = gravity_map.get(rule, 0.0)
            anchor["gravity_distance"] = anchor["distance"] - gravity_bias
        
        # Re-ordenar por distancia gravitacional
        all_anchors.sort(key=lambda x: x["gravity_distance"])
        
        # Tomar las top 4 mejores (o menos si no hay suficientes)
        top_anchors = all_anchors[:4]
        
        # Anti-Burbuja Vectorial: agregar la ancla MÁS LEJANA al usuario como exploración
        if len(all_anchors) > 5:
            # Buscar la más lejana que no esté en el top
            top_ids = {a["anchor_id"] for a in top_anchors}
            for far_anchor in reversed(all_anchors):
                if far_anchor["anchor_id"] not in top_ids:
                    far_anchor["title"] = "Sal de la rutina"
                    far_anchor["subtitle"] = "Descubre algo totalmente nuevo"
                    far_anchor["_is_exploration"] = True
                    top_anchors.append(far_anchor)
                    break
        
        anchors = top_anchors
        
    except Exception as e:
        print(f"[V6 Gravity] Error obteniendo anclas: {e}")
    
    # ============================================
    # PASO 3: PARA CADA ANCLA, BUSCAR PRODUCTOS (KNN Vectorial)
    # ============================================
    import math
    import json
    import random
    
    for anchor in anchors:
        try:
            c.execute("""
                SELECT p.product_id, vec_distance_cosine(p.embedding, a.embedding) AS distance,
                       s.id, s.type, s.storeId, s.name, s.category, s.description,
                       s.price, s.icon, s.imageUrl, s.onSale, s.salePrice, s.likes, s.views, s.purchases,
                       st.name as storeName
                FROM product_vectors p
                JOIN anchor_vectors a ON a.anchor_id = ?
                JOIN search_index s ON p.product_id = s.id AND s.type = 'product'
                LEFT JOIN (SELECT id, name, isOpen FROM search_index WHERE type='store') st ON st.id = s.storeId
                WHERE CAST(s.available AS INTEGER) = 1 AND CAST(st.isOpen AS INTEGER) = 1
                ORDER BY distance ASC
                LIMIT 40
            """, (anchor["anchor_id"],))
            
            raw_items = c.fetchall()
            
            allowed_categories = []
            if anchor.get("allowed_categories"):
                try: allowed_categories = [cat.lower() for cat in json.loads(anchor["allowed_categories"])]
                except: pass
                
            exclude_rules = []
            if anchor.get("exclude_rules"):
                try: exclude_rules = json.loads(anchor["exclude_rules"])
                except: pass
                
            candidate_items = []
            
            for raw_row in raw_items:
                row = dict(raw_row)
                if row["distance"] > 0.8:
                    continue
                
                cat = str(row.get("category", "")).lower()
                if allowed_categories and cat not in allowed_categories:
                    continue
                    
                name_cat = (str(row.get("name", "")) + " " + cat).lower()
                is_excluded = False
                for rule in exclude_rules:
                    if rule and rule.lower() in name_cat:
                        is_excluded = True
                        break
                if is_excluded: continue
                
                rid = row["id"]
                if rid in global_seen_ids: continue
                
                # Hybrid Scoring
                affinity = max(0.0, 1.0 - row["distance"])
                purchases = float(row.get("purchases") or 0)
                likes = float(row.get("likes") or 0)
                views = float(row.get("views") or 0)
                
                popularity = math.log1p(purchases + likes * 0.5) / 10.0
                novelty = 0.2 if (purchases == 0 and views <= 15) else (-0.3 if purchases == 0 and views > 50 else 0.0)
                sale_boost = 0.15 if str(row.get("onSale", "0")) == "1" else 0.0
                
                row["final_score"] = (affinity * 0.6) + (popularity * 0.2) + (novelty * 0.1) + (sale_boost * 0.1)
                candidate_items.append(row)
                
            candidate_items.sort(key=lambda x: x["final_score"], reverse=True)
            
            store_counts = {}
            filtered_items = []
            
            for row in candidate_items:
                rid = row["id"]
                sid = row["storeId"]
                if store_counts.get(sid, 0) >= 4: continue
                
                filtered_items.append(row)
                global_seen_ids.add(rid)
                store_counts[sid] = store_counts.get(sid, 0) + 1
                
                if len(filtered_items) >= 6:
                    break
                    
            if len(filtered_items) >= 2:
                anchor_title = anchor.get("title", "Explorar")
                if not anchor.get("_is_exploration") and anchor.get("titles"):
                    try:
                        titles_list = json.loads(anchor["titles"])
                        if titles_list:
                            anchor_title = random.choice(titles_list)
                    except: pass
                    
                section_id = f"dyn_vector_{anchor['anchor_id']}"
                if anchor.get("_is_exploration"):
                    section_id = f"dyn_explore_{anchor['anchor_id']}"
                    
                feed_sections.append({
                    "id": section_id,
                    "type": "products",
                    "title": anchor_title,
                    "subtitle": anchor["subtitle"],
                    "items": filtered_items
                })
        except Exception as e:
            print(f"[V6] Error obteniendo productos para ancla {anchor.get('anchor_id', '?')}: {e}")

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
        
        # 2. Consultar nuestra base local ultrarr├ípida (SQLite) para buscar productos de esas categor├¡as
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
            # Si no hay suficientes productos, completamos con los m├ís populares
            return get_popular_products()
            
        return {"results": results}
    except Exception as e:
        print(f"Error generando recomendaciones para {uid}:", e)
        return get_popular_products()

@app.get("/api/status")
def get_system_status():
    """Devuelve m├®tricas del estado del sistema para el panel de administraci├│n."""
    try:
        conn = sqlite3.connect(SQLITE_DB)
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        
        c.execute("SELECT COUNT(*) as total FROM search_index WHERE type='product'")
        total_products = c.fetchone()["total"]
        
        c.execute("SELECT COUNT(*) as total FROM search_index WHERE type='store'")
        total_stores = c.fetchone()["total"]
        
        c.execute("SELECT COUNT(*) as total FROM promotions")
        total_promotions = c.fetchone()["total"]
        
        c.execute("SELECT value FROM metadata WHERE key='last_sync_time'")
        row = c.fetchone()
        last_sync = row["value"] if row else "Nunca"
        
        conn.close()
        
        return {
            "status": "ok",
            "cerebro_version": "V3.2",
            "total_products": total_products,
            "total_stores": total_stores,
            "total_promotions": total_promotions,
            "total_clusters": len(MACRO_CLUSTERS_CACHE),
            "total_time_rules": len(TIME_RULES_CACHE),
            "last_sync": last_sync,
            "sqlite_db": SQLITE_DB,
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}

@app.get("/api/admin/users-vectors")
def get_admin_users_vectors(page: int = 1, limit: int = 10):
    """Devuelve los perfiles vectoriales de los usuarios activos calculando su afinidad actual."""
    try:
        users_ref = db.collection('users')
        # Paginaci├│n b├ísica en Firestore
        offset = (page - 1) * limit
        users = users_ref.offset(offset).limit(limit).stream()
        
        # Para saber el total aprox
        total_users = 0 # Firestore count can be slow, but let's assume we return dynamic
        
        results = []
        conn = get_db_connection()
        c = conn.cursor()
        
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc)
        def calc_decay(ts):
            if not ts: return 1.0
            try:
                if hasattr(ts, 'timestamp'): pass
                elif isinstance(ts, str):
                    try:
                        ts = datetime.fromisoformat(ts.replace('Z', '+00:00'))
                    except:
                        try:
                            ts = datetime.fromtimestamp(float(ts)/1000, tz=timezone.utc)
                        except: return 1.0
                elif isinstance(ts, (int, float)):
                    ts = datetime.fromtimestamp(ts/1000, tz=timezone.utc)
                else: return 1.0
                    
                days_ago = (now - ts).days
                if days_ago == 0: return 3.0
                elif days_ago > 7: return 0.2
                elif days_ago > 30: return 0.0
                return 1.0
            except: return 1.0

        current_hour = (datetime.now(timezone.utc).hour - 5) % 24
        
        for u in users:
            uid = u.id
            udata = u.to_dict()
            recent_activity = udata.get('recent_activity', [])
            
            user_vector = calculate_user_vector(recent_activity, calc_decay, current_hour=current_hour)
            anchors = []
            if user_vector:
                c.execute("""
                    SELECT a.anchor_id, m.title, m.subtitle, vec_distance_cosine(a.embedding, ?) AS distance
                    FROM anchor_vectors a
                    JOIN anchor_metadata m ON a.anchor_id = m.anchor_id
                    ORDER BY distance ASC
                    LIMIT 2
                """, (user_vector,))
                anchors = [dict(row) for row in c.fetchall()]
            
            if len(recent_activity) > 0 or len(anchors) > 0:
                results.append({
                    "uid": u.id,
                    "name": udata.get('name', udata.get('email', 'Usuario An├│nimo')),
                    "activity_count": len(recent_activity),
                    "anchors": anchors
                })
                
        conn.close()
        return {"status": "ok", "users": results}
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"status": "error", "message": str(e)}

@app.post("/api/admin/cerebro/anchors")
def create_manual_anchor(req: ManualAnchorRequest):
    try:
        import uuid
        anchor_id = "M" + str(uuid.uuid4()).replace("-", "")[:8]
        primary_title = req.titles[0] if req.titles else req.title
        text = f"{primary_title} - {req.desc}"
        
        import time
        res = None
        for attempt in range(3):
            try:
                res = genai.embed_content(model=EMBEDDING_MODEL, content=text, task_type="retrieval_document")
                break
            except Exception as e:
                time.sleep(1)
                
        if not res or 'embedding' not in res:
            return {"status": "error", "message": "Failed to generate embedding"}
            
        vector_blob = sqlite_vec.serialize_float32(res['embedding'])
        with sqlite_lock:
            conn = get_db_connection()
            c = conn.cursor()
            c.execute(
                "INSERT INTO anchor_metadata (anchor_id, title, subtitle, section_type, allowed_categories, exclude_rules, titles, is_manual) VALUES (?, ?, ?, 'products', ?, ?, ?, 1)",
                (anchor_id, primary_title, req.subtitle, json.dumps(req.allowed_categories), json.dumps(req.exclude_rules), json.dumps(req.titles))
            )
            c.execute(
                "INSERT INTO anchor_vectors (anchor_id, embedding) VALUES (?, ?)",
                (anchor_id, vector_blob)
            )
            conn.commit()
            conn.close()
        return {"status": "ok", "anchor_id": anchor_id}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.put("/api/admin/cerebro/anchors/{anchor_id}")
def update_manual_anchor(anchor_id: str, req: ManualAnchorRequest):
    try:
        primary_title = req.titles[0] if req.titles else req.title
        text = f"{primary_title} - {req.desc}"
        
        import time
        res = None
        for attempt in range(3):
            try:
                res = genai.embed_content(model=EMBEDDING_MODEL, content=text, task_type="retrieval_document")
                break
            except Exception as e:
                time.sleep(1)
                
        if not res or 'embedding' not in res:
            return {"status": "error", "message": "Failed to generate embedding"}
            
        vector_blob = sqlite_vec.serialize_float32(res['embedding'])
        with sqlite_lock:
            conn = get_db_connection()
            c = conn.cursor()
            # Actualizar metadatos
            c.execute("""
                UPDATE anchor_metadata 
                SET title=?, subtitle=?, allowed_categories=?, exclude_rules=?, titles=?
                WHERE anchor_id=?
            """, (primary_title, req.subtitle, json.dumps(req.allowed_categories), json.dumps(req.exclude_rules), json.dumps(req.titles), anchor_id))
            # Actualizar vector
            c.execute("UPDATE anchor_vectors SET embedding=? WHERE anchor_id=?", (vector_blob, anchor_id))
            conn.commit()
            conn.close()
        return {"status": "ok"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.delete("/api/admin/cerebro/anchors/{anchor_id}")
def delete_manual_anchor(anchor_id: str):
    try:
        with sqlite_lock:
            conn = get_db_connection()
            c = conn.cursor()
            c.execute("DELETE FROM anchor_metadata WHERE anchor_id = ?", (anchor_id,))
            c.execute("DELETE FROM anchor_vectors WHERE anchor_id = ?", (anchor_id,))
            conn.commit()
            conn.close()
        return {"status": "ok"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/api/admin/cerebro")
def get_admin_cerebro(page: int = 1, limit: int = 10):
    """Devuelve telemetr├¡a detallada del Cerebro Vectorial para el panel Admin."""
    try:
        conn = get_db_connection()
        c = conn.cursor()
        
        # 1. Total Vectores Productos
        c.execute("SELECT COUNT(*) as c FROM product_vectors")
        total_product_vectors = c.fetchone()["c"]
        
        c.execute("""
            SELECT a.anchor_id, m.title, m.subtitle, m.section_type, a.embedding, m.is_manual, m.rule_type, m.allowed_categories, m.exclude_rules, m.titles
            FROM anchor_vectors a
            LEFT JOIN anchor_metadata m ON a.anchor_id = m.anchor_id
        """)
        
        anchors = []
        for row in c.fetchall():
            anchor_dict = dict(row)
            anchor_dict["rule_type"] = anchor_dict.get("rule_type") or "general"
                
            # Extraer vector para que el admin lo vea
            if anchor_dict.get("embedding"):
                import numpy as np
                vec_array = np.frombuffer(anchor_dict["embedding"], dtype=np.float32)
                anchor_dict["vector_preview"] = f"[{vec_array[0]:.3f}, {vec_array[1]:.3f}, {vec_array[2]:.3f}...]"
                del anchor_dict["embedding"] # remove binary
                
            anchors.append(anchor_dict)
        
        # 3. N productos vectorizados (Paginados)
        offset = (page - 1) * limit
        c.execute("""
            SELECT p.product_id, s.name, s.category, length(p.embedding) as vec_bytes
            FROM product_vectors p
            JOIN search_index s ON p.product_id = s.id AND s.type = 'product'
            LIMIT ? OFFSET ?
        """, (limit, offset))
        sample_products = [dict(row) for row in c.fetchall()]
        
        conn.close()
        
        return {
            "status": "ok",
            "vector_metrics": {
                "total_product_vectors": total_product_vectors,
                "anchors_count": len(anchors),
                "anchors": anchors,
                "sample_products": sample_products,
                "pagination": {
                    "page": page,
                    "limit": limit,
                    "total": total_product_vectors
                }
            }
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/admin/auto-generate-anchors")
def auto_generate_anchors(background_tasks: BackgroundTasks):
    def run_generation():
        try:
            with sqlite_lock:
                conn = get_db_connection()
                c = conn.cursor()
                c.execute("SELECT DISTINCT category FROM search_index WHERE type='product'")
                categories = [row['category'] for row in c.fetchall() if row['category']]
                c.execute("SELECT name, description, category FROM search_index WHERE type='product' ORDER BY RANDOM() LIMIT 100")
                products = [dict(row) for row in c.fetchall()]
                conn.close()
                
            import json
            import google.generativeai as genai
            
            prompt = f'''
            Eres un experto en taxonom├¡a de comercio electr├│nico e inteligencia artificial.
            Aqu├¡ tienes una muestra de los productos y categor├¡as de nuestro supermercado/tienda:
            Categor├¡as: {categories}
            Muestra de productos: {products}
            
            Tu tarea es generar un arreglo JSON con las mejores "Anclas" (Cl├║steres o categor├¡as sem├ínticas) para organizar este inventario en un motor de b├║squeda vectorial.
            El arreglo JSON debe contener entre 6 y 12 objetos con la siguiente estructura exacta:
            [
              {{
                "id": "A1",
                "titles": ["Mascotas", "Para tus peludos", "El rinc├│n animal", "Mascotas felices"],
                "subtitle": "Todo para tu mejor amigo",
                "desc": "Alimentos y accesorios para mascotas",
                "allowed_categories": ["Mascotas", "Veterinaria", "Animales"],
                "exclude_rules": ["perro caliente", "salchicha"]
              }}
            ]
            En "titles", DEBES dar un arreglo de 4 opciones de t├¡tulos atractivos y din├ímicos para esta categor├¡a.
            En "allowed_categories", debes poner un arreglo de strings seleccionando EXACTAMENTE los nombres de las categor├¡as proporcionadas en la lista 'Categor├¡as' que pertenecen a esta ancla. ESTO ES UN FILTRO ESTRICTO. Solo los productos de estas categor├¡as aparecer├ín en esta ancla. ┬íS├® exhaustivo e incluye todas las categor├¡as relevantes de la lista!
            En "exclude_rules", incluye un arreglo de palabras clave que NO deben aparecer (por si hay ambig├╝edad).
            Devuelve SOLO EL JSON v├ílido, sin c├│digo de bloque extra ni markdown.
            '''
            models_to_try = ["gemini-1.5-flash", "gemini-2.5-flash", "gemini-2.0-flash"]
            response = None
            for m in models_to_try:
                try:
                    import time
                    model = genai.GenerativeModel(m)
                    # Use standard generate_content
                    response = model.generate_content(prompt)
                    if response:
                        print(f"Modelo {m} seleccionado exitosamente para generación.")
                        break
                except Exception as e:
                    print(f"Modelo {m} falló: {e}")
                    time.sleep(2)
            if not response:
                raise Exception("Todos los modelos generativos fallaron o no est├ín disponibles en esta API Key.")
                
            raw_text = response.text.strip()
            if raw_text.startswith("```json"): raw_text = raw_text[7:]
            if raw_text.startswith("```"): raw_text = raw_text[3:]
            if raw_text.endswith("```"): raw_text = raw_text[:-3]
            
            anchors_data = json.loads(raw_text.strip())
            
            with sqlite_lock:
                conn = get_db_connection()
                c = conn.cursor()
                
                # Obtener los IDs de anclas generadas por IA (no manuales)
                c.execute("SELECT anchor_id FROM anchor_metadata WHERE is_manual = 0")
                old_ai_anchors = [row[0] for row in c.fetchall()]
                
                for oid in old_ai_anchors:
                    c.execute("DELETE FROM anchor_metadata WHERE anchor_id = ?", (oid,))
                    c.execute("DELETE FROM anchor_vectors WHERE anchor_id = ?", (oid,))
                    
                conn.commit()
                conn.close()
                
            for a in anchors_data:
                primary_title = a.get('titles', [a.get('title', 'Explorar')])[0]
                text = f"{primary_title} - {a.get('desc', '')}"
                import time
                res = None
                for attempt in range(3):
                    try:
                        res = genai.embed_content(model=EMBEDDING_MODEL, content=text, task_type="retrieval_document")
                        break
                    except Exception as e:
                        time.sleep(2 ** attempt)
                
                if res and 'embedding' in res:
                    vector_blob = sqlite_vec.serialize_float32(res['embedding'])
                    with sqlite_lock:
                        conn = get_db_connection()
                        c = conn.cursor()
                        c.execute(
                            "INSERT INTO anchor_metadata (anchor_id, title, subtitle, section_type, allowed_categories, exclude_rules, titles) VALUES (?, ?, ?, 'products', ?, ?, ?)",
                            (a['id'], primary_title, a.get('subtitle', ''), json.dumps(a.get('allowed_categories', [])), json.dumps(a.get('exclude_rules', [])), json.dumps(a.get('titles', [])))
                        )
                        c.execute(
                            "INSERT INTO anchor_vectors (anchor_id, embedding) VALUES (?, ?)",
                            (a['id'], vector_blob)
                        )
                        conn.commit()
                        conn.close()
            print("[Fase 1] Auto-Generación de Anclas con IA completada exitosamente.")
            
            # --- FASE 2: ANCLAS AMBIENTALES VECTORIALES (Clima + Hora) ---
            prompt_ambient = f'''
            Eres un experto en comportamiento del consumidor. Revisa esta muestra de productos y categorías:
            Categorías: {categories}
            Muestra: {products}
            
            Genera anclas ambientales (contextuales) que se activan según clima y hora del día.
            Cada ancla es un cluster semántico con productos relevantes para ese momento.
            
            Devuelve un JSON array con esta estructura EXACTA (SOLO JSON VÁLIDO, SIN MARKDOWN):
            [
              {{
                "id": "AMB_calor",
                "rule_type": "clima_calor",
                "titles": ["Para este calorcito ☀️", "Refresca tu día", "Combate el calor"],
                "subtitle": "Productos frescos para ti",
                "desc": "Helados, jugos, bebidas frías, ropa ligera, protector solar",
                "allowed_categories": ["Heladería", "Jugos", "Ropa"],
                "exclude_rules": ["sopa", "chocolate caliente"]
              }},
              {{
                "id": "AMB_frio",
                "rule_type": "clima_frio",
                "titles": ["Días fríos 🌧️", "Entra en calor", "Para el frío"],
                "subtitle": "Abrígate y come rico",
                "desc": "Café, sopas, chocolate, ropa abrigada, cobijas",
                "allowed_categories": ["Cafetería", "Restaurante", "Ropa"],
                "exclude_rules": ["helado", "pantaloneta"]
              }},
              {{
                "id": "AMB_desayuno",
                "rule_type": "hora_desayuno",
                "titles": ["Buenos días ☕", "Empieza con energía"],
                "subtitle": "Para el desayuno",
                "desc": "Café, pan, arepas, huevos, jugo de naranja",
                "allowed_categories": ["Cafetería", "Panadería"],
                "exclude_rules": []
              }},
              {{
                "id": "AMB_noche",
                "rule_type": "hora_noche",
                "titles": ["Antojos nocturnos 🌙", "Para la noche"],
                "subtitle": "Algo rico antes de dormir",
                "desc": "Pizza, hamburguesa, helado, cerveza, licor, domicilios",
                "allowed_categories": ["Comida Rápida", "Licorería"],
                "exclude_rules": []
              }}
            ]
            
            REGLAS:
            - rule_type DEBE ser uno de: "clima_calor", "clima_frio", "hora_desayuno", "hora_almuerzo", "hora_tarde", "hora_noche", "hora_cena"
            - En allowed_categories usa EXACTAMENTE los nombres de la lista 'Categorías' proporcionada
            - Genera entre 4 y 8 anclas ambientales
            - En "titles" incluye 3-4 títulos atractivos y variados
            '''
            ambient_response = None
            for m in models_to_try:
                try:
                    import time
                    model = genai.GenerativeModel(m)
                    ambient_response = model.generate_content(prompt_ambient)
                    if ambient_response: break
                except Exception as e:
                    print(f"Modelo {m} falló en Fase 2: {e}")
                    time.sleep(2)
            
            if ambient_response:
                r_text = ambient_response.text.strip()
                if r_text.startswith("```json"): r_text = r_text[7:]
                if r_text.startswith("```"): r_text = r_text[3:]
                if r_text.endswith("```"): r_text = r_text[:-3]
                
                ambient_anchors = json.loads(r_text.strip())
                
                for amb in ambient_anchors:
                    primary_title = amb.get('titles', [amb.get('title', 'Explorar')])[0]
                    text = f"{primary_title} - {amb.get('desc', '')}"
                    import time
                    res = None
                    for attempt in range(3):
                        try:
                            res = genai.embed_content(model=EMBEDDING_MODEL, content=text, task_type="retrieval_document")
                            break
                        except Exception as e:
                            time.sleep(2 ** attempt)
                    
                    if res and 'embedding' in res:
                        vector_blob = sqlite_vec.serialize_float32(res['embedding'])
                        with sqlite_lock:
                            conn = get_db_connection()
                            c = conn.cursor()
                            c.execute(
                                "INSERT OR REPLACE INTO anchor_metadata (anchor_id, title, subtitle, section_type, allowed_categories, exclude_rules, titles, rule_type) VALUES (?, ?, ?, 'products', ?, ?, ?, ?)",
                                (amb['id'], primary_title, amb.get('subtitle', ''), json.dumps(amb.get('allowed_categories', [])), json.dumps(amb.get('exclude_rules', [])), json.dumps(amb.get('titles', [])), amb.get('rule_type', 'general'))
                            )
                            c.execute(
                                "INSERT OR REPLACE INTO anchor_vectors (anchor_id, embedding) VALUES (?, ?)",
                                (amb['id'], vector_blob)
                            )
                            conn.commit()
                            conn.close()
                print("[Fase 2] Anclas Ambientales Vectoriales generadas exitosamente.")
            
        except Exception as e:
            print("Error en Auto-Generacion (Fase 1/2):", e)
            
    background_tasks.add_task(run_generation)
    return {"status": "ok", "message": "Descubrimiento de anclas con IA iniciado en background. Espera un minuto."}

@app.post("/api/admin/reset-vectors")
def reset_vectors_db():
    try:
        with sqlite_lock:
            conn = get_db_connection()
            c = conn.cursor()
            c.execute("DROP TABLE IF EXISTS product_vectors")
            c.execute("DROP TABLE IF EXISTS anchor_vectors")
            c.execute("DELETE FROM anchor_metadata")
            conn.commit()
            conn.close()
        
        init_db()
        return {"status": "ok", "message": "Vectores limpiados correctamente."}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/api/reset-clusters")
def reset_clusters_to_defaults():
    """Empuja los defaults del c├│digo a Firestore, reemplazando los cl├║steres existentes.
    ├Ütil cuando los cl├║steres en Firestore est├ín desactualizados (sin storeCategories, etc.)."""
    if not db:
        raise HTTPException(status_code=500, detail="Firebase no est├í inicializado.")
    try:
        doc_ref = db.collection('config').document('algorithm')
        doc_ref.set({"clusters": MACRO_CLUSTERS_CACHE}, merge=True)
        return {
            "message": f"Ô£à {len(MACRO_CLUSTERS_CACHE)} cl├║steres reseteados a los defaults V3.2 correctamente.",
            "clusters_pushed": list(MACRO_CLUSTERS_CACHE.keys())
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ==========================================
# ==========================================
# WEBHOOKS PUSH PARA ACTUALIZAR ├ìNDICE (MINI-ALGOLIA)
# ==========================================

class ProductPayload(BaseModel):
    id: str
    storeId: str
    name: str
    category: Optional[str] = ""
    description: Optional[str] = ""
    price: Optional[float] = 0
    icon: Optional[str] = ""
    imageUrl: Optional[str] = ""
    isOpen: Optional[bool] = True
    onSale: Optional[bool] = False
    salePrice: Optional[float] = None
    likes: Optional[int] = 0
    views: Optional[int] = 0
    purchases: Optional[int] = 0
    available: Optional[bool] = True

class StorePayload(BaseModel):
    id: str
    name: str
    category: Optional[str] = ""
    imageUrl: Optional[str] = ""
    isOpen: Optional[bool] = True

@app.post("/api/index/product")
def index_product(payload: ProductPayload):
    with sqlite_lock:
        conn = sqlite3.connect(SQLITE_DB, timeout=10.0)
        c = conn.cursor()
        c.execute("DELETE FROM search_index WHERE id = ? AND type = 'product'", (payload.id,))
        c.execute("""
            INSERT INTO search_index (id, type, storeId, name, category, description, price, icon, imageUrl, onSale, salePrice, likes, views, purchases, available, isOpen)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
        """, (
            payload.id, 'product', payload.storeId, 
            payload.name, payload.category, payload.description, 
            str(payload.price), payload.icon, payload.imageUrl,
            1 if payload.onSale else 0, payload.salePrice,
            payload.likes, payload.views, payload.purchases,
                        1 if getattr(payload, 'available', True) else 0
                    ))
        conn.commit()
        conn.close()
    vector_worker_pool.submit(async_index_product_vector, payload.id, payload.name, payload.category, payload.description)
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
            INSERT INTO search_index (id, type, storeId, name, category, description, price, icon, imageUrl, onSale, salePrice, likes, views, purchases, available, isOpen)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0, NULL, 0, 0, 0, 1, ?)
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
                        INSERT INTO search_index (id, type, storeId, name, category, description, price, icon, imageUrl, onSale, salePrice, likes, views, purchases, available, isOpen)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0, NULL, 0, 0, 0, 1, ?)
                    """, (
                        s_id, 'store', s_id, 
                        s_data.get('name', ''), 
                        s_data.get('category', ''), 
                        '', '', '', s_data.get('logoUrl', s_data.get('imageUrl', '')),
                        1 if s_data.get('isOpen', True) else 0
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
                conn.close()
            
            if not last_sync_str:
                print("[Delta Sync] Primer arranque o SQLite vac├¡o. Sincronizando todo el cat├ílogo...")
                try:
                    sync_database() # Llamamos a la sincronizaci├│n completa
                    current_time = datetime.now(timezone.utc).isoformat()
                    with sqlite_lock:
                        conn = sqlite3.connect(SQLITE_DB, timeout=10.0)
                        c = conn.cursor()
                        c.execute("INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)", ('last_sync_time', current_time))
                        conn.commit()
                        conn.close()
                    print("[Delta Sync] Sincronizaci├│n inicial completada.")
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
                                INSERT INTO search_index (id, type, storeId, name, category, description, price, icon, imageUrl, onSale, salePrice, likes, views, purchases, available, isOpen)
                                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0, NULL, 0, 0, 0, 1, ?)
                            """, (
                                s_id, 'store', s_id, 
                                s_data.get('name', ''), 
                                s_data.get('category', ''), 
                                '', '', '', s_data.get('logoUrl', s_data.get('imageUrl', '')),
                        1 if s_data.get('isOpen', True) else 0
                    ))
                            
                        for prod in changed_products:
                            p_data = prod.to_dict()
                            p_id = prod.id
                            path_parts = prod.reference.path.split('/')
                            store_id = path_parts[1] if len(path_parts) >= 2 else ""
                            c.execute("DELETE FROM search_index WHERE id = ? AND type = 'product'", (p_id,))
                            if p_data.get('available', True):
                                c.execute("""
                                    INSERT INTO search_index (id, type, storeId, name, category, description, price, icon, imageUrl, onSale, salePrice, likes, views, purchases, available, isOpen)
                                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
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
                                    p_data.get('purchases', 0),
                        1 if p_data.get('available', True) else 0
                    ))
                                vector_worker_pool.submit(async_index_product_vector, p_id, p_data.get('name', ''), p_data.get('category', ''), p_data.get('description', ''))
                        
                        current_time = datetime.now(timezone.utc).isoformat()
                        c.execute("INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)", ('last_sync_time', current_time))
                        conn.commit()
                        conn.close()
                        print(f"[Delta Sync] Sincronizaci├│n exitosa. Siguiente chequeo desde {current_time}.")

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
            # Eliminar actividad de m├ís de 30 d├¡as
            thirty_days_ago = datetime.now(timezone.utc) - timedelta(days=30)
            
            old_activities = db.collection_group('activity').where(filter=FieldFilter("timestamp", "<", thirty_days_ago)).limit(500).stream()
            
            deleted_count = 0
            for doc in old_activities:
                doc.reference.delete()
                deleted_count += 1
                
            if deleted_count > 0:
                print(f"[Cleanup] Eliminados {deleted_count} registros de actividad antiguos.")
                
        except Exception as e:
            print(f"[Cleanup Error]: Requiere ├¡ndice. {e}")
            
        # Esperar 24 horas (86400 segundos)
        time.sleep(86400)

@app.on_event("startup")
def startup_event():
    print("Search backend is ready. Iniciando procesos en segundo plano...")
    threading.Thread(target=delta_sync_loop, daemon=True).start()
    threading.Thread(target=cleanup_activity_loop, daemon=True).start()
