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
# Habilitar CORS para que la app móvil o web pueda consultar sin bloqueos
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
    print(f"ADVERTENCIA: No se encontró '{SERVICE_ACCOUNT_FILE}' ni la variable FIREBASE_SERVICE_ACCOUNT. El endpoint /api/sync fallará.")

@app.post("/api/sync")
def sync_database():
    """Descarga todos los comercios y productos de Firestore y reconstruye el índice SQLite."""
    try:
        if not db:
            raise HTTPException(status_code=500, detail="Firebase no está inicializado.")
        
        with sqlite_lock:
            conn = get_db_connection()
            c = conn.cursor()
            
            # Vaciar el índice actual
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
                # Insertar el comercio en el índice
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
            
            # Leer los productos de este comercio (Sub-colección)
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
        
        return {"message": "Sincronización exitosa", "items_indexed": count}
    except Exception as e:
        print("Sync Error:", e)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/sync/store/{store_id}")
def sync_store(store_id: str):
    """Sincroniza un solo comercio y sus productos (Más rápido)."""
    if not db:
        raise HTTPException(status_code=500, detail="Firebase no está inicializado.")
        
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
        
        # 2. Buscar las 3 anclas más afines al prompt
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
    """Genera un Home dinámico V6 usando solo Vectores con Gravedad Ambiental."""
    if not db: return get_popular_products()
    
    try:
        activities = db.collection('users').document(uid).collection('activity').order_by('timestamp', direction=firestore.Query.DESCENDING).limit(50).stream()
        activities = list(activities)
    except Exception as e:
        print("Error fetching activities:", e)
        activities = []
        
    user_vector = None
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc)
    current_hour = (now.hour - 5) % 24
    
    conn = get_db_connection()
    c = conn.cursor()
    
    if activities:
        try:
            def calc_decay(ts):
                if not ts: return 1.0
                try:
                    if hasattr(ts, 'timestamp'): pass
                    elif isinstance(ts, str):
                        try: ts = datetime.fromisoformat(ts.replace('Z', '+00:00'))
                        except:
                            try: ts = datetime.fromtimestamp(float(ts)/1000, tz=timezone.utc)
                            except: return 1.0
                    elif isinstance(ts, (int, float)): ts = datetime.fromtimestamp(ts/1000, tz=timezone.utc)
                    else: return 1.0
                        
                    days_ago = (now - ts).days
                    if days_ago == 0: return 3.0
                    elif days_ago > 7: return 0.2
                    elif days_ago > 30: return 0.0
                    return 1.0
                except: return 1.0
                
            user_vector = calculate_user_vector(activities, calc_decay, current_hour=current_hour)
        except Exception as e:
            print("Error generando vector de usuario desde local:", e)
            
    global_seen_ids = set()
    feed_sections = []
    
    active_rules = ['general']
    
    # Evaluar hora
    if 5 <= current_hour <= 10: active_rules.append('desayuno')
    elif 11 <= current_hour <= 14: active_rules.append('almuerzo')
    elif 18 <= current_hour <= 23: active_rules.append('cena')
    elif 23 <= current_hour or current_hour <= 4: active_rules.append('madrugada')
    
    # Evaluar clima Open-Meteo
    if req.lat is not None and req.lng is not None:
        try:
            import requests, time
            lat_key = round(req.lat, 1)
            lng_key = round(req.lng, 1)
            loc_key = f"{lat_key}_{lng_key}"
            now_ts = time.time()
            if loc_key in WEATHER_CACHE_STORE and (now_ts - WEATHER_CACHE_STORE[loc_key]["time"] < 3600):
                temp = WEATHER_CACHE_STORE[loc_key]["temp"]
                code = WEATHER_CACHE_STORE[loc_key]["code"]
            else:
                w_res = requests.get(f"https://api.open-meteo.com/v1/forecast?latitude={lat_key}&longitude={lng_key}&current_weather=true", timeout=2).json()
                if "current_weather" in w_res:
                    temp = w_res["current_weather"].get("temperature", 20)
                    code = w_res["current_weather"].get("weathercode", 0)
                    WEATHER_CACHE_STORE[loc_key] = {"temp": temp, "code": code, "time": now_ts}
                else: temp, code = 20, 0
                
            is_night = current_hour < 6 or current_hour >= 18
            if temp >= 24:
                active_rules.append('calor_noche' if is_night else 'calor_dia')
            elif temp <= 16 or code >= 50:
                active_rules.append('frio_noche' if is_night else 'frio_dia')
        except Exception as e:
            print(f"[Weather] Error: {e}")
            
    # Cruce KNN con Gravedad Dinámica
    anchors = []
    if user_vector:
        try:
            c.execute("""
                SELECT a.anchor_id, m.title, m.subtitle, m.allowed_categories, m.exclude_rules, m.titles, m.rule_type,
                       vec_distance_cosine(a.embedding, ?) AS raw_distance, a.embedding
                FROM anchor_vectors a
                JOIN anchor_metadata m ON a.anchor_id = m.anchor_id
            """, (user_vector,))
            
            candidate_anchors = []
            for row in c.fetchall():
                d = dict(row)
                d['distance'] = d['raw_distance']
                if d.get('rule_type') in active_rules and d.get('rule_type') != 'general':
                    d['distance'] -= 0.35 # BOOST MATEMÁTICO AMBIENTAL
                candidate_anchors.append(d)
                
            candidate_anchors.sort(key=lambda x: x['distance'])
            anchors = candidate_anchors[:2]
        except Exception as e:
            print(f"[KNN] Error en Vector Ambiental: {e}")
    else:
        try:
            c.execute("""
                SELECT a.anchor_id, m.title, m.subtitle, m.allowed_categories, m.exclude_rules, m.titles, m.rule_type, a.embedding
                FROM anchor_vectors a
                JOIN anchor_metadata m ON a.anchor_id = m.anchor_id
            """)
            candidate_anchors = []
            for row in c.fetchall():
                d = dict(row)
                d['weight'] = 2.0 if (d.get('rule_type') in active_rules and d.get('rule_type') != 'general') else 1.0
                candidate_anchors.append(d)
                
            import random
            if candidate_anchors:
                anchors = random.choices(candidate_anchors, weights=[x['weight'] for x in candidate_anchors], k=min(2, len(candidate_anchors)))
        except Exception as e:
            pass

    for anchor in anchors:
        try:
            sim_vector = sqlite_vec.serialize_float32(anchor['embedding'])
            c.execute("""
                SELECT p.id, p.type, p.storeId, p.name, p.category, p.description,
                       p.price, p.icon, p.imageUrl, p.onSale, p.salePrice, p.likes, p.views, p.purchases,
                       s.name as storeName, vec_distance_cosine(v.embedding, ?) AS distance
                FROM product_vectors v
                JOIN search_index p ON v.product_id = p.id
                LEFT JOIN (SELECT id, name FROM search_index WHERE type='store') s ON s.id = p.storeId
                WHERE p.type = 'product' AND CAST(p.available AS INTEGER) = 1
                ORDER BY distance ASC
                LIMIT 50
            """, (sim_vector,))
            
            raw_items = c.fetchall()
            candidate_items = []
            import math
            import json
            
            allowed = []
            excluded = []
            try:
                allowed = json.loads(anchor.get('allowed_categories', '[]'))
                excluded = json.loads(anchor.get('exclude_rules', '[]'))
            except: pass
            
            for raw_row in raw_items:
                row = dict(raw_row)
                rid = row["id"]
                if rid in global_seen_ids: continue
                
                cat = row.get("category") or ""
                cat_lower = cat.lower()
                
                if allowed and not any(a.lower() in cat_lower for a in allowed): continue
                if excluded and any(e.lower() in cat_lower for e in excluded): continue
                
                distance = row["distance"]
                affinity = max(0, 1.0 - distance)
                
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
                
                if len(filtered_items) >= 5: break
                    
            if len(filtered_items) >= 2:
                import random
                anchor_title = anchor.get("title", "Explorar")
                if anchor.get("titles"):
                    try:
                        titles_list = json.loads(anchor["titles"])
                        if titles_list: anchor_title = random.choice(titles_list)
                    except: pass
                    
                feed_sections.append({
                    "id": f"dyn_vector_{anchor['anchor_id']}",
                    "type": "products",
                    "title": anchor_title,
                    "subtitle": anchor.get("subtitle", ""),
                    "items": filtered_items
                })
        except Exception as e:
            print(f"[Vector] Error: {e}")

    # Anti-Bubble: Exploración Estricta de Categorías no visitadas
    try:
        user_cats = { (act.to_dict().get('category') or '').lower() for act in activities if hasattr(act, 'to_dict') }
        c.execute("SELECT DISTINCT category FROM search_index WHERE type='product' AND CAST(available AS INTEGER)=1")
        all_cats = [row["category"] for row in c.fetchall() if row["category"]]
        unseen_cats = [cat for cat in all_cats if cat.lower() not in user_cats and cat.lower() != 'general']
        
        if unseen_cats:
            import random
            exp_cat = random.choice(unseen_cats)
            c.execute("""
                SELECT p.id, p.type, p.storeId, p.name, p.category, p.description,
                       p.price, p.icon, p.imageUrl, p.onSale, p.salePrice, p.likes, p.views, p.purchases,
                       s.name as storeName
                FROM search_index p
                LEFT JOIN (SELECT id, name FROM search_index WHERE type='store') s ON s.id = p.storeId
                WHERE p.type = 'product' AND p.category = ? AND CAST(p.available AS INTEGER) = 1
                ORDER BY RANDOM()
                LIMIT 15
            """, (exp_cat,))
            
            exp_items = c.fetchall()
            if len(exp_items) >= 2:
                filtered_exp = []
                store_counts = {}
                for raw_row in exp_items:
                    row = dict(raw_row)
                    rid, sid = row["id"], row["storeId"]
                    if rid in global_seen_ids or store_counts.get(sid, 0) >= 4: continue
                    filtered_exp.append(row)
                    global_seen_ids.add(rid)
                    store_counts[sid] = store_counts.get(sid, 0) + 1
                    if len(filtered_exp) >= 5: break
                
                if len(filtered_exp) >= 2:
                    feed_sections.append({
                        "id": f"dyn_antibubble_{exp_cat.replace(' ', '_')}",
                        "type": "products",
                        "title": f"¿Has probado {exp_cat}?",
                        "subtitle": "Descubre algo totalmente nuevo",
                        "items": filtered_exp
                    })
    except Exception as e:
        print(f"[Anti-Bubble] Error: {e}")

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

@app.get("/api/status")
def get_system_status():
    """Devuelve métricas del estado del sistema para el panel de administración."""
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
            "total_clusters": 0,
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
        # Paginación básica en Firestore
        offset = (page - 1) * limit
        users = users_ref.offset(offset).limit(limit).stream()
        
        # Para saber el total aprox
        total_users = 0 # Firestore count can be slow, but let's assume we return dynamic
        
        results = []
        conn = get_db_connection()
        c = conn.cursor()
        
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc)
        current_hour = (now.hour - 5) % 24
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
                    "name": udata.get('name', udata.get('email', 'Usuario Anónimo')),
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
                "INSERT INTO anchor_metadata (anchor_id, title, subtitle, section_type, allowed_categories, exclude_rules, titles, is_manual, rule_type) VALUES (?, ?, ?, \'products\', ?, ?, ?, 1, \'general\')",
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
    """Devuelve telemetría detallada del Cerebro Vectorial para el panel Admin."""
    try:
        conn = get_db_connection()
        c = conn.cursor()
        
        # 1. Total Vectores Productos
        c.execute("SELECT COUNT(*) as c FROM product_vectors")
        total_product_vectors = c.fetchone()["c"]
        
        c.execute("""
            SELECT a.anchor_id, m.title, m.subtitle, m.section_type, a.embedding, m.is_manual, m.rule_type
            FROM anchor_vectors a
            LEFT JOIN anchor_metadata m ON a.anchor_id = m.anchor_id
        """)
        
        anchors = []
        for row in c.fetchall():
            anchor_dict = dict(row)
            cluster_name = anchor_dict.get("section_type")
            
            if True:
                anchor_dict["keywords"] = "Generado por IA"
                
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
            "fts_clusters": {},
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
                c.execute("SELECT name, description, category FROM search_index WHERE type='product' ORDER BY RANDOM() LIMIT 150")
                products = [dict(row) for row in c.fetchall()]
                conn.close()
                
            import json
            import google.generativeai as genai
            
            prompt = f"""
            Eres un experto en comportamiento del consumidor e inteligencia artificial.
            Aquí tienes las categorías y productos de nuestro marketplace:
            Categorías: {categories}
            Muestra de productos: {products}
            
            Genera un arreglo JSON con las mejores "Anclas" Vectoriales para organizar este inventario.
            Debes generar alrededor de 10-15 anclas en total.
            MUY IMPORTANTE: Incluye anclas ambientales (Clima y Tiempo) asignándoles la propiedad "rule_type" correspondiente.
            Los "rule_type" permitidos son:
            - "general": Ancla estándar (ej: "Licores", "Mascotas", "Salud").
            - "calor_dia": Solo se muestra de día y con calor (ej: "Refrescante", "Helados").
            - "calor_noche": Solo de noche con calor (ej: "Cervezas", "Tragos").
            - "frio_dia": Solo de día con frío/lluvia (ej: "Café", "Panadería").
            - "frio_noche": Solo de noche con frío/lluvia (ej: "Sopas", "Cobijas", "Comida Pesada").
            - "desayuno": (5am a 10am) (ej: "Desayunos", "Huevos").
            - "almuerzo": (11am a 2pm) (ej: "Almuerzos Ejecutivos").
            - "cena": (6pm a 11pm) (ej: "Cena Rápida", "Para compartir").
            
            Estructura exacta:
            [
              {{
                "id": "A1",
                "titles": ["Para este calorcito", "Días soleados", "Refréscate", "Sabor de verano"],
                "subtitle": "Bebidas heladas y jugos",
                "desc": "Helados, jugos, paletas y bebidas frías",
                "rule_type": "calor_dia",
                "allowed_categories": ["Heladería", "Jugos", "Bebidas"],
                "exclude_rules": ["sopa", "caldo", "café caliente"]
              }}
            ]
            
            En "titles", da 4 opciones dinámicas y muy atractivas.
            Devuelve SOLO EL JSON VÁLIDO.
            """
            
            models_to_try = ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-pro", "gemini-1.5-flash"]
            response = None
            for m in models_to_try:
                try:
                    model = genai.GenerativeModel(m)
                    response = model.generate_content(prompt)
                    if response: break
                except: pass
                    
            if not response: return
                
            raw_text = response.text.strip()
            if raw_text.startswith("```json"): raw_text = raw_text[7:]
            if raw_text.startswith("```"): raw_text = raw_text[3:]
            if raw_text.endswith("```"): raw_text = raw_text[:-3]
            
            anchors_data = json.loads(raw_text.strip())
            
            with sqlite_lock:
                conn = get_db_connection()
                c = conn.cursor()
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
                    except: time.sleep(1)
                
                if res and 'embedding' in res:
                    vector_blob = sqlite_vec.serialize_float32(res['embedding'])
                    rule_type = a.get('rule_type', 'general')
                    with sqlite_lock:
                        conn = get_db_connection()
                        c = conn.cursor()
                        c.execute(
                            "INSERT INTO anchor_metadata (anchor_id, title, subtitle, section_type, allowed_categories, exclude_rules, titles, rule_type) VALUES (?, ?, ?, 'products', ?, ?, ?, ?)",
                            (a['id'], primary_title, a.get('subtitle', ''), json.dumps(a.get('allowed_categories', [])), json.dumps(a.get('exclude_rules', [])), json.dumps(a.get('titles', [])), rule_type)
                        )
                        c.execute(
                            "INSERT INTO anchor_vectors (anchor_id, embedding) VALUES (?, ?)",
                            (a['id'], vector_blob)
                        )
                        conn.commit()
                        conn.close()
                        
            print("[Vector Universal] Auto-Generación completada. Ya no hay Fase 2 FTS.")
        except Exception as e:
            print("Error en Auto-Generación V6:", e)
            
    background_tasks.add_task(run_generation)
    return {"status": "ok", "message": "Descubrimiento V6 de Anclas Universales iniciado."}


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
        
        # Limpieza Nube (V6 Exterminio FTS)
        if db:
            try:
                db.collection('config').document('algorithm').delete()
            except Exception as e:
                print("No se pudo limpiar firebase:", e)
                
        return {"status": "ok", "message": "Vectores locales y nube FTS limpiados (V6)."}
    except Exception as e:
        return {"status": "error", "message": str(e)}

# ==========================================
# ==========================================
# WEBHOOKS PUSH PARA ACTUALIZAR ÍNDICE (MINI-ALGOLIA)
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
