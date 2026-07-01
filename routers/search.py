from fastapi import APIRouter, Depends, HTTPException
import sqlite3
import json
import math
import random
import struct
import sqlite_vec
from functools import lru_cache

from typing import Optional
from core.database import get_db_dep, get_db_connection, sqlite_lock
import core.firebase
from core.genai_client import embed_text


@lru_cache(maxsize=512)
def _embed_query_cached(q: str) -> tuple:
    """Embebe una query de búsqueda con caché LRU: la misma palabra no se re-embebe en Gemini.
    Devuelve una tupla (inmutable) para que sea cacheable de forma segura."""
    return tuple(embed_text(q, task_type="retrieval_query"))
from data.clusters import MACRO_CLUSTERS_CACHE
from data.synonyms import REVERSE_SYNONYMS, SYNONYMS
from services.recommender import calculate_user_vector
from routers.home import build_cluster_fts_query
from pydantic import BaseModel

class SimulateRequest(BaseModel):
    prompt: str


class SearchClickLog(BaseModel):
    query: str
    clicked_id: str
    clicked_category: Optional[str] = ""
    result_count: Optional[int] = 0


from routers.home import build_cluster_fts_query

router = APIRouter()


@router.post("/api/log/search-click")
def log_search_click(body: SearchClickLog):
    """Registra un clic de búsqueda (query real → producto). Alimenta el aprendizaje de sinónimos
    por co-clics y el CTR de búsqueda. Antes la app llamaba a este endpoint pero NO existía (404)."""
    q = (body.query or "").strip().lower()
    if not q or not body.clicked_id:
        return {"status": "ok"}
    try:
        with sqlite_lock:
            conn = get_db_connection()
            try:
                conn.execute(
                    "INSERT INTO search_logs (query, clicked_id, clicked_category, result_count) VALUES (?, ?, ?, ?)",
                    (q, body.clicked_id, body.clicked_category or "", body.result_count or 0)
                )
                conn.commit()
            finally:
                conn.close()
        return {"status": "ok"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@router.get("/api/search")
def search(q: str = "", category: str = "", history: str = "", conn: sqlite3.Connection = Depends(get_db_dep)):
    """Busca en el índice FTS5 y Vectorial con Soporte para Categorías y Perfil de Usuario."""
    c = conn.cursor()

    if not q.strip() and category:
        # Búsqueda pura por categoría (Filtro de pestañas del buscador).
        # Match por la categoría del COMERCIO (app) o la del producto. (FIX: s.category
        # debe venir en el subquery del JOIN; antes daba error SQL.)
        c.execute("""
            SELECT p.id, p.type, p.storeId, p.name, p.category, p.description,
                   p.price, p.icon, p.imageUrl, p.onSale, p.salePrice, p.likes, p.views, p.purchases,
                   s.name as storeName, s.isOpen as storeIsOpen, s.category as storeCategory
            FROM search_index p
            LEFT JOIN (SELECT id, name, isOpen, category FROM search_index WHERE type='store') s ON s.id = p.storeId
            WHERE p.type = 'product' AND (s.category = ? OR p.category = ?)
            AND CAST(p.available AS INTEGER) = 1
            ORDER BY CAST(p.likes AS INTEGER) DESC, CAST(p.views AS INTEGER) DESC
            LIMIT 50
        """, (category, category))
        prod_rows = [dict(row) for row in c.fetchall()]

        # También las tiendas de esa categoría (para la sección de comercios)
        c.execute("""
            SELECT id, type, storeId, name, category, description,
                   price, icon, imageUrl, onSale, salePrice, likes, views, purchases,
                   name as storeName, isOpen as storeIsOpen
            FROM search_index
            WHERE type = 'store' AND category = ?
            ORDER BY CAST(likes AS INTEGER) DESC
            LIMIT 30
        """, (category,))
        store_rows = [dict(row) for row in c.fetchall()]

        return {"results": store_rows + prod_rows, "exact_match": True}

    if not q.strip():
        return {"results": []}
    
    safe_q = q.replace('"', '').replace("'", "").lower().strip()
    
    # === CEREBRO V2 INJECTION ===
    # Solo expandimos a cluster si el usuario escribe el NOMBRE del cluster (ej. "desayuno").
    # Una palabra de producto ("hamburguesa") NO debe secuestrar la búsqueda hacia todo un cluster.
    cluster_match = None
    cluster_name = None
    if safe_q in MACRO_CLUSTERS_CACHE:
        cluster_match = True
        cluster_name = safe_q
            
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
        c.execute("""
            SELECT p.id, p.type, p.storeId, p.name, p.category, p.description,
                   p.price, p.icon, p.imageUrl, p.onSale, p.salePrice, p.likes, p.views, p.purchases,
                   s.name as storeName, s.isOpen as storeIsOpen, s.category as storeCategory
            FROM search_index p
            LEFT JOIN (SELECT id, name, isOpen, category FROM search_index WHERE type='store') s ON s.id = p.storeId
            WHERE search_index MATCH ?
            AND CAST(p.available AS INTEGER) = 1
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
        
        exact_match = True
        
        # FALLBACK 1: LIKE Substring match (ideal para fragmentos como "ur" en "burguer")
        if len(results) == 0 and len(safe_q) >= 2:
            like_q = f"%{safe_q}%"
            c.execute("""
                SELECT p.id, p.type, p.storeId, p.name, p.category, p.description,
                       p.price, p.icon, p.imageUrl, p.onSale, p.salePrice, p.likes, p.views, p.purchases,
                       s.name as storeName, s.isOpen as storeIsOpen, s.category as storeCategory
                FROM search_index p
                LEFT JOIN (SELECT id, name, isOpen, category FROM search_index WHERE type='store') s ON s.id = p.storeId
                WHERE (p.name LIKE ? OR p.category LIKE ? OR p.description LIKE ?)
                AND CAST(p.available AS INTEGER) = 1
                ORDER BY
                    (COALESCE(CAST(p.likes AS REAL), 0) * 10.0) +
                    (COALESCE(CAST(p.purchases AS REAL), 0) * 15.0) +
                    (COALESCE(CAST(p.views AS REAL), 0) * 0.5) +
                    (CASE WHEN COALESCE(CAST(p.views AS INTEGER), 0) < 50 THEN ABS(RANDOM() % 300) ELSE 0 END) +
                    ABS(RANDOM() % 20) DESC
                LIMIT 50
            """, (like_q, like_q, like_q))
            
            rows_like = c.fetchall()
            results = [dict(row) for row in rows_like]
            if results:
                exact_match = False

        # FALLBACK 1.5: FUZZY (corrección de typos, ej. "hamburgesa" -> "hamburguesa")
        if len(results) == 0 and len(safe_q) >= 3:
            import difflib
            vocab = set()
            for r in c.execute("SELECT name, category FROM search_index WHERE type='product'").fetchall():
                blob = (str(r["name"] or "") + " " + str(r["category"] or "")).lower()
                for w in blob.replace(",", " ").split():
                    w = w.strip(".,()-")
                    if len(w) >= 4:
                        vocab.add(w)
            vocab_list = list(vocab)
            corrected = []
            changed = False
            for word in safe_q.split():
                if word in vocab or len(word) < 4:
                    corrected.append(word)
                else:
                    m = difflib.get_close_matches(word, vocab_list, n=1, cutoff=0.78)
                    if m:
                        corrected.append(m[0]); changed = True
                    else:
                        corrected.append(word)
            if changed:
                conds, params = [], []
                for w in corrected:
                    conds.append("(p.name LIKE ? OR p.category LIKE ?)")
                    params.extend([f"%{w}%", f"%{w}%"])
                c.execute(f"""
                    SELECT p.id, p.type, p.storeId, p.name, p.category, p.description,
                           p.price, p.icon, p.imageUrl, p.onSale, p.salePrice, p.likes, p.views, p.purchases,
                           s.name as storeName, s.isOpen as storeIsOpen, s.category as storeCategory
                    FROM search_index p
                    LEFT JOIN (SELECT id, name, isOpen, category FROM search_index WHERE type='store') s ON s.id = p.storeId
                    WHERE ({' OR '.join(conds)})
                    AND CAST(p.available AS INTEGER) = 1
                    ORDER BY (COALESCE(CAST(p.likes AS REAL), 0) * 10.0) + (COALESCE(CAST(p.purchases AS REAL), 0) * 15.0) DESC
                    LIMIT 50
                """, params)
                results = [dict(row) for row in c.fetchall()]
                if results:
                    exact_match = False

        # VECTOR SEARCH ENHANCEMENT WITH USER PROFILE (OPTION 3)
        if len(results) < 5 and len(safe_q) >= 3 and not cluster_match:
            query_vector = None
            for attempt in range(2):
                try:
                    raw_query_vector = list(_embed_query_cached(safe_q))

                    # Interpolar con el Perfil del Usuario
                    if history:
                        try:
                            activities = json.loads(history)
                            from datetime import datetime, timezone
                            now = datetime.now(timezone.utc)
                            def calc_decay(ts):
                                if not ts: return 1.0
                                try:
                                    if isinstance(ts, str):
                                        ts = datetime.fromisoformat(ts.replace('Z', '+00:00'))
                                    elif isinstance(ts, (int, float)):
                                        ts = datetime.fromtimestamp(ts/1000, tz=timezone.utc)
                                    else: return 1.0
                                    days_ago = (now - ts).days
                                    if days_ago == 0: return 3.0
                                    elif days_ago > 7: return 0.2
                                    return 1.0
                                except: return 1.0
                            
                            user_vector_bytes = calculate_user_vector(activities, current_hour=datetime.now().hour)
                            if user_vector_bytes:
                                u_v_floats = struct.unpack(f"{len(raw_query_vector)}f", user_vector_bytes)
                                query_vector = sqlite_vec.serialize_float32(
                                    [q_v * 0.85 + u_v * 0.15 for q_v, u_v in zip(raw_query_vector, u_v_floats)]
                                )
                            else:
                                query_vector = sqlite_vec.serialize_float32(raw_query_vector)
                        except Exception as e:
                            print("Error procesando history de usuario:", e)
                            query_vector = sqlite_vec.serialize_float32(raw_query_vector)
                    else:
                        query_vector = sqlite_vec.serialize_float32(raw_query_vector)
                    break
                except Exception as e:
                    print("Error vectorizando search query:", e)
                    
            if query_vector:
                c.execute("""
                    SELECT p.id, p.type, p.storeId, p.name, p.category, p.description,
                           p.price, p.icon, p.imageUrl, p.onSale, p.salePrice, p.likes, p.views, p.purchases,
                           s.name as storeName, s.isOpen as storeIsOpen, s.category as storeCategory, vec_distance_cosine(v.embedding, ?) AS distance
                    FROM product_vectors v
                    JOIN search_index p ON p.id = v.product_id
                    LEFT JOIN (SELECT id, name, isOpen, category FROM search_index WHERE type='store') s ON s.id = p.storeId
                    WHERE CAST(p.available AS INTEGER) = 1
                    ORDER BY distance ASC
                    LIMIT 20
                """, (query_vector,))
                raw_products = [dict(row) for row in c.fetchall()]

                c.execute("""
                    SELECT p.id, p.type, p.storeId, p.name, p.category, p.description,
                           p.price, p.icon, p.imageUrl, p.onSale, p.salePrice, p.likes, p.views, p.purchases,
                           s.name as storeName, p.isOpen as storeIsOpen, vec_distance_cosine(v.embedding, ?) AS distance
                    FROM store_vectors v
                    JOIN search_index p ON p.id = v.store_id
                    LEFT JOIN (SELECT id, name FROM search_index WHERE type='store') s ON s.id = p.storeId
                    ORDER BY distance ASC
                    LIMIT 10
                """, (query_vector,))
                raw_stores = [dict(row) for row in c.fetchall()]

                best_dist = 1.0
                if raw_products and raw_stores:
                    best_dist = min(raw_products[0]['distance'], raw_stores[0]['distance'])
                elif raw_products:
                    best_dist = raw_products[0]['distance']
                elif raw_stores:
                    best_dist = raw_stores[0]['distance']

                if len(results) == 0:
                    exact_match = False

                dynamic_thresh = min(best_dist + 0.15, 0.65)
                
                vec_products = [r for r in raw_products if r['distance'] <= dynamic_thresh]
                vec_stores = [r for r in raw_stores if r['distance'] <= dynamic_thresh]
                
                if not exact_match:
                    vec_products = vec_products[:4]
                    vec_stores = vec_stores[:2]
                
                vec_all = vec_stores + vec_products
                vec_all.sort(key=lambda x: x['distance'])
                
                seen_ids = set([r["id"] for r in results])
                for v_item in vec_all:
                    if v_item["id"] not in seen_ids:
                        if 'distance' in v_item:
                            del v_item['distance']
                        results.append(v_item)
                        seen_ids.add(v_item["id"])
                        
        final_stores = [r for r in results if r.get('type') == 'store']
        final_products = [r for r in results if r.get('type') != 'store']

        # Para consultas de intención (no exactas), evitar que UNA categoría acapare,
        # PERO sin romper la relevancia: se conserva el orden (lo más relevante arriba) y
        # solo se baja el exceso de una misma categoría. Lo irrelevante NO se promueve.
        if not exact_match and len(final_products) > 6:
            CAP = 4
            cat_counts = {}
            kept, overflow = [], []
            for p in final_products:  # ya vienen en orden de relevancia
                cat = (p.get('category') or 'otros').lower()
                if cat_counts.get(cat, 0) < CAP:
                    kept.append(p)
                    cat_counts[cat] = cat_counts.get(cat, 0) + 1
                else:
                    overflow.append(p)
            final_products = kept + overflow

        top_stores = final_stores[:4]
        results = top_stores + final_stores[4:] + final_products
                    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": str(e), "traceback": traceback.format_exc(), "results": [], "exact_match": False}
    
    if category and q.strip():
        results = [r for r in results if r.get('category') == category or r.get('storeName') == category or r.get('storeCategory') == category]
    
    return {"results": results, "exact_match": exact_match}

@router.get("/api/popular")
def get_popular_products(conn: sqlite3.Connection = Depends(get_db_dep)):
    """Devuelve productos recomendados o populares."""
    c = conn.cursor()
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
    return {"results": [dict(row) for row in rows]}

@router.get("/api/promotions")
def get_promotions(conn: sqlite3.Connection = Depends(get_db_dep)):
    """Devuelve las promociones indexadas."""
    c = conn.cursor()
    c.execute("SELECT * FROM promotions")
    rows = c.fetchall()
    return {"results": [dict(row) for row in rows]}


@router.get("/api/recommendations/{uid}")
def get_user_recommendations(uid: str, conn: sqlite3.Connection = Depends(get_db_dep)):
    """Obtiene recomendaciones on-demand con vectores, sección nuevos y comercios populares."""
    try:
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc)
        current_hour = (now.hour - 5) % 24
        
        def calc_decay(ts):
            if not ts: return 1.0
            try:
                if isinstance(ts, str):
                    ts = datetime.fromisoformat(ts.replace('Z', '+00:00'))
                elif isinstance(ts, (int, float)):
                    ts = datetime.fromtimestamp(ts/1000 if ts > 10000000000 else ts, tz=timezone.utc)
                else: return 1.0
                days_ago = (now - ts).days
                if days_ago == 0: return 3.0
                elif days_ago > 7: return 0.2
                elif days_ago > 30: return 0.0
                return 1.0
            except: return 1.0
            
        activities = []
        if core.firebase.db:
            user_doc = core.firebase.db.collection('users').document(uid).get()
            if user_doc.exists:
                user_data = user_doc.to_dict() or {}
                activities = user_data.get('recent_activity', [])
            
        user_vector = calculate_user_vector(activities, current_hour=current_hour) if activities else None

        c = conn.cursor()
        recommended = []
        if user_vector:
            c.execute("""
                SELECT p.id, p.type, p.storeId, p.name, p.category, p.description,
                       p.price, p.icon, p.imageUrl, p.onSale, p.salePrice, p.likes, p.views, p.purchases,
                       s.name as storeName, vec_distance_cosine(v.embedding, ?) AS distance
                FROM product_vectors v
                JOIN search_index p ON p.id = v.product_id AND p.type = 'product'
                LEFT JOIN (SELECT id, name FROM search_index WHERE type='store') s ON s.id = p.storeId
                ORDER BY distance ASC
                LIMIT 4
            """, (user_vector,))
            recommended = [dict(row) for row in c.fetchall()]
            
        if len(recommended) < 4:
            limit_needed = 4 - len(recommended)
            existing_ids = [r['id'] for r in recommended]
            if existing_ids:
                placeholders = ','.join('?' for _ in existing_ids)
                query = f"""
                    SELECT p.id, p.type, p.storeId, p.name, p.category, p.description,
                           p.price, p.icon, p.imageUrl, p.onSale, p.salePrice, p.likes, p.views, p.purchases,
                           s.name as storeName
                    FROM search_index p
                    LEFT JOIN (SELECT id, name FROM search_index WHERE type='store') s ON s.id = p.storeId
                    WHERE p.type = 'product' AND p.id NOT IN ({placeholders})
                    ORDER BY CAST(p.likes AS INTEGER) DESC
                    LIMIT ?
                """
                c.execute(query, existing_ids + [limit_needed])
            else:
                c.execute("""
                    SELECT p.id, p.type, p.storeId, p.name, p.category, p.description,
                           p.price, p.icon, p.imageUrl, p.onSale, p.salePrice, p.likes, p.views, p.purchases,
                           s.name as storeName
                    FROM search_index p
                    LEFT JOIN (SELECT id, name FROM search_index WHERE type='store') s ON s.id = p.storeId
                    WHERE p.type = 'product'
                    ORDER BY CAST(p.likes AS INTEGER) DESC
                    LIMIT ?
                """, (limit_needed,))
            recommended.extend([dict(row) for row in c.fetchall()])
            
        c.execute("""
            SELECT p.id, p.type, p.storeId, p.name, p.category, p.description,
                   p.price, p.icon, p.imageUrl, p.onSale, p.salePrice, p.likes, p.views, p.purchases,
                   s.name as storeName
            FROM search_index p
            LEFT JOIN (SELECT id, name FROM search_index WHERE type='store') s ON s.id = p.storeId
            WHERE p.type = 'product'
            ORDER BY RANDOM()
            LIMIT 4
        """)
        explore = [dict(row) for row in c.fetchall()]
        
        c.execute("""
            SELECT p.id, p.type, p.storeId, p.name, p.category, p.description,
                   p.price, p.icon, p.imageUrl, p.onSale, p.salePrice, p.likes, p.views, p.purchases,
                   s.name as storeName
            FROM search_index p
            LEFT JOIN (SELECT id, name FROM search_index WHERE type='store') s ON s.id = p.storeId
            WHERE p.type = 'product'
            ORDER BY CAST(p.purchases AS INTEGER) DESC, CAST(p.likes AS INTEGER) DESC
            LIMIT 4
        """)
        best_sellers = [dict(row) for row in c.fetchall()]

        c.execute("""
            SELECT id as store_id, name, category, description, imageUrl, imageUrl as logoUrl, likes, isOpen as open
            FROM search_index
            WHERE type = 'store' AND CAST(isOpen AS INTEGER) = 1
            ORDER BY CAST(likes AS INTEGER) DESC
            LIMIT 3
        """)
        stores_rows = c.fetchall()
        popular_stores = []
        for raw_row in stores_rows:
            row = dict(raw_row)
            likes_val = int(row["likes"] or 0)
            row["rating"] = round(min(5.0, 4.0 + (likes_val / 100)), 1)
            row["time"] = "15-25 min"
            row["deliveryFee"] = 0
            row["type"] = "store"
            row["id"] = row["store_id"]
            popular_stores.append(row)

        return {
            "recommended": recommended,
            "explore": explore,
            "best_sellers": best_sellers,
            "stores": popular_stores
        }
    except Exception as e:
        print(f"Error generando recomendaciones API estructuradas para {uid}:", e)
        return {"recommended": [], "explore": [], "best_sellers": [], "stores": []}
