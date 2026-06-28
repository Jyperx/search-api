from fastapi import APIRouter, Depends, HTTPException
import sqlite3
import json
import math
import random
import struct
import sqlite_vec

from core.database import get_db_dep, sqlite_lock
import core.firebase
from core.genai_client import embed_text
from data.clusters import MACRO_CLUSTERS_CACHE
from data.synonyms import REVERSE_SYNONYMS, SYNONYMS
from services.recommender import calculate_user_vector
from routers.home import build_cluster_fts_query
from pydantic import BaseModel

class SimulateRequest(BaseModel):
    prompt: str

from routers.home import build_cluster_fts_query

router = APIRouter()

@router.get("/api/search")
def search(q: str = "", category: str = "", history: str = "", conn: sqlite3.Connection = Depends(get_db_dep)):
    """Busca en el índice FTS5 y Vectorial con Soporte para Categorías y Perfil de Usuario."""
    c = conn.cursor()

    if not q.strip() and category:
        # Búsqueda pura por categoría (Filtro de Pestañas FrontEnd)
        c.execute("""
            SELECT p.id, p.type, p.storeId, p.name, p.category, p.description,
                   p.price, p.icon, p.imageUrl, p.onSale, p.salePrice, p.likes, p.views, p.purchases,
                   s.name as storeName
            FROM search_index p
            LEFT JOIN (SELECT id, name FROM search_index WHERE type='store') s ON s.id = p.storeId
            WHERE p.category = ? OR s.category = ?
            ORDER BY CAST(p.likes AS INTEGER) DESC, CAST(p.views AS INTEGER) DESC
            LIMIT 50
        """, (category, category))
        rows = c.fetchall()
        return {"results": [dict(row) for row in rows]}

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
                   s.name as storeName
            FROM search_index p
            LEFT JOIN (SELECT id, name, isOpen FROM search_index WHERE type='store') s ON s.id = p.storeId
            WHERE search_index MATCH ? 
            AND CAST(p.available AS INTEGER) = 1 
            AND CAST(s.isOpen AS INTEGER) = 1
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
                       s.name as storeName
                FROM search_index p
                LEFT JOIN (SELECT id, name, isOpen FROM search_index WHERE type='store') s ON s.id = p.storeId
                WHERE (p.name LIKE ? OR p.category LIKE ? OR p.description LIKE ?)
                AND CAST(p.available AS INTEGER) = 1 
                AND CAST(s.isOpen AS INTEGER) = 1
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
                           s.name as storeName
                    FROM search_index p
                    LEFT JOIN (SELECT id, name, isOpen FROM search_index WHERE type='store') s ON s.id = p.storeId
                    WHERE ({' OR '.join(conds)})
                    AND CAST(p.available AS INTEGER) = 1 AND CAST(s.isOpen AS INTEGER) = 1
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
                    raw_query_vector = embed_text(safe_q, task_type="retrieval_query")
                    
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
                           s.name as storeName, vec_distance_cosine(v.embedding, ?) AS distance
                    FROM product_vectors v
                    JOIN search_index p ON p.id = v.product_id
                    LEFT JOIN (SELECT id, name FROM search_index WHERE type='store') s ON s.id = p.storeId
                    ORDER BY distance ASC
                    LIMIT 20
                """, (query_vector,))
                raw_products = [dict(row) for row in c.fetchall()]
                
                c.execute("""
                    SELECT p.id, p.type, p.storeId, p.name, p.category, p.description,
                           p.price, p.icon, p.imageUrl, p.onSale, p.salePrice, p.likes, p.views, p.purchases,
                           s.name as storeName, vec_distance_cosine(v.embedding, ?) AS distance
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

@router.post("/api/simulate")
def simulate_home_feed(req: SimulateRequest, conn: sqlite3.Connection = Depends(get_db_dep)):
    """Simulador para probar el Cerebro Vectorial en el panel de Admin"""
    try:
        emb = embed_text(req.prompt, task_type="retrieval_query")
        if not emb:
            return {"status": "error", "error": "No se pudo generar el embedding."}

        sim_vector = sqlite_vec.serialize_float32(emb)
        c = conn.cursor()
        
        c.execute("""
            SELECT a.anchor_id, m.title, m.subtitle, m.allowed_categories, m.exclude_rules, m.titles, vec_distance_cosine(a.embedding, ?) AS distance
            FROM anchor_vectors a
            JOIN anchor_metadata m ON a.anchor_id = m.anchor_id
            ORDER BY distance ASC
            LIMIT 3
        """, (sim_vector,))
        anchors = [dict(row) for row in c.fetchall()]
        
        feed_sections = []
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
                
        return {"results": feed_sections}
    except Exception as e:
        import traceback
        traceback.print_exc()
        return []

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
