from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
import sqlite3
import json
import numpy as np
import google.generativeai as genai
import sqlite_vec

from database import get_db_connection, get_db_connection_raw, init_db, db
from schemas import ManualAnchorRequest, SearchClickPayload, ProductPayload, StorePayload
from core.config import (global_sync_state, EMBEDDING_MODEL, MACRO_CLUSTERS_CACHE,
                         TIME_RULES_CACHE, SQLITE_DB, SYNONYMS, load_synonyms_from_firestore)
from services.embeddings import async_index_product_vector, async_index_store_vector
from services.recommender import calculate_user_vector

# Importar tareas de fondo (se definirán en background_jobs.py)
from background_jobs import do_seed_anchors, do_sync_database, run_auto_learn_synonyms_sync

router = APIRouter()

@router.get("/api/admin/sync-status")
async def get_sync_status():
    return global_sync_state

@router.post("/api/sync")
def sync_database(background_tasks: BackgroundTasks):
    """Descarga todos los comercios y productos de Firestore y reconstruye el índice SQLite."""
    global global_sync_state
    global_sync_state["is_syncing"] = True
    global_sync_state["total_products"] = 1
    global_sync_state["completed_products"] = 0
    global_sync_state["status"] = "Iniciando sincronización..."
    
    background_tasks.add_task(do_sync_database)
    return {"status": "processing", "message": "Sincronización en curso"}

@router.post("/api/sync/store/{store_id}")
def sync_store(store_id: str, conn: sqlite3.Connection = Depends(get_db_connection)):
    """Sincroniza un solo comercio y sus productos (Más rápido)."""
    if not db:
        raise HTTPException(status_code=500, detail="Firebase no está inicializado.")
        
    c = conn.cursor()
    c.execute("DELETE FROM search_index WHERE storeId = ?", (store_id,))
    
    count = 0
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
        
        products_ref = store_ref.collection("products")
        store_product_names = []
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
            if p_data.get('available', True):
                store_product_names.append(p_data.get('name', ''))
                # Somete a vectorización en background
                from services.embeddings import vector_worker_pool
                vector_worker_pool.submit(
                    async_index_product_vector, 
                    product.id, 
                    p_data.get('name', ''), 
                    p_data.get('category', ''), 
                    p_data.get('description', '')
                )

    conn.commit()
    products_summary = ", ".join(store_product_names[:10])
    from services.embeddings import vector_worker_pool
    vector_worker_pool.submit(async_index_store_vector, store_id, s_data.get('name', ''), s_data.get('category', ''), s_data.get('description', ''), products_summary)
    
    return {"message": f"Comercio {store_id} sincronizado", "items_indexed": count}

@router.post("/api/seed-anchors")
def seed_anchors_endpoint(background_tasks: BackgroundTasks):
    """Siembra los vectores ancla base en SQLite en bg para evitar Timeout."""
    background_tasks.add_task(do_seed_anchors)
    return {"status": "processing", "message": "Vectores ancla sembrándose en segundo plano."}

@router.get("/api/admin/users-vectors")
def get_admin_users_vectors(page: int = 1, limit: int = 10, conn: sqlite3.Connection = Depends(get_db_connection)):
    """Devuelve los perfiles vectoriales de los usuarios activos calculando su afinidad actual."""
    try:
        users_ref = db.collection('users')
        offset = (page - 1) * limit
        users = users_ref.offset(offset).limit(limit).stream()
        results = []
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
            
            user_vector = calculate_user_vector(conn, recent_activity, calc_decay, current_hour=current_hour)
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
        
        return {"status": "ok", "users": results}
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"status": "error", "message": str(e)}

@router.post("/api/admin/cerebro/anchors")
def create_manual_anchor(req: ManualAnchorRequest, conn: sqlite3.Connection = Depends(get_db_connection)):
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
            
        vector_blob = sqlite_vec.serialize_float32(res['embedding'][:768])
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
        return {"status": "ok", "anchor_id": anchor_id}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@router.put("/api/admin/cerebro/anchors/{anchor_id}")
def update_manual_anchor(anchor_id: str, req: ManualAnchorRequest, conn: sqlite3.Connection = Depends(get_db_connection)):
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
            return {"status": "error", "message": "Failed to update embedding"}
            
        vector_blob = sqlite_vec.serialize_float32(res['embedding'][:768])
        c = conn.cursor()
        c.execute(
            "UPDATE anchor_metadata SET title=?, subtitle=?, allowed_categories=?, exclude_rules=?, titles=? WHERE anchor_id=? AND is_manual=1",
            (primary_title, req.subtitle, json.dumps(req.allowed_categories), json.dumps(req.exclude_rules), json.dumps(req.titles), anchor_id)
        )
        c.execute(
            "UPDATE anchor_vectors SET embedding=? WHERE anchor_id=?",
            (vector_blob, anchor_id)
        )
        conn.commit()
        return {"status": "ok", "anchor_id": anchor_id}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@router.delete("/api/admin/cerebro/anchors/{anchor_id}")
def delete_manual_anchor(anchor_id: str, conn: sqlite3.Connection = Depends(get_db_connection)):
    try:
        c = conn.cursor()
        c.execute("DELETE FROM anchor_metadata WHERE anchor_id=? AND is_manual=1", (anchor_id,))
        c.execute("DELETE FROM anchor_vectors WHERE anchor_id=?", (anchor_id,))
        conn.commit()
        return {"status": "ok", "anchor_id": anchor_id}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@router.get("/api/admin/cerebro")
def get_admin_cerebro(page: int = 1, store_page: int = 1, anchor_page: int = 1, limit: int = 10, conn: sqlite3.Connection = Depends(get_db_connection)):
    """Devuelve telemetría detallada del Cerebro Vectorial para el panel Admin."""
    try:
        c = conn.cursor()
        
        c.execute("SELECT COUNT(*) as c FROM product_vectors")
        total_product_vectors = c.fetchone()["c"]
        
        c.execute("SELECT COUNT(*) as c FROM store_vectors")
        total_store_vectors = c.fetchone()["c"]
        
        c.execute("SELECT COUNT(*) as c FROM anchor_vectors")
        total_anchors = c.fetchone()["c"]
 
        anchor_offset = (anchor_page - 1) * limit
        c.execute("""
            SELECT a.anchor_id, m.title, m.subtitle, m.section_type, a.embedding, m.is_manual
            FROM anchor_vectors a
            LEFT JOIN anchor_metadata m ON a.anchor_id = m.anchor_id
            LIMIT ? OFFSET ?
        """, (limit, anchor_offset))
        
        anchors = []
        for row in c.fetchall():
            anchor_dict = dict(row)
            cluster_name = anchor_dict.get("section_type")
            
            if cluster_name and cluster_name in MACRO_CLUSTERS_CACHE:
                anchor_dict["keywords"] = MACRO_CLUSTERS_CACHE[cluster_name].get("keywords", "Generado por IA")
            else:
                anchor_dict["keywords"] = "Generado por IA"
                
            if anchor_dict.get("embedding"):
                vec_array = np.frombuffer(anchor_dict["embedding"], dtype=np.float32)
                anchor_dict["vector_preview"] = f"[{vec_array[0]:.3f}, {vec_array[1]:.3f}, {vec_array[2]:.3f}...]"
                del anchor_dict["embedding"]
                
            anchors.append(anchor_dict)
        
        offset = (page - 1) * limit
        c.execute("""
            SELECT p.product_id, s.name, s.category, length(p.embedding) as vec_bytes
            FROM product_vectors p
            JOIN search_index s ON p.product_id = s.id AND s.type = 'product'
            LIMIT ? OFFSET ?
        """, (limit, offset))
        sample_products = [dict(row) for row in c.fetchall()]
        
        store_offset = (store_page - 1) * limit
        c.execute("""
            SELECT p.store_id, s.name, s.category, length(p.embedding) as vec_bytes
            FROM store_vectors p
            JOIN search_index s ON p.store_id = s.id AND s.type = 'store'
            LIMIT ? OFFSET ?
        """, (limit, store_offset))
        sample_stores = [dict(row) for row in c.fetchall()]
        
        return {
            "status": "ok",
            "fts_clusters": MACRO_CLUSTERS_CACHE,
            "synonyms": SYNONYMS,
            "vector_metrics": {
                "total_product_vectors": total_product_vectors,
                "total_store_vectors": total_store_vectors,
                "anchors_count": total_anchors,
                "anchors": anchors,
                "sample_products": sample_products,
                "sample_stores": sample_stores,
                "pagination": {
                    "page": page,
                    "limit": limit,
                    "total": total_product_vectors
                },
                "store_pagination": {
                    "page": store_page,
                    "limit": limit,
                    "total": total_store_vectors
                },
                "anchor_pagination": {
                    "page": anchor_page,
                    "limit": limit,
                    "total": total_anchors
                }
            }
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/api/admin/auto-generate-anchors")
def auto_generate_anchors(background_tasks: BackgroundTasks):
    # run_generation se definirá en background_jobs.py o localmente ya que es asíncrona de fondo
    from background_jobs import run_generation_task
    background_tasks.add_task(run_generation_task)
    return {"status": "ok", "message": "Descubrimiento de anclas con IA iniciado en background. Espera un minuto."}

@router.post("/api/admin/reset-vectors")
def reset_vectors_db(conn: sqlite3.Connection = Depends(get_db_connection)):
    try:
        c = conn.cursor()
        c.execute("DROP TABLE IF EXISTS product_vectors")
        c.execute("DROP TABLE IF EXISTS store_vectors")
        c.execute("DROP TABLE IF EXISTS anchor_vectors")
        c.execute("DELETE FROM anchor_metadata")
        conn.commit()
        
        init_db()
        return {"status": "ok", "message": "Vectores limpiados correctamente."}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@router.post("/api/admin/reset-users-activity")
def reset_users_activity():
    if not db:
        raise HTTPException(status_code=500, detail="Firebase no está inicializado.")
    try:
        users_ref = db.collection('users')
        users = users_ref.stream()
        batch = db.batch()
        count = 0
        for doc in users:
            batch.update(doc.reference, {"recent_activity": []})
            count += 1
            if count >= 400:
                batch.commit()
                batch = db.batch()
                count = 0
        if count > 0:
            batch.commit()
        return {"status": "ok", "message": "Historial de actividad limpiado para todos los usuarios."}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@router.post("/api/log/search-click")
def log_search_click(payload: SearchClickPayload, conn: sqlite3.Connection = Depends(get_db_connection)):
    """Guarda silenciosamente qué clickeó el usuario para una búsqueda específica."""
    try:
        c = conn.cursor()
        c.execute('''
            INSERT INTO search_logs (query, clicked_id, clicked_category, result_count)
            VALUES (?, ?, ?, ?)
        ''', (payload.query.lower().strip(), payload.clicked_id, payload.clicked_category, payload.result_count))
        conn.commit()
        return {"status": "ok"}
    except Exception as e:
        print(f"Error logging search click: {e}")
        return {"status": "error"}

@router.post("/api/admin/auto-learn-synonyms")
def trigger_auto_learn(background_tasks: BackgroundTasks):
    background_tasks.add_task(run_auto_learn_synonyms_sync)
    return {"status": "ok", "message": "Proceso de auto-aprendizaje de sinónimos iniciado en background."}

@router.post("/api/reset-clusters")
def reset_clusters_to_defaults():
    if not db:
        raise HTTPException(status_code=500, detail="Firebase no está inicializado.")
    try:
        doc_ref = db.collection('config').document('algorithm')
        doc_ref.set({"clusters": MACRO_CLUSTERS_CACHE}, merge=True)
        return {
            "message": f"✅ {len(MACRO_CLUSTERS_CACHE)} clústeres reseteados a los defaults V3.2 correctamente.",
            "clusters_pushed": list(MACRO_CLUSTERS_CACHE.keys())
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/api/index/product")
def index_product(payload: ProductPayload, conn: sqlite3.Connection = Depends(get_db_connection)):
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
    from services.embeddings import vector_worker_pool
    vector_worker_pool.submit(async_index_product_vector, payload.id, payload.name, payload.category, payload.description)
    return {"status": "indexed", "id": payload.id}

@router.delete("/api/index/product/{product_id}")
def delete_product_index(product_id: str, conn: sqlite3.Connection = Depends(get_db_connection)):
    c = conn.cursor()
    c.execute("DELETE FROM search_index WHERE id = ? AND type = 'product'", (product_id,))
    conn.commit()
    return {"status": "deleted", "id": product_id}

@router.post("/api/index/store")
def index_store(payload: StorePayload, conn: sqlite3.Connection = Depends(get_db_connection)):
    c = conn.cursor()
    c.execute("DELETE FROM search_index WHERE id = ? AND type = 'store'", (payload.id,))
    c.execute("""
        INSERT INTO search_index (id, type, storeId, name, category, description, price, icon, imageUrl, onSale, salePrice, likes, views, purchases, available, isOpen)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0, NULL, 0, 0, 0, 1, ?)
    """, (
        payload.id, 'store', payload.id, 
        payload.name, payload.category, '', '', '', payload.imageUrl,
        1 if payload.isOpen else 0
    ))
    conn.commit()
    from services.embeddings import vector_worker_pool
    vector_worker_pool.submit(async_index_store_vector, payload.id, payload.name, payload.category, '', '')
    return {"status": "indexed", "id": payload.id}

@router.patch("/api/index/store/{store_id}/status")
def update_store_status(store_id: str, isOpen: bool, conn: sqlite3.Connection = Depends(get_db_connection)):
    c = conn.cursor()
    c.execute(
        "UPDATE search_index SET isOpen = ? WHERE id = ? AND type = 'store'",
        (1 if isOpen else 0, store_id)
    )
    conn.commit()
    return {"status": "ok", "store_id": store_id, "isOpen": isOpen}

@router.get("/api/status")
def get_system_status(conn: sqlite3.Connection = Depends(get_db_connection)):
    """Devuelve métricas del estado del sistema para el panel de administración."""
    try:
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
