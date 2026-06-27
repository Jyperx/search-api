import json
import logging
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List, Optional
import google.generativeai as genai
import sqlite_vec

from core.database import get_db_connection, get_db_dep, sqlite_lock, init_db
from core.firebase import db
from core.config import EMBEDDING_MODEL
from data.clusters import MACRO_CLUSTERS_CACHE
from services.recommender import calculate_user_vector

logger = logging.getLogger(__name__)

router = APIRouter()

class ManualAnchorRequest(BaseModel):
    title: str
    desc: str
    subtitle: str
    allowed_categories: List[str] = []
    exclude_rules: List[str] = []
    titles: List[str] = []


@router.get("/api/admin/users-vectors")
def get_admin_users_vectors(page: int = 1, limit: int = 10):
    """Devuelve los perfiles vectoriales de los usuarios activos calculando su afinidad actual."""
    try:
        users_ref = db.collection('users')
        # PaginaciÔö£Ôöén bÔö£├¡sica en Firestore
        offset = (page - 1) * limit
        users = users_ref.offset(offset).limit(limit).stream()
        
        # Para saber el total aprox
        total_users = 0 # Firestore count can be slow, but let's assume we return dynamic
        
        results = []
        conn = get_db_connection()
        try:
            c = conn.cursor()
            
            from datetime import datetime, timezone
            
            current_hour = (datetime.now(timezone.utc).hour - 5) % 24
            
            for u in users:
                uid = u.id
                udata = u.to_dict()
                recent_activity = udata.get('recent_activity', [])
                
                user_vector = calculate_user_vector(recent_activity, current_hour=current_hour)
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
        finally:
            conn.close()
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"status": "error", "message": str(e)}

@router.post("/api/admin/cerebro/anchors")
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

@router.put("/api/admin/cerebro/anchors/{anchor_id}")
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

@router.delete("/api/admin/cerebro/anchors/{anchor_id}")
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

@router.get("/api/admin/cerebro")
def get_admin_cerebro(page: int = 1, limit: int = 10):
    """Devuelve telemetrÔö£┬ía detallada del Cerebro Vectorial para el panel Admin."""
    try:
        conn = get_db_connection()
        c = conn.cursor()
        
        # 1. Total Vectores Productos
        c.execute("SELECT COUNT(*) as c FROM product_vectors")
        total_product_vectors = c.fetchone()["c"]
        
        c.execute("""
            SELECT a.anchor_id, m.title, m.subtitle, m.section_type, a.embedding, m.is_manual
            FROM anchor_vectors a
            LEFT JOIN anchor_metadata m ON a.anchor_id = m.anchor_id
        """)
        
        anchors = []
        for row in c.fetchall():
            anchor_dict = dict(row)
            cluster_name = anchor_dict.get("section_type")
            
            if cluster_name and cluster_name in MACRO_CLUSTERS_CACHE:
                anchor_dict["keywords"] = MACRO_CLUSTERS_CACHE[cluster_name].get("keywords", "Generado por IA")
            else:
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
            "fts_clusters": MACRO_CLUSTERS_CACHE,
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

@router.get("/api/admin/metrics/engagement")
def get_engagement_metrics():
    """Dashboard metrics: search CTR, top queries, top categories, section performance."""
    try:
        conn = get_db_connection()
        c = conn.cursor()

        c.execute("SELECT COUNT(*) as total FROM search_logs")
        total_searches = c.fetchone()["total"]

        c.execute("SELECT COUNT(*) as total FROM search_logs WHERE clicked_id IS NOT NULL AND clicked_id != ''")
        searches_with_clicks = c.fetchone()["total"]

        ctr = round(searches_with_clicks / max(total_searches, 1) * 100, 1)

        c.execute("""
            SELECT query, COUNT(*) as count FROM search_logs
            WHERE query != '' GROUP BY query ORDER BY count DESC LIMIT 15
        """)
        top_queries = [dict(r) for r in c.fetchall()]

        c.execute("""
            SELECT clicked_category as category, COUNT(*) as count FROM search_logs
            WHERE clicked_category != '' GROUP BY clicked_category ORDER BY count DESC LIMIT 10
        """)
        top_categories = [dict(r) for r in c.fetchall()]

        c.execute("""
            SELECT section_id, COUNT(*) as impressions, SUM(clicked) as clicks
            FROM section_impressions
            GROUP BY section_id ORDER BY clicks DESC LIMIT 15
        """)
        section_perf = []
        for r in c.fetchall():
            r = dict(r)
            r["ctr"] = round((r["clicks"] or 0) / max(r["impressions"], 1) * 100, 1)
            section_perf.append(r)

        c.execute("""
            SELECT activity_type, COUNT(*) as count FROM user_activity_cache
            GROUP BY activity_type ORDER BY count DESC
        """)
        activity_funnel = [dict(r) for r in c.fetchall()]

        conn.close()
        return {
            "status": "ok",
            "total_searches": total_searches,
            "searches_with_clicks": searches_with_clicks,
            "ctr_pct": ctr,
            "top_queries": top_queries,
            "top_categories": top_categories,
            "section_performance": section_perf,
            "activity_funnel": activity_funnel,
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"status": "error", "message": str(e)}

@router.post("/api/admin/auto-generate-anchors")
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
            Eres un experto en taxonomÔö£┬ía de comercio electrÔö£Ôöénico e inteligencia artificial.
            AquÔö£┬í tienes una muestra de los productos y categorÔö£┬ías de nuestro supermercado/tienda:
            CategorÔö£┬ías: {categories}
            Muestra de productos: {products}
            
            Tu tarea es generar un arreglo JSON con las mejores "Anclas" (ClÔö£Ôòæsteres o categorÔö£┬ías semÔö£├¡nticas) para organizar este inventario en un motor de bÔö£Ôòæsqueda vectorial.
            El arreglo JSON debe contener entre 6 y 12 objetos con la siguiente estructura exacta:
            [
              {{
                "id": "A1",
                "titles": ["Mascotas", "Para tus peludos", "El rincÔö£Ôöén animal", "Mascotas felices"],
                "subtitle": "Todo para tu mejor amigo",
                "desc": "Alimentos y accesorios para mascotas",
                "allowed_categories": ["Mascotas", "Veterinaria", "Animales"],
                "exclude_rules": ["perro caliente", "salchicha"]
              }}
            ]
            En "titles", DEBES dar un arreglo de 4 opciones de tÔö£┬ítulos atractivos y dinÔö£├¡micos para esta categorÔö£┬ía.
            En "allowed_categories", debes poner un arreglo de strings seleccionando EXACTAMENTE los nombres de las categorÔö£┬ías proporcionadas en la lista 'CategorÔö£┬ías' que pertenecen a esta ancla. ESTO ES UN FILTRO ESTRICTO. Solo los productos de estas categorÔö£┬ías aparecerÔö£├¡n en esta ancla. Ôö¼├¡SÔö£┬« exhaustivo e incluye todas las categorÔö£┬ías relevantes de la lista!
            En "exclude_rules", incluye un arreglo de palabras clave que NO deben aparecer (por si hay ambigÔö£ÔòØedad).
            Devuelve SOLO EL JSON vÔö£├¡lido, sin cÔö£Ôöédigo de bloque extra ni markdown.
            '''
            models_to_try = ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-pro", "gemini-1.5-flash"]
            response = None
            for m in models_to_try:
                try:
                    model = genai.GenerativeModel(m)
                    response = model.generate_content(prompt)
                    if response:
                        print(f"Modelo {m} seleccionado exitosamente para generaciÔö£Ôöén.")
                        break
                except Exception as e:
                    print(f"Modelo {m} fallÔö£Ôöé: {e}")
                    
            if not response:
                raise Exception("Todos los modelos generativos fallaron o no estÔö£├¡n disponibles en esta API Key.")
                
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
            print("[Fase 1] Auto-GeneraciÔö£Ôöén de Anclas con IA completada exitosamente.")
            
            # --- FASE 2: CLUSTERS AMBIENTALES/FTS ---
            prompt_macro = f'''
            Eres un experto en comportamiento del consumidor. Revisa esta muestra de productos y categorÔö£┬ías de nuestro ecosistema:
            CategorÔö£┬ías: {categories}
            Muestra: {products}
            
            Genera reglas dinÔö£├¡micas de descubrimiento, con dos objetos en un JSON: "clusters" y "time_rules".
            Ejemplo de estructura esperada (DEVUELVE SOLO JSON VÔö£├╝LIDO SIN MARKDOWN):
            {{
              "clusters": {{
                 "calor_dia": {{
                    "titles": ["Para este calorcito", "DÔö£┬ías soleados"],
                    "keywords": "helado OR jugo OR pantaloneta",
                    "storeCategories": "HeladerÔö£┬ía, Ropa",
                    "negativeKeywords": "sopa OR chaqueta",
                    "relatedClusters": "postres"
                 }},
                 "calor_noche": {{
                    "titles": ["Noches cÔö£├¡lidas", "Refrescate esta noche"],
                    "keywords": "helado OR cerveza OR licor",
                    "storeCategories": "HeladerÔö£┬ía, Bar",
                    "negativeKeywords": "sopa OR tinto",
                    "relatedClusters": "licores"
                 }},
                 "frio_dia": {{
                    "titles": ["DÔö£┬ías frÔö£┬íos", "AcompaÔö£ÔûÆalo con cafÔö£┬«"],
                    "keywords": "cafe OR tinto OR chaqueta",
                    "storeCategories": "CafeterÔö£┬ía, Ropa",
                    "negativeKeywords": "helado",
                    "relatedClusters": "desayuno"
                 }},
                 "frio_noche": {{
                    "titles": ["Noches frÔö£┬ías", "No salgas de casa"],
                    "keywords": "sopa OR pizza OR hamburguesa",
                    "storeCategories": "Restaurante",
                    "negativeKeywords": "helado",
                    "relatedClusters": "comida_rapida"
                 }}
              }},
              "time_rules": [
                 {{"startHour": 5, "endHour": 10, "cluster": "desayuno", "scoreBoost": 5.0}}
              ]
            }}
            Debes definir al menos los clusters de clima ("clima_calor", "clima_frio") y algunos temporales (ej: desayuno, almuerzo, noche).
            Usa el operador OR en "keywords" y "negativeKeywords".
            '''
            macro_response = None
            for m in models_to_try:
                try:
                    model = genai.GenerativeModel(m)
                    macro_response = model.generate_content(prompt_macro)
                    if macro_response: break
                except Exception as e:
                    pass
            
            if macro_response:
                r_text = macro_response.text.strip()
                if r_text.startswith("```json"): r_text = r_text[7:]
                if r_text.startswith("```"): r_text = r_text[3:]
                if r_text.endswith("```"): r_text = r_text[:-3]
                
                macro_data = json.loads(r_text.strip())
                new_clusters = macro_data.get("clusters")
                new_time_rules = macro_data.get("time_rules")
                
                if new_clusters and new_time_rules and db:
                    # Sincronizar globalmente en Firebase
                    db.collection('config').document('algorithm').set({
                        "clusters": new_clusters,
                        "time_rules": new_time_rules
                    }, merge=True)
                    print("[Fase 2] Clusters Ambientales generados y sincronizados en Firebase.")
            
        except Exception as e:
            print("Error en Auto-GeneraciÔö£Ôöén (Fase 1/2):", e)
            
    background_tasks.add_task(run_generation)
    return {"status": "ok", "message": "Descubrimiento de anclas con IA iniciado en background. Espera un minuto."}

@router.post("/api/admin/reset-vectors")
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

@router.get("/api/admin/clusters")
def get_clusters():
    return {"status": "ok", "clusters": MACRO_CLUSTERS_CACHE}

@router.put("/api/admin/clusters")
def update_clusters(body: dict):
    """Update clusters in Firestore and hot-reload in memory."""
    try:
        new_clusters = body.get("clusters", {})
        if not new_clusters:
            return {"status": "error", "message": "No clusters provided"}
        if db:
            db.collection('config').document('algorithm').set({"clusters": new_clusters}, merge=True)
        MACRO_CLUSTERS_CACHE.clear()
        MACRO_CLUSTERS_CACHE.update(new_clusters)
        return {"status": "ok", "count": len(new_clusters)}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@router.get("/api/admin/concepts")
def get_concepts():
    from data.concepts import DICCIONARIO_CONCEPTOS_RAW, CATEGORY_WEIGHTS, DICCIONARIO_CONCEPTOS
    return {
        "status": "ok",
        "concepts": DICCIONARIO_CONCEPTOS_RAW,
        "category_weights": CATEGORY_WEIGHTS,
        "loaded_count": len(DICCIONARIO_CONCEPTOS),
    }

@router.post("/api/admin/build-concepts")
def build_concepts_endpoint(background_tasks: BackgroundTasks):
    """Reconstruye los vectores de conceptos ambientales (clima/hora) en background."""
    def _run():
        try:
            from data.concepts import build_concept_dictionary, cargar_conceptos_en_memoria
            build_concept_dictionary()
            cargar_conceptos_en_memoria()
            print("[Conceptos] Reconstrucción completada.")
        except Exception as e:
            print(f"[Conceptos] Error reconstruyendo: {e}")
    background_tasks.add_task(_run)
    return {"status": "processing", "message": "Construyendo conceptos ambientales en background."}

@router.post("/api/reset-clusters")
def reset_clusters_to_defaults():
    """Empuja los defaults del cÔö£Ôöédigo a Firestore, reemplazando los clÔö£Ôòæsteres existentes.
    Ôö£├£til cuando los clÔö£Ôòæsteres en Firestore estÔö£├¡n desactualizados (sin storeCategories, etc.)."""
    if not db:
        raise HTTPException(status_code=500, detail="Firebase no estÔö£├¡ inicializado.")
    try:
        doc_ref = db.collection('config').document('algorithm')
        doc_ref.set({"clusters": MACRO_CLUSTERS_CACHE}, merge=True)
        return {
            "message": f"├ö┬ú├á {len(MACRO_CLUSTERS_CACHE)} clÔö£Ôòæsteres reseteados a los defaults V3.2 correctamente.",
            "clusters_pushed": list(MACRO_CLUSTERS_CACHE.keys())
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ==========================================
# ==========================================
# WEBHOOKS PUSH PARA ACTUALIZAR Ôö£├¼NDICE (MINI-ALGOLIA)
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

from core.config import global_sync_state
from services.sync import do_sync_database, do_seed_anchors, do_sync_store, vector_worker_pool, async_index_product_vector

@router.post("/api/webhook/product")
def webhook_product_upsert(payload: ProductPayload):
    """Real-time product indexing: upsert in FTS5 + queue vector generation."""
    try:
        with sqlite_lock:
            conn = get_db_connection()
            c = conn.cursor()
            c.execute("DELETE FROM search_index WHERE id = ? AND type = 'product'", (payload.id,))
            c.execute(
                "INSERT INTO search_index (id, type, storeId, name, category, description, price, icon, imageUrl, onSale, salePrice, likes, views, purchases, available, isOpen) "
                "VALUES (?, 'product', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, 1, ?)",
                (payload.id, payload.storeId, payload.name, payload.category, payload.description,
                 payload.price, payload.icon, payload.imageUrl,
                 1 if payload.onSale else 0, payload.salePrice or 0,
                 payload.likes, payload.views, payload.isOpen)
            )
            conn.commit()
            conn.close()
        vector_worker_pool.submit(
            async_index_product_vector, payload.id, payload.name, payload.category or '', payload.description or ''
        )
        return {"status": "ok", "product_id": payload.id}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@router.post("/api/webhook/product-delete/{product_id}")
def webhook_product_delete(product_id: str):
    """Remove a product from FTS5 index and vectors."""
    try:
        with sqlite_lock:
            conn = get_db_connection()
            conn.execute("DELETE FROM search_index WHERE id = ? AND type = 'product'", (product_id,))
            conn.execute("DELETE FROM product_vectors WHERE product_id = ?", (product_id,))
            conn.execute("DELETE FROM vector_queue WHERE product_id = ?", (product_id,))
            conn.commit()
            conn.close()
        return {"status": "ok"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@router.get("/api/admin/sync-status")
def get_sync_status():
    return global_sync_state

@router.post("/api/sync")
def sync_database(background_tasks: BackgroundTasks):
    if global_sync_state.get("is_syncing", False):
        return {"status": "already_running"}
    global_sync_state["is_syncing"] = True
    global_sync_state["status"] = "Iniciando..."
    global_sync_state["completed_products"] = 0
    background_tasks.add_task(do_sync_database)
    return {"status": "processing"}

@router.post("/api/sync/store/{store_id}")
def sync_store(store_id: str, background_tasks: BackgroundTasks):
    background_tasks.add_task(do_sync_store, store_id)
    return {"status": "processing", "store_id": store_id}

@router.post("/api/seed-anchors")
def seed_anchors(background_tasks: BackgroundTasks):
    background_tasks.add_task(do_seed_anchors)
    return {"status": "processing"}
