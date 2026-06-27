import json
import logging
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List, Optional
import sqlite_vec

from core.database import get_db_connection, get_db_dep, sqlite_lock, init_db
import core.firebase
from core.genai_client import embed_text, generate_text
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
        users_ref = core.firebase.db.collection('users')
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
            
            CONCEPT_LABELS = {
                "ENV_CALOR": "☀️ Calor", "ENV_FRIO": "🌧️ Frío", "ENV_NOCHE": "🌙 Noche",
                "ENV_MANANA": "🌅 Mañana", "ENV_MEDIODIA": "🍽️ Mediodía", "ENV_SALUDABLE": "🥗 Saludable",
                "ENV_GUAYABO": "🤕 Guayabo", "ENV_PEREZA": "🛋️ Pereza",
            }

            for u in users:
                uid = u.id
                udata = u.to_dict()
                recent_activity = udata.get('recent_activity', [])

                user_vector = calculate_user_vector(recent_activity, current_hour=current_hour)
                has_vector = user_vector is not None
                nearest_concept = None
                if user_vector:
                    c.execute(
                        "SELECT id, vec_distance_cosine(embedding, ?) AS distance "
                        "FROM concept_vectors ORDER BY distance ASC LIMIT 1",
                        (user_vector,)
                    )
                    row = c.fetchone()
                    if row:
                        nearest_concept = {
                            "id": row["id"],
                            "label": CONCEPT_LABELS.get(row["id"], row["id"]),
                            "distance": row["distance"],
                        }

                if len(recent_activity) > 0:
                    results.append({
                        "uid": u.id,
                        "name": udata.get('name', udata.get('email', 'Usuario Anónimo')),
                        "activity_count": len(recent_activity),
                        "has_vector": has_vector,
                        "nearest_concept": nearest_concept,
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
        emb = None
        for attempt in range(3):
            try:
                emb = embed_text(text, task_type="retrieval_document")
                break
            except Exception as e:
                time.sleep(1)

        if not emb:
            return {"status": "error", "message": "Failed to generate embedding"}

        vector_blob = sqlite_vec.serialize_float32(emb)
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
        emb = None
        for attempt in range(3):
            try:
                emb = embed_text(text, task_type="retrieval_document")
                break
            except Exception as e:
                time.sleep(1)

        if not emb:
            return {"status": "error", "message": "Failed to generate embedding"}

        vector_blob = sqlite_vec.serialize_float32(emb)
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
def get_admin_cerebro(page: int = 1, limit: int = 10, store_page: int = 1, anchor_page: int = 1):
    """Devuelve telemetrÔö£┬ía detallada del Cerebro Vectorial para el panel Admin."""
    try:
        conn = get_db_connection()
        c = conn.cursor()

        # 1. Total Vectores Productos
        c.execute("SELECT COUNT(*) as c FROM product_vectors")
        total_product_vectors = c.fetchone()["c"]

        c.execute("SELECT COUNT(*) as c FROM store_vectors")
        total_store_vectors = c.fetchone()["c"]

        # Diagnóstico de vectorización: indexados vs vectorizados vs en cola de reintentos
        c.execute("SELECT COUNT(*) as c FROM search_index WHERE type='product'")
        total_products_indexed = c.fetchone()["c"]
        c.execute("SELECT COUNT(*) as c FROM search_index WHERE type='store'")
        total_stores_indexed = c.fetchone()["c"]
        try:
            c.execute("SELECT COUNT(*) as c FROM vector_queue")
            vector_queue_count = c.fetchone()["c"]
        except Exception:
            vector_queue_count = 0
        
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

        # 4. N comercios vectorizados (Paginados)
        store_offset = (store_page - 1) * limit
        c.execute("""
            SELECT v.store_id, s.name, s.category, length(v.embedding) as vec_bytes
            FROM store_vectors v
            JOIN search_index s ON v.store_id = s.id AND s.type = 'store'
            LIMIT ? OFFSET ?
        """, (limit, store_offset))
        sample_stores = [dict(row) for row in c.fetchall()]

        conn.close()

        from data.synonyms import SYNONYMS

        return {
            "status": "ok",
            "fts_clusters": MACRO_CLUSTERS_CACHE,
            "synonyms": SYNONYMS,
            "vector_metrics": {
                "total_product_vectors": total_product_vectors,
                "total_store_vectors": total_store_vectors,
                "total_products_indexed": total_products_indexed,
                "total_stores_indexed": total_stores_indexed,
                "vector_queue_count": vector_queue_count,
                "anchors_count": len(anchors),
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
            SELECT section_id, impressions, clicks
            FROM section_stats
            ORDER BY clicks DESC LIMIT 15
        """)
        section_perf = []
        for r in c.fetchall():
            r = dict(r)
            r["ctr"] = round((r["clicks"] or 0) / max(r["impressions"] or 0, 1) * 100, 1)
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
            response_text = None
            for m in models_to_try:
                try:
                    response_text = generate_text(prompt, model=m)
                    if response_text:
                        print(f"Modelo {m} seleccionado exitosamente para generación.")
                        break
                except Exception as e:
                    print(f"Modelo {m} falló: {e}")

            if not response_text:
                raise Exception("Todos los modelos generativos fallaron o no están disponibles en esta API Key.")

            raw_text = response_text.strip()
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
                emb = None
                for attempt in range(3):
                    try:
                        emb = embed_text(text, task_type="retrieval_document")
                        break
                    except Exception as e:
                        time.sleep(2 ** attempt)

                if emb:
                    vector_blob = sqlite_vec.serialize_float32(emb)
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
            macro_text = None
            for m in models_to_try:
                try:
                    macro_text = generate_text(prompt_macro, model=m)
                    if macro_text: break
                except Exception as e:
                    pass

            if macro_text:
                r_text = macro_text.strip()
                if r_text.startswith("```json"): r_text = r_text[7:]
                if r_text.startswith("```"): r_text = r_text[3:]
                if r_text.endswith("```"): r_text = r_text[:-3]
                
                macro_data = json.loads(r_text.strip())
                new_clusters = macro_data.get("clusters")
                new_time_rules = macro_data.get("time_rules")
                
                if new_clusters and new_time_rules and core.firebase.db:
                    # Sincronizar globalmente en Firebase
                    core.firebase.db.collection('config').document('algorithm').set({
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

class SynonymGroup(BaseModel):
    root: str
    alternatives: List[str] = []

@router.get("/api/admin/synonyms")
def get_synonyms():
    from data.synonyms import SYNONYMS
    return {"status": "ok", "synonyms": SYNONYMS}

@router.post("/api/admin/synonyms")
def upsert_synonym(req: SynonymGroup):
    from data.synonyms import set_synonym_group
    try:
        set_synonym_group(core.firebase.db, req.root, req.alternatives)
        return {"status": "ok"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@router.delete("/api/admin/synonyms/{root}")
def remove_synonym(root: str):
    from data.synonyms import delete_synonym_group
    try:
        delete_synonym_group(core.firebase.db, root)
        return {"status": "ok"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@router.post("/api/admin/auto-learn-synonyms")
def auto_learn_synonyms():
    """Aprendizaje matemático de sinónimos por co-clics (sin IA)."""
    from data.synonyms import learn_synonyms_from_clicks
    try:
        learned = learn_synonyms_from_clicks(core.firebase.db)
        return {"status": "ok", "learned": learned, "count": len(learned)}
    except Exception as e:
        import traceback
        traceback.print_exc()
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
        if core.firebase.db:
            core.firebase.db.collection('config').document('algorithm').set({"clusters": new_clusters}, merge=True)
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

@router.put("/api/admin/concepts")
def update_concepts(body: dict, background_tasks: BackgroundTasks):
    """Edita los textos de los conceptos ambientales (clima/hora), persiste en Firestore y regenera embeddings."""
    try:
        texts = body.get("concepts", {})
        if not texts:
            return {"status": "error", "message": "No concepts provided"}
        from data.concepts import CONCEPTOS_SEMILLA, DICCIONARIO_CONCEPTOS_RAW
        for k, v in texts.items():
            if isinstance(v, str) and v.strip():
                CONCEPTOS_SEMILLA[k] = v
                DICCIONARIO_CONCEPTOS_RAW[k] = v
        if core.firebase.db:
            core.firebase.db.collection('config').document('concepts').set({"texts": CONCEPTOS_SEMILLA}, merge=True)

        def _rebuild():
            try:
                from data.concepts import build_concept_dictionary, cargar_conceptos_en_memoria
                build_concept_dictionary()
                cargar_conceptos_en_memoria()
                print("[Conceptos] Embeddings regenerados tras edición.")
            except Exception as e:
                print(f"[Conceptos] Error regenerando: {e}")
        background_tasks.add_task(_rebuild)
        return {"status": "ok", "message": "Conceptos guardados. Regenerando embeddings en background."}
    except Exception as e:
        return {"status": "error", "message": str(e)}

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
    if not core.firebase.db:
        raise HTTPException(status_code=500, detail="Firebase no estÔö£├¡ inicializado.")
    try:
        doc_ref = core.firebase.db.collection('config').document('algorithm')
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
from services.sync import do_sync_database, do_seed_anchors, do_sync_store, vector_worker_pool, async_index_product_vector, index_store_vector

@router.post("/api/index/product")
def webhook_product_upsert(payload: ProductPayload):
    """Indexado en tiempo real (lo llama la app de comercio al guardar): upsert FTS5 + vector."""
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

@router.delete("/api/index/product/{product_id}")
def webhook_product_delete(product_id: str):
    """Quita un producto del índice y sus vectores (lo llama la app al agotar/eliminar)."""
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


class StorePayload(BaseModel):
    id: str
    name: str
    category: Optional[str] = ""
    description: Optional[str] = ""
    imageUrl: Optional[str] = ""
    isOpen: Optional[bool] = True


@router.post("/api/index/store")
def index_store(payload: StorePayload):
    """Indexado en tiempo real del comercio (lo llama la app al guardar el perfil): upsert FTS5 + vector."""
    try:
        with sqlite_lock:
            conn = get_db_connection()
            conn.execute("DELETE FROM search_index WHERE id = ? AND type = 'store'", (payload.id,))
            conn.execute(
                "INSERT INTO search_index (id, type, storeId, name, category, description, price, icon, imageUrl, onSale, salePrice, likes, views, purchases, available, isOpen) "
                "VALUES (?, 'store', ?, ?, ?, ?, '0', '', ?, 0, '', 0, 0, 0, 1, ?)",
                (payload.id, payload.id, payload.name, payload.category or '', payload.description or '',
                 payload.imageUrl or '', 1 if payload.isOpen else 0)
            )
            conn.commit()
            conn.close()
        vector_worker_pool.submit(
            index_store_vector, payload.id, payload.name, payload.category or '', payload.description or ''
        )
        return {"status": "ok", "store_id": payload.id}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@router.patch("/api/index/store/{store_id}/status")
def index_store_status(store_id: str, isOpen: bool = True):
    """Actualiza solo el estado abierto/cerrado del comercio en el índice (ultra-rápido)."""
    try:
        val = 1 if isOpen else 0
        with sqlite_lock:
            conn = get_db_connection()
            conn.execute("UPDATE search_index SET isOpen=? WHERE id=? AND type='store'", (val, store_id))
            conn.execute("UPDATE search_index SET isOpen=? WHERE storeId=? AND type='product'", (val, store_id))
            conn.commit()
            conn.close()
        return {"status": "ok", "isOpen": isOpen}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@router.get("/api/admin/sync-status")
def get_sync_status():
    return global_sync_state

DEMO_STORES = [
    ("Restaurante", "Sazón de la Abuela", ["Bandeja Paisa", "Sancocho de Gallina", "Churrasco", "Mojarra Frita", "Ajiaco Santafereño", "Pechuga Gratinada", "Sopa de Costilla", "Arroz con Pollo", "Cazuela de Mariscos", "Lomo al Trapo"]),
    ("Comidas rápidas", "El Rey del Perro", ["Hamburguesa Doble", "Perro Caliente Especial", "Salchipapa", "Picada Personal", "Choriperro", "Alitas BBQ", "Burger con Tocineta", "Papas a la Francesa", "Hot Dog Americano", "Sandwich Cubano"]),
    ("Panadería", "Pan Caliente", ["Pan de Bono", "Almojábana", "Pandebono", "Buñuelo", "Croissant", "Roscón con Arequipe", "Mantecada", "Torta de Naranja", "Empanada de Pollo", "Pan Aliñado"]),
    ("Mercados", "Mercamás Express", ["Arroz 1kg", "Aceite Premier", "Huevos AA x30", "Leche Entera", "Panela", "Frijol Cargamanto", "Café Molido", "Atún en Lata", "Azúcar 1kg", "Papel Higiénico x4"]),
    ("Spa & Belleza", "Bella Spa", ["Masaje Relajante", "Limpieza Facial", "Manicure Semipermanente", "Pedicure Spa", "Depilación", "Tratamiento Capilar", "Maquillaje Social", "Exfoliación Corporal", "Pestañas Pelo a Pelo", "Diseño de Cejas"]),
    ("Farmacia", "Farmacia La Salud", ["Acetaminofén", "Ibuprofeno 400", "Suero Oral", "Vitamina C", "Alcohol Antiséptico", "Curas x10", "Gel Antibacterial", "Loratadina", "Omeprazol", "Jarabe para la Tos"]),
    ("Licores", "Licorera La 70", ["Aguardiente Antioqueño", "Ron Medellín", "Cerveza Águila x6", "Whisky 12 Años", "Vino Tinto", "Tequila Reposado", "Vodka", "Cerveza Corona x6", "Ginebra", "Hielo x2kg"]),
    ("Ropa & Moda", "Moda Urbana", ["Camiseta Básica", "Jean Slim", "Vestido Casual", "Tenis Urbanos", "Buzo con Capucha", "Gorra", "Chaqueta Jean", "Blusa Manga Larga", "Short Deportivo", "Medias x3"]),
    ("Tecnología", "TecnoPunto", ["Audífonos Bluetooth", "Cargador Tipo C", "Power Bank 10000mAh", "Mouse Inalámbrico", "Cable HDMI", "Memoria USB 64GB", "Soporte Celular", "Teclado Bluetooth", "Protector de Pantalla", "Parlante Portátil"]),
    ("Barbería", "Barbería El Corte", ["Corte Clásico", "Corte + Barba", "Perfilado de Barba", "Corte Niño", "Mascarilla Negra", "Tinte Capilar", "Línea de Diseño", "Cejas Hombre", "Ritual Hot Towel", "Corte Premium"]),
]


def _do_seed_demo():
    """Crea 10 comercios x 10 productos en Firestore + los indexa y vectoriza."""
    import random
    from firebase_admin import firestore as _fs
    fdb = core.firebase.db
    try:
        rows = []  # (tuple para search_index)
        embed_jobs = []  # (callable args)
        for i, (cat, sname, products) in enumerate(DEMO_STORES):
            sid = f"demo_store_{i+1}"
            sdesc = f"El mejor lugar de {cat.lower()} en tu zona."
            s_likes = random.randint(5, 200)
            fdb.collection('stores').document(sid).set({
                "name": sname, "category": cat, "description": sdesc,
                "isOpen": True, "likes": s_likes, "views": random.randint(50, 1000),
                "isDemo": True, "createdAt": _fs.SERVER_TIMESTAMP,
            })
            rows.append((sid, 'store', sid, sname, cat, sdesc, '0', '', '', 0, '', s_likes, 0, 0, 1, 1))
            embed_jobs.append(('store', sid, sname, cat, sdesc))

            for j, pname in enumerate(products):
                pid = f"demo_prod_{i+1}_{j+1}"
                price = random.choice([5000, 8000, 12000, 15000, 20000, 25000, 30000])
                on_sale = random.random() < 0.3
                sale_price = int(price * 0.8) if on_sale else None
                pdesc = f"{pname} de {sname}."
                p_likes = random.randint(0, 80)
                p_views = random.randint(10, 400)
                p_purch = random.randint(0, 40)
                fdb.collection('stores').document(sid).collection('products').document(pid).set({
                    "name": pname, "storeId": sid, "category": cat, "description": pdesc,
                    "price": price, "available": True, "onSale": on_sale, "salePrice": sale_price,
                    "likes": p_likes, "views": p_views, "purchases": p_purch,
                    "isDemo": True, "createdAt": _fs.SERVER_TIMESTAMP,
                })
                rows.append((pid, 'product', sid, pname, cat, pdesc, str(price), 'fastfood', '',
                             1 if on_sale else 0, str(sale_price or ''), p_likes, p_views, p_purch, 1, 1))
                embed_jobs.append(('product', pid, pname, cat, pdesc))

        # Indexar todo en SQLite de una vez
        with sqlite_lock:
            conn = get_db_connection()
            for r in rows:
                conn.execute("DELETE FROM search_index WHERE id = ? AND type = ?", (r[0], r[1]))
                conn.execute(
                    "INSERT INTO search_index (id, type, storeId, name, category, description, price, icon, imageUrl, onSale, salePrice, likes, views, purchases, available, isOpen) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", r)
            conn.commit()
            conn.close()

        # Vectorizar en background
        for job in embed_jobs:
            if job[0] == 'store':
                vector_worker_pool.submit(index_store_vector, job[1], job[2], job[3], job[4])
            else:
                vector_worker_pool.submit(async_index_product_vector, job[1], job[2], job[3], job[4])

        global_sync_state["status"] = f"Demo creada: {len(DEMO_STORES)} comercios, {len(rows) - len(DEMO_STORES)} productos"
        print(f"[Seed Demo] Listo: {len(rows)} filas indexadas, vectorizando en background.")
    except Exception as e:
        import traceback
        traceback.print_exc()
        global_sync_state["status"] = f"error demo: {e}"
    finally:
        global_sync_state["is_syncing"] = False


@router.post("/api/admin/seed-demo")
def seed_demo(background_tasks: BackgroundTasks):
    if not core.firebase.db:
        return {"status": "error", "message": "Firebase no inicializado"}
    if global_sync_state.get("is_syncing", False):
        return {"status": "already_running"}
    global_sync_state["is_syncing"] = True
    global_sync_state["status"] = "Creando datos de prueba..."
    background_tasks.add_task(_do_seed_demo)
    return {"status": "processing"}


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
