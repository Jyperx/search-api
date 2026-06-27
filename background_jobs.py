import sqlite3
import sqlite_vec
import time
import json
import threading
import google.generativeai as genai
from datetime import datetime, timezone, timedelta
from google.cloud.firestore_v1.base_query import FieldFilter

from database import get_db_connection_raw, init_db, db
from core.config import (global_sync_state, EMBEDDING_MODEL, LLM_MODEL, SQLITE_DB,
                         MACRO_CLUSTERS_CACHE, TIME_RULES_CACHE, SYNONYMS,
                         load_synonyms_from_firestore)
from services.embeddings import (async_index_product_vector, async_index_store_vector,
                                 vector_worker_pool)

ENV_ANCHORS = [
    {"id": "ENV_FRIO_NOCHE", "title": "Ideal para el frío de hoy", "subtitle": "Combate el frío", "desc": "Noche fría y lluviosa. Comida caliente, caldos, domicilios rápidos, sopas, pizza, hamburguesas, cobijas, sacos."},
    {"id": "ENV_CALOR_DIA", "title": "Refréscate del calor", "subtitle": "Perfecto para este sol", "desc": "Día soleado y caluroso. Bebidas frías, helados, jugos, paletas, ensaladas de frutas, ropa fresca, gafas."},
    {"id": "ENV_FRIO_DIA", "title": "Entra en calor", "subtitle": "Acompáñalo con algo caliente", "desc": "Día frío o nublado. Café, tinto, chocolate caliente, empanadas, panadería, postres, tamal, chaquetas."},
    {"id": "ENV_CALOR_NOCHE", "title": "Noche cálida", "subtitle": "Para compartir y refrescarte", "desc": "Noche calurosa. Helado, cerveza, licor, cócteles, refrescos fríos, bebidas heladas."}
]

def do_seed_anchors():
    """Lógica interna para sembrar anclas en segundo plano."""
    global global_sync_state
    global_sync_state["is_syncing"] = True
    global_sync_state["total_products"] = 1
    global_sync_state["completed_products"] = 0
    global_sync_state["status"] = "Analizando categorías..."
    try:
        conn = get_db_connection_raw()
        c = conn.cursor()
        c.execute("DELETE FROM anchor_vectors")
        c.execute("DELETE FROM anchor_metadata")
        
        c.execute("""
            SELECT category, COUNT(*) as c 
            FROM search_index 
            WHERE type='product' AND CAST(available AS INTEGER) = 1 
            GROUP BY category 
            ORDER BY c DESC 
            LIMIT 30
        """)
        category_rows = c.fetchall()
        
        conn.commit()
        conn.close()
            
        dynamic_anchors = []
        import hashlib
        for i, row in enumerate(category_rows):
            cat_name = row["category"]
            if not cat_name or str(cat_name).strip().lower() == "general": continue
            
            cat_id = "DYN_CAT_" + hashlib.md5(cat_name.encode()).hexdigest()[:8]
            
            dynamic_anchors.append({
                "id": cat_id,
                "title": f"Todo en {cat_name.title()}",
                "subtitle": f"Tus favoritos de {cat_name.lower()}",
                "desc": f"Catálogo completo de {cat_name.lower()} y similares.",
                "titles": json.dumps([f"Todo en {cat_name.title()}", f"Lo mejor de {cat_name.title()}", f"Tus favoritos de {cat_name.title()}", f"Explora {cat_name.title()}"]),
                "allowed_categories": json.dumps([cat_name.lower()])
            })
            
        all_anchors = dynamic_anchors + ENV_ANCHORS
        global_sync_state["total_products"] = len(all_anchors)
        global_sync_state["status"] = f"Sembrando {len(all_anchors)} anclas..."
        
        for a in all_anchors:
            text = f"{a['title']} - {a['desc']}"
            res = None
            for attempt in range(3):
                try:
                    res = genai.embed_content(model=EMBEDDING_MODEL, content=text, task_type="retrieval_document")
                    break
                except Exception as e:
                    print(f"Error in embed_content (attempt {attempt}):", e)
                    time.sleep(2 ** attempt)
            
            time.sleep(2)
            
            if res and 'embedding' in res:
                vector_blob = sqlite_vec.serialize_float32(res['embedding'][:768])
                conn = get_db_connection_raw()
                c = conn.cursor()
                c.execute(
                    "INSERT INTO anchor_metadata (anchor_id, title, subtitle, section_type, titles, allowed_categories) VALUES (?, ?, ?, 'products', ?, ?)",
                    (a['id'], a['title'], a['subtitle'], a.get('titles'), a.get('allowed_categories'))
                )
                c.execute(
                    "INSERT INTO anchor_vectors (anchor_id, embedding) VALUES (?, ?)",
                    (a['id'], vector_blob)
                )
                conn.commit()
                conn.close()
                    
            global_sync_state["completed_products"] += 1
                    
        print(f"Vectores ancla dinámicos sembrados ({len(dynamic_anchors)} categorias + {len(ENV_ANCHORS)} ambientales).")
        global_sync_state["status"] = "Completado"
    except Exception as e:
        print("Error seeding anchors en bg:", e)
        global_sync_state["status"] = f"Error: {e}"
    finally:
        global_sync_state["is_syncing"] = False

def do_sync_database():
    """Lógica interna de sincronización en segundo plano."""
    global global_sync_state
    try:
        if not db:
            global_sync_state["is_syncing"] = False
            global_sync_state["status"] = "Error: Firebase no está inicializado"
            return
            
        conn = get_db_connection_raw()
        c = conn.cursor()
        
        c.execute("DELETE FROM search_index")
        c.execute("DELETE FROM promotions")
        c.execute("DELETE FROM product_vectors")
        c.execute("DELETE FROM store_vectors")
        c.execute("DELETE FROM anchor_vectors")
        conn.commit()
        conn.close()
    
        now_ms = int(time.time() * 1000)
        camps_ref = db.collection("marketing_campaigns")
        camps = list(camps_ref.stream())
        
        count_banners = 0
        if len(camps) > 0:
            conn = get_db_connection_raw()
            c = conn.cursor()
            for promo in camps:
                p_data = promo.to_dict()
                if p_data.get('type') in ['simple', 'premium_product', 'premium_store']:
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
            default_ads = [
                ("1", "simple", "store", "", "", "local-offer", "Descubre Ofertas", "En los mejores comercios", "#FFE4E1", "#DC143C", "#CD5C5C")
            ]
            conn = get_db_connection_raw()
            c = conn.cursor()
            for ad in default_ads:
                c.execute("INSERT INTO promotions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", ad)
            conn.commit()
            conn.close()

        global_sync_state["status"] = "Leyendo Comercios..."
        
        stores_ref = db.collection("stores")
        stores = list(stores_ref.stream())
        
        count = 0
        total_items_to_process = len(stores)
        
        stores_data = []
        for store in stores:
            s_data = store.to_dict()
            s_id = store.id
            products_ref = stores_ref.document(s_id).collection("products")
            products = list(products_ref.stream())
            total_items_to_process += len(products)
            stores_data.append((s_id, s_data, products))
            
        global_sync_state["total_products"] = total_items_to_process
        global_sync_state["status"] = "Vectorizando datos con IA..."
        
        for s_id, s_data, products in stores_data:
            conn = get_db_connection_raw()
            c = conn.cursor()
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
            
            conn = get_db_connection_raw()
            c = conn.cursor()
            store_product_names = []
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
                    store_product_names.append(p_data.get('name', ''))
                    vector_worker_pool.submit(
                        async_index_product_vector, 
                        product.id, 
                        p_data.get('name', ''), 
                        p_data.get('category', ''), 
                        p_data.get('description', '')
                    )
                else:
                    global_sync_state["completed_products"] += 1
            conn.commit()
            conn.close()
                
            products_summary = ", ".join(store_product_names[:10])
            vector_worker_pool.submit(async_index_store_vector, s_id, s_data.get('name', ''), s_data.get('category', ''), s_data.get('description', ''), products_summary)
        
        # Sincronización de anclas al final
        do_seed_anchors()
        
    except Exception as e:
        print("Sync Error:", e)
        global_sync_state["is_syncing"] = False
        global_sync_state["status"] = f"Error: {str(e)}"

def run_generation_task():
    global global_sync_state
    global_sync_state["is_syncing"] = True
    global_sync_state["total_products"] = 0
    global_sync_state["completed_products"] = 0
    global_sync_state["status"] = "Analizando taxonomía con Gemini..."
    try:
        conn = get_db_connection_raw()
        try:
            c = conn.cursor()
            c.execute("SELECT DISTINCT category FROM search_index WHERE type='product'")
            categories = [row['category'] for row in c.fetchall() if row['category']]
            c.execute("SELECT name, description, category FROM search_index WHERE type='product' ORDER BY RANDOM() LIMIT 100")
            products = [dict(row) for row in c.fetchall()]
        finally:
            conn.close()
            
        prompt = f'''
        Eres un experto en taxonomía de comercio electrónico e inteligencia artificial.
        Aquí tienes una muestra de los productos y categorías de nuestro supermercado/tienda:
        Categorías: {categories}
        Muestra de productos: {products}
        
        Tu tarea es generar un arreglo JSON con las mejores "Anclas" (Clústeres o categorías semánticas) para organizar este inventario en un motor de búsqueda vectorial.
        El arreglo JSON debe contener entre 6 y 12 objetos con la siguiente estructura exacta:
        [
          {{
            "id": "A1",
            "titles": ["Mascotas", "Para tus peludos", "El rincón animal", "Mascotas felices"],
            "subtitle": "Todo para tu mejor amigo",
            "desc": "Alimentos y accesorios para mascotas",
            "allowed_categories": ["Mascotas", "Veterinaria", "Animales"],
            "exclude_rules": ["perro caliente", "salchicha"]
          }}
        ]
        En "titles", DEBES dar un arreglo de 4 opciones de títulos atractivos y dinámicos para esta categoría.
        En "allowed_categories", debes poner un arreglo de strings seleccionando EXACTAMENTE los nombres de las categorías proporcionadas en la lista 'Categorías' que pertenecen a esta ancla. ESTO ES UN FILTRO ESTRICTO. Solo los productos de estas categorías aparecerán en esta ancla. ¡Sé exhaustivo e incluye todas las categorías relevantes de la lista!
        En "exclude_rules", incluye un arreglo de palabras clave que NO deben aparecer (por si hay ambigüedad).
        Devuelve SOLO EL JSON válido, sin código de bloque extra ni markdown.
        '''
        models_to_try = ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-pro", "gemini-1.5-flash"]
        response = None
        for m in models_to_try:
            try:
                model = genai.GenerativeModel(m)
                response = model.generate_content(prompt)
                if response:
                    print(f"Modelo {m} seleccionado exitosamente para generación.")
                    break
            except Exception as e:
                print(f"Modelo {m} falló: {e}")
                
        if not response:
            raise Exception("Todos los modelos generativos fallaron o no están disponibles en esta API Key.")
            
        raw_text = response.text.strip()
        if raw_text.startswith("```json"): raw_text = raw_text[7:]
        if raw_text.startswith("```"): raw_text = raw_text[3:]
        if raw_text.endswith("```"): raw_text = raw_text[:-3]
        
        anchors_data = json.loads(raw_text.strip())
        global_sync_state["total_products"] = len(anchors_data)
        global_sync_state["status"] = "Vectorizando anclas..."
        
        conn = get_db_connection_raw()
        try:
            c = conn.cursor()
            c.execute("SELECT anchor_id FROM anchor_metadata WHERE is_manual = 0")
            old_ai_anchors = [row[0] for row in c.fetchall()]
            
            for oid in old_ai_anchors:
                c.execute("DELETE FROM anchor_metadata WHERE anchor_id = ?", (oid,))
                c.execute("DELETE FROM anchor_vectors WHERE anchor_id = ?", (oid,))
            conn.commit()
        finally:
            conn.close()
            
        for a in anchors_data:
            primary_title = a.get('titles', [a.get('title', 'Explorar')])[0]
            text = f"{primary_title} - {a.get('desc', '')}"
            res = None
            for attempt in range(3):
                try:
                    res = genai.embed_content(model=EMBEDDING_MODEL, content=text, task_type="retrieval_document")
                    break
                except Exception as e:
                    time.sleep(2 ** attempt)
            
            if res and 'embedding' in res:
                vector_blob = sqlite_vec.serialize_float32(res['embedding'][:768])
                conn = get_db_connection_raw()
                try:
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
                finally:
                    conn.close()
            global_sync_state["completed_products"] += 1
                    
        print("[Fase 1] Auto-Generación de Anclas con IA completada exitosamente.")
        global_sync_state["status"] = "Generando clusters ambientales..."
        
        prompt_macro = f'''
        Eres un experto en comportamiento del consumidor. Revisa esta muestra de productos y categorías de nuestro ecosistema:
        Categorías: {categories}
        Muestra: {products}
        
        Genera reglas dinámicas de descubrimiento, con dos objetos en un JSON: "clusters" y "time_rules".
        Ejemplo de estructura esperada (DEVUELVE SOLO JSON VÁLIDO SIN MARKDOWN):
        {{
          "clusters": {{
             "calor_dia": {{
                "titles": ["Para este calorcito", "Días soleados"],
                "keywords": "helado OR jugo OR pantaloneta",
                "storeCategories": "Heladería, Ropa",
                "negativeKeywords": "sopa OR chaqueta",
                "relatedClusters": "postres"
             }},
             "calor_noche": {{
                "titles": ["Noches cálidas", "Refrescate esta noche"],
                "keywords": "helado OR cerveza OR licor",
                "storeCategories": "Heladería, Bar",
                "negativeKeywords": "sopa OR tinto",
                "relatedClusters": "licores"
             }},
             "frio_dia": {{
                "titles": ["Días fríos", "Acompañalo con café"],
                "keywords": "cafe OR tinto OR chaqueta",
                "storeCategories": "Cafetería, Ropa",
                "negativeKeywords": "helado",
                "relatedClusters": "desayuno"
             }},
             "frio_noche": {{
                "titles": ["Noches frías", "No salgas de casa"],
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
            except:
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
                db.collection('config').document('algorithm').set({
                    "clusters": new_clusters,
                    "time_rules": new_time_rules
                }, merge=True)
                print("[Fase 2] Clusters Ambientales generados y sincronizados en Firebase.")
        
        global_sync_state["status"] = "Completado"
        
    except Exception as e:
        print("Error en Auto-Generación (Fase 1/2):", e)
        global_sync_state["status"] = f"Error: {str(e)}"
    finally:
        global_sync_state["is_syncing"] = False

async def run_auto_learn_synonyms():
    """Analiza logs de búsqueda y auto-aprende sinónimos colombianos reales."""
    print("[Auto-Learn] Iniciando fase de descubrimiento...")
    try:
        conn = get_db_connection_raw()
        try:
            c = conn.cursor()
            c.execute('''
                SELECT query, clicked_id, COUNT(*) as clicks
                FROM search_logs
                GROUP BY query, clicked_id
                HAVING clicks >= 1
            ''')
            rows = c.fetchall()
        finally:
            conn.close()
            
        if not rows:
            print("[Auto-Learn] Sin datos suficientes para aprender.")
            return

        product_queries = {}
        for row in rows:
            q, pid, count = row
            if pid not in product_queries: product_queries[pid] = []
            product_queries[pid].append({"query": q, "count": count})

        candidates_to_evaluate = []
        for pid, queries in product_queries.items():
            if len(queries) > 1:
                words = [q["query"] for q in queries if len(q["query"]) > 2]
                if len(words) > 1:
                    candidates_to_evaluate.append(words)

        if not candidates_to_evaluate:
            print("[Auto-Learn] No hay candidatos suficientes.")
            return

        print(f"[Auto-Learn] {len(candidates_to_evaluate)} grupos de candidatos a evaluar por Gemini.")
        
        prompt = f"""Actúa como un experto lingüista colombiano. Revisa estos grupos de palabras que los usuarios buscaron y que terminaron en el mismo producto.
        Identifica cuáles son sinónimos reales o jerga local, y descarta los que son casualidades o palabras genéricas.
        
        Candidatos:
        {json.dumps(candidates_to_evaluate)}
        
        Devuelve SOLO un JSON con los nuevos sinónimos validados. Agrupa por la palabra más común como 'root'.
        Ejemplo: {{"auto_synonyms": {{"hamburguesa": ["burger", "burguer", "hambur"]}} }}
        """
        
        model = genai.GenerativeModel("gemini-2.5-flash")
        response = model.generate_content(prompt)
        
        r_text = response.text.strip()
        if r_text.startswith("```json"): r_text = r_text[7:]
        if r_text.startswith("```"): r_text = r_text[3:]
        if r_text.endswith("```"): r_text = r_text[:-3]
        
        data = json.loads(r_text.strip())
        new_syns = data.get("auto_synonyms", {})
        
        if new_syns and db:
            db.collection('config').document('synonyms').set({
                "auto_synonyms": new_syns
            }, merge=True)
            print(f"[Auto-Learn] {len(new_syns)} nuevos grupos de sinónimos descubiertos y guardados.")
            load_synonyms_from_firestore()
            
    except Exception as e:
        print(f"Error en auto-learn-synonyms: {e}")

def run_auto_learn_synonyms_sync():
    import asyncio
    asyncio.run(run_auto_learn_synonyms())

def on_stores_snapshot(col_snapshot, changes, read_time):
    try:
        conn = get_db_connection_raw()
        try:
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
                    vector_worker_pool.submit(
                        async_index_store_vector,
                        s_id,
                        s_data.get('name', ''),
                        s_data.get('category', ''),
                        s_data.get('description', ''),
                        ''
                    )
                elif change.type.name == 'REMOVED':
                    c.execute("DELETE FROM search_index WHERE id = ? AND type = 'store'", (s_id,))
            conn.commit()
            print(f"[Realtime Sync] Procesados {len(changes)} cambios en Stores")
        finally:
            conn.close()
    except Exception as e:
        print(f"Error en on_stores_snapshot: {e}")

def delta_sync_loop():
    if not db:
        print("Firebase no inicializado. No se puede iniciar delta sync.")
        return

    while True:
        try:
            conn = get_db_connection_raw()
            try:
                c = conn.cursor()
                c.execute("SELECT value FROM metadata WHERE key = 'last_sync_time'")
                row = c.fetchone()
                last_sync_str = row[0] if row else None
            finally:
                conn.close()
            
            if not last_sync_str:
                print("[Delta Sync] Primer arranque o SQLite vacío. Sincronizando todo el catálogo...")
                try:
                    do_sync_database() # Llamamos directamente a do_sync_database (evita bug de BackgroundTasks en thread)
                    current_time = datetime.now(timezone.utc).isoformat()
                    conn = get_db_connection_raw()
                    try:
                        c = conn.cursor()
                        c.execute("INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)", ('last_sync_time', current_time))
                        conn.commit()
                    finally:
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
                    conn = get_db_connection_raw()
                    try:
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
                            vector_worker_pool.submit(
                                async_index_store_vector,
                                s_id,
                                s_data.get('name', ''),
                                s_data.get('category', ''),
                                s_data.get('description', ''),
                                ''
                            )
                            
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
                        print(f"[Delta Sync] Sincronización exitosa. Siguiente chequeo desde {current_time}.")
                    finally:
                        conn.close()

        except Exception as e:
            print(f"[Delta Sync Error]: {e}")

        time.sleep(60)

def cleanup_activity_loop():
    if not db:
        return
        
    time.sleep(60)
    
    while True:
        try:
            thirty_days_ago = datetime.now(timezone.utc) - timedelta(days=30)
            
            old_activities = db.collection_group('activity').where(filter=FieldFilter("timestamp", "<", thirty_days_ago)).limit(500).stream()
            
            deleted_count = 0
            for doc in old_activities:
                doc.reference.delete()
                deleted_count += 1
                
            if deleted_count > 0:
                print(f"[Cleanup] Eliminados {deleted_count} registros de actividad antiguos.")
                
            conn = get_db_connection_raw()
            try:
                c = conn.cursor()
                c.execute("DELETE FROM search_logs WHERE timestamp < datetime('now', '-30 days')")
                deleted_logs = c.rowcount
                conn.commit()
            finally:
                conn.close()
            if deleted_logs > 0:
                print(f"[Cleanup] Eliminados {deleted_logs} logs de búsqueda antiguos.")
                
            print("[Auto-Learn] Iniciando auto-aprendizaje de sinónimos...")
            try:
                import asyncio
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.create_task(run_auto_learn_synonyms())
                else:
                    threading.Thread(target=run_auto_learn_synonyms_sync, daemon=True).start()
            except Exception as e:
                print(f"[Auto-Learn Error]: {e}")
                
        except Exception as e:
            print(f"[Cleanup Error]: Requiere índice o error general. {e}")
            
        time.sleep(86400)
