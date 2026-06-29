import logging
import asyncio
import threading
import numpy as np
import sqlite_vec
from concurrent.futures import ThreadPoolExecutor
from core.database import get_db_connection, sqlite_lock
from core.config import global_sync_state
import core.firebase
from core.genai_client import embed_text
from services.embeddings import generate_product_embedding, generate_product_embeddings_batch

logger = logging.getLogger(__name__)

_progress_lock = threading.Lock()


def _report_vector_progress():
    """Avanza el contador de la barra de sync (thread-safe). Apaga la barra al terminar."""
    if not global_sync_state.get("is_syncing"):
        return
    with _progress_lock:
        done = global_sync_state.get("completed_products", 0) + 1
        total = global_sync_state.get("total_products", 0)
        global_sync_state["completed_products"] = done
        if total > 0 and done >= total:
            global_sync_state["is_syncing"] = False
            global_sync_state["status"] = "Completado"
        else:
            global_sync_state["status"] = f"Vectorizando {done}/{total}..."

vector_worker_pool = ThreadPoolExecutor(max_workers=6)


def index_store_location(store_id: str, location):
    """Guarda la ubicación (lat/lng) de un comercio para el ranking por cercanía."""
    if not location or not isinstance(location, dict):
        return
    lat = location.get("latitude", location.get("lat"))
    lng = location.get("longitude", location.get("lng"))
    if lat is None or lng is None:
        return
    try:
        with sqlite_lock:
            conn = get_db_connection()
            conn.execute(
                "INSERT OR REPLACE INTO store_locations (store_id, lat, lng) VALUES (?, ?, ?)",
                (store_id, float(lat), float(lng))
            )
            conn.commit()
            conn.close()
    except Exception as e:
        logger.error(f"Error guardando ubicación de {store_id}: {e}")


def index_store_vector(store_id: str, name: str, category: str, description: str = ""):
    """Genera y guarda el embedding de un comercio (para 'Puntos para ti' y búsqueda de tiendas)."""
    try:
        text = f"Comercio: {name}. Categoría: {category}. {description or ''}".strip()
        emb = embed_text(text, task_type="retrieval_document")
        v = np.array(emb, dtype=np.float32)
        n = np.linalg.norm(v)
        if n > 0:
            v = v / n
        blob = sqlite_vec.serialize_float32(v.tolist())
        with sqlite_lock:
            conn = get_db_connection()
            # Las tablas vec0 no respetan INSERT OR REPLACE → borrar e insertar
            conn.execute("DELETE FROM store_vectors WHERE store_id = ?", (store_id,))
            conn.execute("INSERT INTO store_vectors (store_id, embedding) VALUES (?, ?)", (store_id, blob))
            conn.commit()
            conn.close()
    except Exception as e:
        logger.error(f"Error indexing store vector {store_id}: {e}")

def async_index_product_vector(product_id: str, name: str, category: str, description: str):
    """Worker sincrónico corriendo en pool"""
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        vector_bytes, source_hint = loop.run_until_complete(
            generate_product_embedding(name, category, description)
        )
        loop.close()

        with sqlite_lock:
            conn = get_db_connection()
            if vector_bytes:
                # Las tablas vec0 no respetan INSERT OR REPLACE → borrar e insertar
                conn.execute("DELETE FROM product_vectors WHERE product_id = ?", (product_id,))
                conn.execute(
                    "INSERT INTO product_vectors (product_id, embedding) VALUES (?, ?)",
                    (product_id, vector_bytes)
                )
                # Borrar de la cola si tuvo éxito
                conn.execute("DELETE FROM vector_queue WHERE product_id = ?", (product_id,))
            else:
                # FIX B3: Cola de reintentos
                conn.execute(
                    "INSERT OR REPLACE INTO vector_queue (product_id, name, category, description, attempts, last_attempt, source_hint) "
                    "VALUES (?, ?, ?, ?, ?, datetime('now'), ?)",
                    (product_id, name, category, description, 1, source_hint)
                )
            conn.commit()
            conn.close()
    except Exception as e:
        logger.error(f"Error in async vector indexing for {product_id}: {e}")
    finally:
        _report_vector_progress()  # garantiza que la barra avance aunque falle el embedding


def index_products_batch(items):
    """Worker de LOTE: vectoriza una tanda de productos con una sola llamada a Gemini y los guarda.
    items = [(product_id, name, category, description), ...].
    Lo que falla cae a vector_queue y lo reintenta el path normal (con enriquecimiento completo)."""
    try:
        # El embed (lento) ocurre FUERA del lock; los inserts (rápidos) van dentro.
        results = generate_product_embeddings_batch(items)
        with sqlite_lock:
            conn = get_db_connection()
            for (pid, blob, source), item in zip(results, items):
                try:
                    if blob:
                        conn.execute("DELETE FROM product_vectors WHERE product_id = ?", (pid,))
                        conn.execute("INSERT INTO product_vectors (product_id, embedding) VALUES (?, ?)", (pid, blob))
                        conn.execute("DELETE FROM vector_queue WHERE product_id = ?", (pid,))
                    else:
                        conn.execute(
                            "INSERT OR REPLACE INTO vector_queue (product_id, name, category, description, attempts, last_attempt, source_hint) "
                            "VALUES (?, ?, ?, ?, ?, datetime('now'), ?)",
                            (pid, item[1], item[2], item[3], 1, source)
                        )
                except Exception as e:
                    logger.error(f"[Batch] Error guardando {pid}: {e}")
            conn.commit()
            conn.close()
    except Exception as e:
        logger.error(f"[Batch] Error en el lote de {len(items)} productos: {e}")
    finally:
        for _ in items:
            _report_vector_progress()  # avanza la barra por cada producto del lote


def do_sync_database():
    """Sincronización masiva de Firestore a SQLite (FTS5 + Vectores)"""
    global vector_worker_pool

    # OJO: el endpoint /api/sync ya pone is_syncing=True para feedback inmediato.
    # Aquí solo abortamos si no hay Firebase (antes había un deadlock con is_syncing).
    if not core.firebase.db:
        global_sync_state["is_syncing"] = False
        global_sync_state["status"] = "error: sin Firebase"
        return

    print("[Sync] Iniciando sincronización desde Firestore...")
    global_sync_state["is_syncing"] = True
    global_sync_state["status"] = "Indexando comercios y productos..."
    global_sync_state["total_products"] = 0
    global_sync_state["completed_products"] = 0
    embed_args = []

    # FIX B2: Cancelar workers anteriores
    vector_worker_pool.shutdown(wait=False, cancel_futures=True)
    vector_worker_pool = ThreadPoolExecutor(max_workers=6)
    
    # FIX B2: Swap atómico con tabla temporal
    with sqlite_lock:
        conn = get_db_connection()
        conn.execute("DROP TABLE IF EXISTS search_index_new")
        conn.execute("""
            CREATE VIRTUAL TABLE search_index_new USING fts5(
                id, type, storeId, name, category, description, price,
                icon, imageUrl UNINDEXED, onSale UNINDEXED, salePrice UNINDEXED,
                likes UNINDEXED, views UNINDEXED, purchases UNINDEXED,
                available UNINDEXED, isOpen UNINDEXED
            )
        """)
        conn.commit()
        conn.close()
        
    try:
        stores = core.firebase.db.collection('stores').stream()
        for store in stores:
            s_data = store.to_dict()
            s_id = store.id
            s_name = s_data.get('name', '')
            s_cat = s_data.get('category', '')
            s_desc = s_data.get('description', '')
            s_img = s_data.get('logoUrl') or s_data.get('image') or ''  # el comercio usa logoUrl
            is_open = int(bool(s_data.get('isOpen', True))) # FIX B1
            
            products = list(core.firebase.db.collection('stores').document(s_id).collection('products').stream())
            
            with sqlite_lock:
                conn = get_db_connection()
                # Insert store
                conn.execute("""
                    INSERT INTO search_index_new
                        (id, type, storeId, name, category, description,
                         price, icon, imageUrl, onSale, salePrice,
                         likes, views, purchases, available, isOpen)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    s_id, 'store', s_id, s_name, s_cat, s_desc,
                    '0', '', s_img, 0, '', 0, 0, 0, 1, is_open
                ))
                
                # Insert products
                for p in products:
                    p_data = p.to_dict()
                    p_id = p.id
                    p_name = p_data.get('name', '')
                    p_cat = p_data.get('category', s_cat)
                    p_desc = p_data.get('description', '')
                    p_avail = int(bool(p_data.get('available', True))) # FIX B1
                    
                    conn.execute("""
                        INSERT INTO search_index_new
                            (id, type, storeId, name, category, description,
                             price, icon, imageUrl, onSale, salePrice,
                             likes, views, purchases, available, isOpen)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        p_id, 'product', s_id, p_name, p_cat, p_desc,
                        str(p_data.get('price', '')), p_data.get('icon', ''), p_data.get('imageUrl', ''),
                        1 if p_data.get('onSale') else 0,
                        str(p_data.get('salePrice', '')),
                        p_data.get('likes', 0),
                        p_data.get('views', 0),
                        p_data.get('purchases', 0),
                        p_avail, is_open
                    ))
                conn.commit()
                conn.close()

            # Vectorizar el comercio (en background) + guardar su ubicación
            vector_worker_pool.submit(
                index_store_vector, s_id, s_name, s_cat, s_data.get('description', '')
            )
            index_store_location(s_id, s_data.get('location'))

            # Recolectar productos a vectorizar (se lanzan DESPUÉS de fijar el total, sin carrera)
            for p in products:
                p_data = p.to_dict()
                if int(bool(p_data.get('available', True))):
                    embed_args.append((p.id, p_data.get('name', ''), p_data.get('category', s_cat), p_data.get('description', '')))

        # FIX B2: Aplicar Swap atómico
        with sqlite_lock:
            conn = get_db_connection()
            conn.execute("DROP TABLE IF EXISTS search_index_old")
            # Si no existe, RENAME fallará, manejar con TRY/CATCH
            try:
                conn.execute("ALTER TABLE search_index RENAME TO search_index_old")
            except: pass
            
            conn.execute("ALTER TABLE search_index_new RENAME TO search_index")
            conn.execute("DROP TABLE IF EXISTS search_index_old")
            conn.commit()
            conn.close()

        # Fijar el total ANTES de lanzar las vectorizaciones (los workers reportan progreso)
        total = len(embed_args)
        print(f"[Sync] Índice reconstruido. Vectorizando {total} productos en segundo plano...")
        global_sync_state["total_products"] = total
        global_sync_state["completed_products"] = 0
        if total == 0:
            global_sync_state["is_syncing"] = False
            global_sync_state["status"] = "Sin productos para vectorizar"
        else:
            global_sync_state["status"] = f"Vectorizando 0/{total}..."
            # Vectorización en LOTES: cada tanda = 1 sola llamada a Gemini (mucho más rápido).
            # Los lotes se reparten entre los workers del pool.
            BATCH_SIZE = 50
            for i in range(0, total, BATCH_SIZE):
                chunk = embed_args[i:i + BATCH_SIZE]
                vector_worker_pool.submit(index_products_batch, chunk)
            # is_syncing queda True; los workers la apagan al llegar a total

    except Exception as e:
        logger.error(f"Sync failed: {e}")
        global_sync_state["is_syncing"] = False
        global_sync_state["status"] = f"error: {e}"

def reconcile_catalog():
    """Quita del índice los productos/comercios que ya NO existen en Firestore (anti-fantasmas).
    No re-vectoriza (barato): solo elimina lo que sobra. Las altas las cubren los webhooks/sync."""
    if not core.firebase.db:
        return
    try:
        valid_stores = set()
        valid_products = set()
        for store in core.firebase.db.collection('stores').stream():
            valid_stores.add(store.id)
            for p in core.firebase.db.collection('stores').document(store.id).collection('products').stream():
                valid_products.add(p.id)

        # Si Firestore vino vacío, no borramos nada (evita vaciar por un fallo de lectura)
        if not valid_stores:
            return

        removed = 0
        with sqlite_lock:
            conn = get_db_connection()
            rows = conn.execute("SELECT id, type FROM search_index").fetchall()
            for row in rows:
                rid, rtype = row["id"], row["type"]
                if rtype == 'product' and rid not in valid_products:
                    conn.execute("DELETE FROM search_index WHERE id=? AND type='product'", (rid,))
                    conn.execute("DELETE FROM product_vectors WHERE product_id=?", (rid,))
                    conn.execute("DELETE FROM item_stats WHERE product_id=?", (rid,))
                    removed += 1
                elif rtype == 'store' and rid not in valid_stores:
                    conn.execute("DELETE FROM search_index WHERE id=? AND type='store'", (rid,))
                    conn.execute("DELETE FROM store_vectors WHERE store_id=?", (rid,))
                    conn.execute("DELETE FROM store_locations WHERE store_id=?", (rid,))
                    removed += 1
            conn.commit()
            conn.close()
        if removed:
            print(f"[Reconcile] {removed} fantasmas eliminados del índice.")
    except Exception as e:
        logger.error(f"[Reconcile] Error: {e}")


def retry_vector_queue_task():
    """Función para el scheduler de APScheduler para reintentos"""
    rows_to_retry = []
    with sqlite_lock:
        conn = get_db_connection()
        rows = conn.execute(
            "SELECT * FROM vector_queue WHERE attempts < 5 ORDER BY attempts ASC LIMIT 20"
        ).fetchall()
        
        for row in rows:
            conn.execute(
                "UPDATE vector_queue SET attempts = attempts + 1, last_attempt = datetime('now') WHERE product_id = ?",
                (row['product_id'],)
            )
            rows_to_retry.append(dict(row))
        conn.commit()
        conn.close()
        
    for row in rows_to_retry:
        vector_worker_pool.submit(
            async_index_product_vector, 
            row['product_id'], 
            row['name'], 
            row['category'], 
            row['description']
        )

def do_sync_store(store_id: str):
    """Re-sincroniza un solo comercio: refresca su fila + productos en FTS y los vectoriza."""
    if not core.firebase.db:
        return
    try:
        doc = core.firebase.db.collection('stores').document(store_id).get()
        if not doc.exists:
            return
        s_data = doc.to_dict()
        s_name = s_data.get('name', '')
        s_cat = s_data.get('category', '')
        s_desc = s_data.get('description', '')
        s_img = s_data.get('logoUrl') or s_data.get('image') or ''
        is_open = int(bool(s_data.get('isOpen', True)))
        products = list(core.firebase.db.collection('stores').document(store_id).collection('products').stream())

        with sqlite_lock:
            conn = get_db_connection()
            # Borrar comercio + sus productos del índice
            conn.execute("DELETE FROM search_index WHERE id = ? OR storeId = ?", (store_id, store_id))
            conn.execute("""
                INSERT INTO search_index
                    (id, type, storeId, name, category, description, price, icon, imageUrl,
                     onSale, salePrice, likes, views, purchases, available, isOpen)
                VALUES (?, 'store', ?, ?, ?, ?, '0', '', ?, 0, '', 0, 0, 0, 1, ?)
            """, (store_id, store_id, s_name, s_cat, s_desc, s_img, is_open))
            for p in products:
                p_data = p.to_dict()
                conn.execute("""
                    INSERT INTO search_index
                        (id, type, storeId, name, category, description, price, icon, imageUrl,
                         onSale, salePrice, likes, views, purchases, available, isOpen)
                    VALUES (?, 'product', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    p.id, store_id, p_data.get('name', ''), p_data.get('category', s_cat),
                    p_data.get('description', ''), str(p_data.get('price', '')),
                    p_data.get('icon', ''), p_data.get('imageUrl', ''),
                    1 if p_data.get('onSale') else 0, str(p_data.get('salePrice', '')),
                    p_data.get('likes', 0), p_data.get('views', 0), p_data.get('purchases', 0),
                    int(bool(p_data.get('available', True))), is_open
                ))
            conn.commit()
            conn.close()

        # Vectorizar comercio + productos + ubicación
        vector_worker_pool.submit(index_store_vector, store_id, s_name, s_cat, s_data.get('description', ''))
        index_store_location(store_id, s_data.get('location'))
        for p in products:
            p_data = p.to_dict()
            if int(bool(p_data.get('available', True))):
                vector_worker_pool.submit(
                    async_index_product_vector, p.id, p_data.get('name', ''),
                    p_data.get('category', s_cat), p_data.get('description', '')
                )
    except Exception as e:
        logger.error(f"Error en do_sync_store {store_id}: {e}")

def do_seed_anchors():
    pass # To be implemented if needed
