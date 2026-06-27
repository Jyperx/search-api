import os
import time
import google.generativeai as genai
import sqlite_vec
from concurrent.futures import ThreadPoolExecutor
from core.config import EMBEDDING_MODEL, global_sync_state
from database import get_db_connection_raw

# Configurar API de Gemini
genai.configure(api_key=os.getenv("VITE_GEMINI_API_KEY", os.getenv("GOOGLE_API_KEY", "")))

vector_worker_pool = ThreadPoolExecutor(max_workers=3)

def generate_product_embedding(name, category, description):
    text = f"Producto: {name}. Categoría: {category}. Descripción: {description}."
    for attempt in range(3):
        try:
            res = genai.embed_content(model=EMBEDDING_MODEL, content=text, task_type="retrieval_document")
            return sqlite_vec.serialize_float32(res['embedding'][:768])
        except Exception as e:
            print(f"Error generando embedding para {name} (Intento {attempt}): {e}")
            time.sleep(2 ** attempt)
    return None

def async_index_product_vector(p_id, name, category, description):
    global global_sync_state
    try:
        vector_bytes = generate_product_embedding(name, category, description)
        if vector_bytes:
            try:
                conn = get_db_connection_raw()
                c = conn.cursor()
                c.execute("DELETE FROM product_vectors WHERE product_id = ?", (p_id,))
                c.execute("INSERT INTO product_vectors (product_id, embedding) VALUES (?, ?)", (p_id, vector_bytes))
                conn.commit()
                conn.close()
            except Exception as e:
                print(f"Error guardando vector: {e}")
    finally:
        global_sync_state["completed_products"] += 1
        if global_sync_state["completed_products"] >= global_sync_state["total_products"] and global_sync_state["total_products"] > 0:
            global_sync_state["is_syncing"] = False
            global_sync_state["status"] = "Completado"

def async_index_store_vector(s_id, name, category, description, products_summary):
    global global_sync_state
    try:
        text = f"Comercio: {name}. Categoría: {category}. Descripción: {description}. Productos principales que vende: {products_summary}."
        vector_bytes = None
        for attempt in range(3):
            try:
                res = genai.embed_content(model=EMBEDDING_MODEL, content=text, task_type="retrieval_document")
                vector_bytes = sqlite_vec.serialize_float32(res['embedding'][:768])
                break
            except Exception as e:
                print(f"Error generando embedding para store {name} (Intento {attempt}): {e}")
                time.sleep(2 ** attempt)
                
        if vector_bytes:
            try:
                conn = get_db_connection_raw()
                c = conn.cursor()
                c.execute("DELETE FROM store_vectors WHERE store_id = ?", (s_id,))
                c.execute("INSERT INTO store_vectors (store_id, embedding) VALUES (?, ?)", (s_id, vector_bytes))
                conn.commit()
                conn.close()
            except Exception as e:
                print(f"Error guardando vector de tienda: {e}")
    finally:
        # Los comercios también suman al progreso
        global_sync_state["completed_products"] += 1
        if global_sync_state["completed_products"] >= global_sync_state["total_products"] and global_sync_state["total_products"] > 0:
            global_sync_state["is_syncing"] = False
            global_sync_state["status"] = "Completado"
