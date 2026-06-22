import re

file_path = 'main.py'
with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

setup_code = """import time
import struct
import google.generativeai as genai
from concurrent.futures import ThreadPoolExecutor

# Configurar Gemini
genai.configure(api_key=os.getenv("GOOGLE_API_KEY", ""))
EMBEDDING_MODEL = "models/embedding-001"
vector_worker_pool = ThreadPoolExecutor(max_workers=3)

def generate_product_embedding(product_name, category, description):
    text = f"Categoría: {category}. Nombre: {product_name}. Descripción: {description}"
    for attempt in range(3):
        try:
            result = genai.embed_content(
                model=EMBEDDING_MODEL,
                content=text,
                task_type="retrieval_document"
            )
            import sqlite_vec
            return sqlite_vec.serialize_float32(result['embedding'])
        except Exception as e:
            err_msg = str(e)
            if '429' in err_msg or 'Quota' in err_msg or 'Timeout' in err_msg:
                if attempt < 2:
                    time.sleep(2 ** attempt)
                    continue
            print(f"Error generando embedding para producto (intento {attempt+1}): {e}")
            break
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
            print(f"Error guardando vector en DB: {e}")
"""

if 'ThreadPoolExecutor' not in content:
    content = content.replace('import os\n', 'import os\n' + setup_code + '\n', 1)

webhook_insert = """
    with sqlite_lock:
        conn = get_db_connection()
        c = conn.cursor()
        c.execute("DELETE FROM search_index WHERE id = ? AND type = 'product'", (payload.id,))
        c.execute(\"\"\"
            INSERT INTO search_index (id, type, storeId, name, category, description, price, icon, imageUrl, onSale, salePrice, likes, views, purchases)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        \"\"\", (
            payload.id, 'product', payload.storeId, 
            payload.name, payload.category, payload.description, 
            str(payload.price), payload.icon, payload.imageUrl,
            1 if payload.onSale else 0, payload.salePrice,
            payload.likes, payload.views, payload.purchases
        ))
        conn.commit()
        conn.close()
    vector_worker_pool.submit(async_index_product_vector, payload.id, payload.name, payload.category, payload.description)
    return {"status": "indexed", "id": payload.id}
"""
content = re.sub(
    r'with sqlite_lock:\s+conn = get_db_connection\(\)\s+c = conn\.cursor\(\).*?return \{"status": "indexed", "id": payload\.id\}',
    webhook_insert.strip(),
    content,
    flags=re.DOTALL
)

# Delta Sync Update
delta_str = """
                                    p_data.get('views', 0),
                                    p_data.get('purchases', 0)
                                ))
"""
delta_rep = delta_str + "                                vector_worker_pool.submit(async_index_product_vector, p_id, p_data.get('name', ''), p_data.get('category', ''), p_data.get('description', ''))\n"
if "vector_worker_pool.submit(async_index_product_vector, p_id" not in content:
    content = content.replace(delta_str, delta_rep)

# Full Sync Update
full_sync_str = """
                p_data.get('views', 0),
                p_data.get('purchases', 0)
            ))
            count += 1
"""
full_sync_rep = full_sync_str + "            vector_worker_pool.submit(async_index_product_vector, product.id, p_data.get('name', ''), p_data.get('category', ''), p_data.get('description', ''))\n"
if "vector_worker_pool.submit(async_index_product_vector, product.id" not in content:
    content = content.replace(full_sync_str, full_sync_rep)

with open(file_path, 'w', encoding='utf-8') as f:
    f.write(content)

print("Phase 2 applied successfully with ThreadPoolExecutor!")
