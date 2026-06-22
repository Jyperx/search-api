import os
import re

file_path = 'main.py'
with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

# Add numpy if not present
if 'import numpy as np' not in content:
    content = content.replace('import time\n', 'import time\nimport numpy as np\n', 1)

# Fix Gemini API key to read from VITE_GEMINI_API_KEY
content = content.replace('os.getenv("GOOGLE_API_KEY", "")', 'os.getenv("VITE_GEMINI_API_KEY", os.getenv("GOOGLE_API_KEY", ""))')

refactored_func = """
def calculate_user_vector(activity_docs, calculate_time_decay_func):
    \"\"\"
    Calcula el vector del usuario evitando N+1 queries.
    \"\"\"
    product_ids = []
    decay_weights = {}
    
    for doc in activity_docs:
        # Asumiendo que to_dict() es invocado antes o es un dict directo
        data = doc.to_dict() if hasattr(doc, 'to_dict') else doc
        p_id = data.get('productId')
        if p_id:
            weight = calculate_time_decay_func(data.get('timestamp'))
            if p_id not in decay_weights:
                product_ids.append(p_id)
            decay_weights[p_id] = decay_weights.get(p_id, 0.0) + weight
            
    if not product_ids:
        return None
        
    # N+1 FIX: Una sola consulta SQL
    conn = get_db_connection()
    c = conn.cursor()
    placeholders = ','.join(['?'] * len(product_ids))
    c.execute(f"SELECT product_id, embedding FROM product_vectors WHERE product_id IN ({placeholders})", tuple(product_ids))
    rows = c.fetchall()
    conn.close()
    
    # Diccionario en memoria {product_id: vector_numpy}
    vectors_map = {}
    for row in rows:
        if row['embedding']:
            vectors_map[row['product_id']] = np.frombuffer(row['embedding'], dtype=np.float32)
            
    user_vector = np.zeros(768, dtype=np.float32)
    total_weight = 0.0
    
    for p_id in product_ids:
        if p_id in vectors_map:
            vec = vectors_map[p_id]
            w = decay_weights[p_id]
            user_vector += (vec * w)
            total_weight += w
            
    if total_weight > 0:
        user_vector = user_vector / total_weight
        import sqlite_vec
        return sqlite_vec.serialize_float32(user_vector.tolist())
        
    return None
"""

if 'def calculate_user_vector' not in content:
    content += '\n' + refactored_func

with open(file_path, 'w', encoding='utf-8') as f:
    f.write(content)
print("Phase 3 script finished")
