import numpy as np
import sqlite_vec
from core.config import REVERSE_SYNONYMS, SYNONYMS
from services.embeddings import generate_product_embedding

def calculate_user_vector(conn, activity_docs, calculate_time_decay_func, current_hour=None):
    product_ids = []
    decay_weights = {}
    
    for doc in activity_docs:
        data = doc.to_dict() if hasattr(doc, 'to_dict') else doc
        p_id = data.get('productId')
        if not p_id:
            continue
        act_type = data.get('type', 'view')
        ts = data.get('timestamp')
        
        # Action Weighting Multiplier
        act_multiplier = 1.0
        if act_type == 'purchase': act_multiplier = 5.0
        elif act_type == 'cart': act_multiplier = 3.0
        elif act_type == 'search': act_multiplier = 2.0
        elif act_type == 'view' or act_type == 'click': act_multiplier = 1.0
        elif act_type == 'ignored': act_multiplier = -0.5
        
        # Circadian Boost (Memoria Horaria)
        if current_hour is not None and ts:
            try:
                from datetime import datetime, timezone
                act_dt = None
                if isinstance(ts, str):
                    act_dt = datetime.fromisoformat(ts.replace('Z', '+00:00'))
                elif isinstance(ts, (int, float)):
                    act_dt = datetime.fromtimestamp(ts/1000 if ts > 10000000000 else ts, tz=timezone.utc)
                    
                if act_dt:
                    act_hour = (act_dt.hour - 5) % 24
                    diff = abs(act_hour - current_hour)
                    if diff > 12: diff = 24 - diff
                    if diff <= 3:
                        act_multiplier *= 3.0
                    elif diff >= 8:
                        act_multiplier *= 0.3
            except: pass
        
        decay = calculate_time_decay_func(ts)
        weight = act_multiplier * decay
        if p_id not in decay_weights:
            product_ids.append(p_id)
            decay_weights[p_id] = 0.0
        decay_weights[p_id] += weight
        
    ignored_counts = {}
    positive_counts = {}
    category_ignored = {}
    category_positive = {}

    for doc in activity_docs:
        data = doc.to_dict() if hasattr(doc, 'to_dict') else doc
        p_id = data.get('productId')
        c_id = data.get('categoryId') or data.get('category')
        act_type = data.get('type', 'view')
        
        if p_id:
            if act_type == 'ignored':
                ignored_counts[p_id] = ignored_counts.get(p_id, 0) + 1
                if c_id: category_ignored[c_id] = category_ignored.get(c_id, 0) + 1
            else:
                positive_counts[p_id] = positive_counts.get(p_id, 0) + 1
                if c_id: category_positive[c_id] = category_positive.get(c_id, 0) + 1

    for p_id, w in list(decay_weights.items()):
        if w < 0:
            ignores = ignored_counts.get(p_id, 0)
            positives = positive_counts.get(p_id, 0)
            if ignores >= 3 and positives == 0:
                decay_weights[p_id] = w * 2.4
            elif ignores == 2 and positives == 0:
                decay_weights[p_id] = w * 1.6
            elif positives > 0:
                decay_weights[p_id] = w * 0.5
                
    if not product_ids:
        return None
        
    c = conn.cursor()
    placeholders = ','.join(['?'] * len(product_ids))
    c.execute(f"SELECT product_id, embedding FROM product_vectors WHERE product_id IN ({placeholders})", tuple(product_ids))
    rows = c.fetchall()
    
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
        
        for cat, ignores in category_ignored.items():
            pos = category_positive.get(cat, 0)
            if ignores >= 5 and pos == 0:
                try:
                    cat_vec = generate_product_embedding(cat, cat, "Categoría rechazada por el usuario")
                    if cat_vec:
                        user_vector -= (np.array(cat_vec, dtype=np.float32) * 0.3)
                except: pass
                
        norm = np.linalg.norm(user_vector)
        if norm > 0:
            user_vector = user_vector / norm
            
        return sqlite_vec.serialize_float32(user_vector.tolist())
    return None

def build_cluster_fts_query(cluster_name, c_val, include_cluster_name=True):
    cluster_match = c_val.get("keywords", "")
    cluster_words = [w.strip() for w in cluster_match.split(" OR ") if w.strip()]
    
    if include_cluster_name and cluster_name not in cluster_words:
        cluster_words.append(cluster_name)
        
    expanded_parts = []
    for w in cluster_words:
        if w in REVERSE_SYNONYMS:
            root = REVERSE_SYNONYMS[w]
            syns = SYNONYMS[root]
            group = " OR ".join([f'"{s}"*' for s in syns])
            expanded_parts.append(f"({group})")
        else:
            expanded_parts.append(f'"{w}"*')
            
    if not expanded_parts:
        return ""
    base_fts = " OR ".join(expanded_parts)
    return base_fts
