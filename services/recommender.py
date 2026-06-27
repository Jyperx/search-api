import numpy as np
from datetime import datetime, timezone
import sqlite_vec
import logging
from core.database import get_db_connection, sqlite_lock
from data.clusters import TIME_RULES_CACHE

logger = logging.getLogger(__name__)

MAX_BOOST = 2.0
MIN_SCORE = 0.0

def apply_time_boost(base_score: float, cluster: str, current_hour: int) -> float:
    """FIX B6 — Cap en time boost con porcentaje simétrico"""
    boost = 0.0
    for rule in TIME_RULES_CACHE:
        if rule.get('cluster') == cluster:
            start_hour = rule.get('startHour', 0)
            end_hour = rule.get('endHour', 23)
            # Manejar horas que cruzan la medianoche
            if start_hour <= end_hour:
                in_window = start_hour <= current_hour <= end_hour
            else:
                in_window = current_hour >= start_hour or current_hour <= end_hour
                
            if in_window:
                boost = rule.get('scoreBoost', 0.0)
                break
                
    # Cap simétrico
    boost = max(-MAX_BOOST, min(MAX_BOOST, boost))
    # Aplicar como porcentaje del score base, no suma absoluta
    return max(MIN_SCORE, base_score * (1 + boost * 0.15))

def calculate_time_decay_func(ts) -> float:
    if not ts: return 0.5
    try:
        if isinstance(ts, str):
            act_dt = datetime.fromisoformat(ts.replace('Z', '+00:00'))
        else:
            # Handle miliseconds timestamp
            act_dt = datetime.fromtimestamp(ts/1000 if ts > 10000000000 else ts, tz=timezone.utc)
            
        age_days = (datetime.now(timezone.utc) - act_dt).total_seconds() / 86400
        # Exponential decay: e^(-lambda * t), halflife of 7 days -> lambda ~ 0.1
        weight = max(0.1, np.exp(-0.1 * age_days))
        return float(weight)
    except:
        return 0.5

def calculate_user_vector(activity_docs: list, current_hour: int) -> bytes | None:
    product_ids = []
    decay_weights = {}
    
    for doc in activity_docs:
        data = doc.to_dict() if hasattr(doc, 'to_dict') else doc
        p_id = data.get('productId')
        act_type = data.get('type', 'view')
        ts = data.get('timestamp')
        
        act_multiplier = 1.0
        if act_type == 'purchase': act_multiplier = 5.0
        elif act_type == 'cart': act_multiplier = 3.0
        elif act_type == 'search': act_multiplier = 2.0
        elif act_type == 'view' or act_type == 'click': act_multiplier = 1.0
        elif act_type == 'ignored': act_multiplier = -0.5
        
        if current_hour is not None and ts:
            try:
                act_dt = None
                if isinstance(ts, str):
                    act_dt = datetime.fromisoformat(ts.replace('Z', '+00:00'))
                elif isinstance(ts, (int, float)):
                    act_dt = datetime.fromtimestamp(ts/1000 if ts > 10000000000 else ts, tz=timezone.utc)
                    
                if act_dt:
                    act_hour = (act_dt.hour - 5) % 24 # Colombia timezone aprox
                    diff = abs(act_hour - current_hour)
                    if diff > 12: diff = 24 - diff
                    if diff <= 3:
                        act_multiplier *= 3.0
                    elif diff >= 8:
                        act_multiplier *= 0.3
            except: pass
        
        if p_id:
            weight = calculate_time_decay_func(ts) * act_multiplier
            # TODO: Implementar lógica de aversión de categorías (restar si ignored > 5)
            if p_id not in decay_weights:
                product_ids.append(p_id)
            decay_weights[p_id] = decay_weights.get(p_id, 0.0) + weight
            
    if not product_ids:
        return None
        
    conn = get_db_connection()
    placeholders = ','.join(['?'] * len(product_ids))
    rows = conn.execute(f"SELECT product_id, embedding FROM product_vectors WHERE product_id IN ({placeholders})", tuple(product_ids)).fetchall()
    conn.close()
    
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
        return sqlite_vec.serialize_float32(user_vector.tolist())
        
    return None

def persist_user_vector(user_id: str, vector_bytes: bytes, event_count: int):
    try:
        with sqlite_lock:
            conn = get_db_connection()
            conn.execute("INSERT OR REPLACE INTO user_vectors (user_id, embedding) VALUES (?, ?)", (user_id, vector_bytes))
            conn.execute(
                "INSERT OR REPLACE INTO user_vector_meta (user_id, last_updated, event_count, source) VALUES (?, datetime('now'), ?, ?)",
                (user_id, event_count, 'calculated')
            )
            conn.commit()
            conn.close()
    except Exception as e:
        logger.error(f"Error persisting user vector for {user_id}: {e}")

def get_or_calculate_user_vector(user_id: str, activity_docs: list, current_hour: int) -> bytes | None:
    conn = get_db_connection()
    try:
        row = conn.execute("SELECT embedding FROM user_vectors WHERE user_id = ?", (user_id,)).fetchone()
        meta = conn.execute("SELECT last_updated, event_count FROM user_vector_meta WHERE user_id = ?", (user_id,)).fetchone()
    except Exception as e:
        logger.error(f"Error checking cache for {user_id}: {e}")
        row = None
        meta = None
    finally:
        conn.close()

    if row and meta:
        try:
            last_updated_str = meta['last_updated']
            if len(last_updated_str) == 19:
                last_updated = datetime.strptime(last_updated_str, "%Y-%m-%d %H:%M:%S")
            else:
                last_updated = datetime.fromisoformat(last_updated_str)
                
            age_hours = (datetime.now(timezone.utc) - last_updated.replace(tzinfo=timezone.utc)).total_seconds() / 3600
            
            # Si el caché tiene menos de 2 horas de antigüedad, retornar.
            if age_hours < 2:
                return row['embedding']
        except Exception as e:
            logger.warning(f"Error parsing date for user {user_id}: {e}")

    # No hay caché válido, calcular de nuevo
    vector_bytes = calculate_user_vector(activity_docs, current_hour)
    if vector_bytes:
        persist_user_vector(user_id, vector_bytes, len(activity_docs))
        
    return vector_bytes
