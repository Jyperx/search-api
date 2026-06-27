import logging
import math
import random
import json
import time
import requests
from datetime import datetime, timezone
from fastapi import APIRouter
from pydantic import BaseModel
from typing import List, Optional

from core.database import get_db_connection
from data.clusters import MACRO_CLUSTERS_CACHE, TIME_RULES_CACHE
from services.recommender import get_or_calculate_user_vector
from services.weather import WEATHER_CACHE_STORE

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/home", tags=["Home"])

class HomeFeedRequest(BaseModel):
    activities: List[dict] = []
    lat: Optional[float] = None
    lng: Optional[float] = None

def build_cluster_fts_query(cluster_name: str, c_val: dict, include_cluster_name: bool = True) -> str:
    cluster_match = c_val.get("keywords", "")
    cluster_words = [w.strip() for w in cluster_match.split(" OR ") if w.strip()]
    
    if include_cluster_name and cluster_name not in cluster_words:
        cluster_words.append(cluster_name)
        
    parts = [f'"{w}"*' for w in cluster_words]
    if not parts:
        return ""
    base_fts = " OR ".join(parts)
    
    neg_keywords_str = c_val.get("negativeKeywords", "")
    neg_parts = []
    if neg_keywords_str:
        neg_words = [w.strip() for w in neg_keywords_str.split(" OR ") if w.strip()]
        if neg_words:
            neg_group = " OR ".join([f'"{w}"*' for w in neg_words])
            neg_parts.append(f"NOT ({neg_group})")
            
    store_cats_str = c_val.get("storeCategories", "")
    cat_parts = []
    if store_cats_str:
        cats = [cat.strip() for cat in store_cats_str.split(",") if cat.strip()]
        if cats:
            cat_terms = " OR ".join([f'"{cat}"*' for cat in cats])
            cat_parts.append(f"{{category}} : ({cat_terms})")
            
    fts_query_parts = []
    if cat_parts:
        fts_query_parts.append(f"({cat_parts[0]}) AND ({base_fts})")
    else:
        fts_query_parts.append(f"({base_fts})")
        
    if neg_parts:
        fts_query_parts.append(neg_parts[0])
        
    return " ".join(fts_query_parts)

@router.post("/{uid}")
def get_dynamic_home_feed(uid: str, req: HomeFeedRequest):
    """Devuelve el inicio completo (Home Feed) basado en el Motor Híbrido (KNN Vectorial + FTS5)."""
    feed_sections = []
    conn = get_db_connection()
    try:
        c = conn.cursor()
        
        now = datetime.now(timezone.utc)
        current_hour = (now.hour - 5) % 24 # Colombia approx
        
        # 1. Obtener o calcular vector de usuario (Fase 3 Cache)
        user_vector = None
        if req.activities:
            user_vector = get_or_calculate_user_vector(uid, req.activities, current_hour)
            
        global_seen_ids = set()
        
        # 2. Cruce 1: Encontrar el Ancla ganadora (Contexto)
        anchors = []
        if user_vector:
            try:
                c.execute("""
                    SELECT a.anchor_id, m.title, m.subtitle, m.allowed_categories, m.exclude_rules, m.titles, vec_distance_cosine(a.embedding, ?) AS distance
                    FROM anchor_vectors a
                    JOIN anchor_metadata m ON a.anchor_id = m.anchor_id
                    ORDER BY distance ASC
                    LIMIT 2
                """, (user_vector,))
                anchors = [dict(row) for row in c.fetchall()]
                
                # Inyección de Exploración
                c.execute("""
                    SELECT a.anchor_id, m.title, m.subtitle, m.allowed_categories, m.exclude_rules, m.titles
                    FROM anchor_vectors a
                    JOIN anchor_metadata m ON a.anchor_id = m.anchor_id
                    WHERE a.anchor_id NOT IN (?, ?)
                    ORDER BY RANDOM()
                    LIMIT 1
                """, (anchors[0]['anchor_id'] if len(anchors) > 0 else '', anchors[1]['anchor_id'] if len(anchors) > 1 else ''))
                
                random_anchor = c.fetchone()
                if random_anchor:
                    random_anchor = dict(random_anchor)
                    random_anchor["title"] = "Sal de la rutina"
                    random_anchor["subtitle"] = "Descubre algo nuevo hoy"
                    random_anchor["isExploratory"] = True
                    anchors.append(random_anchor)
                    
            except Exception as e:
                logger.error(f"[Cruce 1] Error en KNN Anclas: {e}")
        else:
            try:
                # Para usuarios nuevos, 2 anclas al azar
                c.execute("""
                    SELECT a.anchor_id, m.title, m.subtitle, m.allowed_categories, m.exclude_rules, m.titles
                    FROM anchor_vectors a
                    JOIN anchor_metadata m ON a.anchor_id = m.anchor_id
                    ORDER BY RANDOM()
                    LIMIT 2
                """)
                anchors = [dict(row) for row in c.fetchall()]
            except Exception as e:
                logger.error(f"[Cruce 1 Random] Error: {e}")
                
        # 3. Cruce 2: Buscar Productos para el Ancla Ganadora
        for anchor in anchors:
            try:
                c.execute("""
                    SELECT p.product_id, vec_distance_cosine(p.embedding, a.embedding) AS distance,
                           s.id, s.type, s.storeId, s.name, s.category, s.description,
                           s.price, s.icon, s.imageUrl, s.onSale, s.salePrice, s.likes, s.views, s.purchases,
                           st.name as storeName
                    FROM product_vectors p
                    JOIN anchor_vectors a ON a.anchor_id = ?
                    JOIN search_index s ON p.product_id = s.id AND s.type = 'product'
                    LEFT JOIN (SELECT id, name, isOpen FROM search_index WHERE type='store') st ON st.id = s.storeId
                    WHERE CAST(s.available AS INTEGER) = 1 AND CAST(st.isOpen AS INTEGER) = 1
                    ORDER BY distance ASC
                    LIMIT 40
                """, (anchor["anchor_id"],))
                
                raw_items = c.fetchall()
                
                allowed_categories = []
                if anchor.get("allowed_categories"):
                    try: allowed_categories = [cat.lower() for cat in json.loads(anchor["allowed_categories"])]
                    except: pass
                    
                exclude_rules = []
                if anchor.get("exclude_rules"):
                    try: exclude_rules = json.loads(anchor["exclude_rules"])
                    except: pass
                    
                candidate_items = []
                
                for raw_row in raw_items:
                    row = dict(raw_row)
                    if row["distance"] > 0.8:
                        continue
                    
                    cat = str(row.get("category", "")).lower()
                    if allowed_categories and cat not in allowed_categories:
                        continue
                        
                    name_cat = (str(row.get("name", "")) + " " + cat).lower()
                    is_excluded = any(rule and rule.lower() in name_cat for rule in exclude_rules)
                    if is_excluded: continue
                    
                    rid = row["id"]
                    if rid in global_seen_ids: continue
                    
                    affinity = max(0.0, 1.0 - row["distance"])
                    purchases = float(row.get("purchases") or 0)
                    likes = float(row.get("likes") or 0)
                    views = float(row.get("views") or 0)
                    
                    cr_boost = 0.0
                    if views >= 10:
                        cr = purchases / views
                        if cr > 0.1: cr_boost = cr * 2.0
                        
                    C, m = 20.0, 1.0
                    bayes_purchases = (views * (purchases / (views + 0.1)) + C * m) / (views + C)
                    
                    popularity = math.log1p(bayes_purchases * 5.0 + likes * 0.5) / 10.0
                    novelty = 0.2 if (purchases == 0 and views <= 15) else (-0.3 if purchases == 0 and views > 50 else 0.0)
                    sale_boost = 0.15 if str(row.get("onSale", "0")) == "1" else 0.0
                    random_noise = random.uniform(0.0, 0.1)
                    
                    row["final_score"] = (affinity * 0.6) + (popularity * 0.2) + cr_boost + (novelty * 0.1) + (sale_boost * 0.1) + random_noise
                    candidate_items.append(row)
                    
                candidate_items.sort(key=lambda x: x["final_score"], reverse=True)
                
                store_counts = {}
                filtered_items = []
                
                for row in candidate_items:
                    rid = row["id"]
                    sid = row["storeId"]
                    if store_counts.get(sid, 0) >= 4: continue
                    
                    filtered_items.append(row)
                    global_seen_ids.add(rid)
                    store_counts[sid] = store_counts.get(sid, 0) + 1
                    
                    if len(filtered_items) >= 5:
                        break
                        
                if len(filtered_items) >= 1:
                    anchor_title = anchor.get("title", "Explorar")
                    if anchor.get("titles"):
                        try:
                            titles_list = json.loads(anchor["titles"])
                            if titles_list:
                                anchor_title = random.choice(titles_list)
                        except: pass
                        
                    feed_sections.append({
                        "id": f"dyn_vector_{anchor['anchor_id']}",
                        "type": "products",
                        "title": anchor_title,
                        "subtitle": anchor["subtitle"],
                        "items": filtered_items,
                        "isExploratory": anchor.get("isExploratory", False)
                    })
            except Exception as e:
                logger.error(f"[Cruce 2] Error obteniendo productos para ancla {anchor['anchor_id']}: {e}")
                
        # 4. Fallback Léxico (FTS5) - MACRO_CLUSTERS_CACHE
        cluster_scores = {k: 0.0 for k in MACRO_CLUSTERS_CACHE.keys()}
        
        for act in req.activities:
            cat = (act.get('category') or '').lower()
            score = 2.0 if act.get('type') == 'search' else 1.0
            for c_key, c_val in MACRO_CLUSTERS_CACHE.items():
                if cat in c_val['keywords'].lower() or cat == c_key:
                    cluster_scores[c_key] += score
                    
        for rule in TIME_RULES_CACHE:
            sh, eh = int(rule.get("startHour", 0)), int(rule.get("endHour", 23))
            rule_cluster, boost = rule.get("cluster", ""), float(rule.get("scoreBoost", 0))
            if rule_cluster in cluster_scores:
                if sh <= eh and sh <= current_hour <= eh:
                    cluster_scores[rule_cluster] += boost
                elif sh > eh and (current_hour >= sh or current_hour <= eh):
                    cluster_scores[rule_cluster] += boost
                    
        # 4.1. Reglas Ambientales (Clima Open-Meteo con Caché)
        if req.lat is not None and req.lng is not None:
            try:
                lat_key, lng_key = round(req.lat, 1), round(req.lng, 1)
                loc_key = f"{lat_key}_{lng_key}"
                now_ts = time.time()
                
                if loc_key in WEATHER_CACHE_STORE and (now_ts - WEATHER_CACHE_STORE[loc_key]["time"] < 3600):
                    temp = WEATHER_CACHE_STORE[loc_key]["temp"]
                    code = WEATHER_CACHE_STORE[loc_key]["code"]
                else:
                    w_res = requests.get(f"https://api.open-meteo.com/v1/forecast?latitude={lat_key}&longitude={lng_key}&current_weather=true", timeout=2).json()
                    if "current_weather" in w_res:
                        temp = w_res["current_weather"].get("temperature", 20)
                        code = w_res["current_weather"].get("weathercode", 0)
                        WEATHER_CACHE_STORE[loc_key] = {"temp": temp, "code": code, "time": now_ts}
                    else:
                        temp, code = 20, 0
                        
                is_night = current_hour < 6 or current_hour >= 18
                env_anchor_id = None
                if temp >= 24:
                    key = "calor_noche" if is_night else "calor_dia"
                    env_anchor_id = "ENV_CALOR_NOCHE" if is_night else "ENV_CALOR_DIA"
                    cluster_scores[key] = cluster_scores.get(key, 0) + 15.0
                elif temp <= 16 or code >= 50:
                    key = "frio_noche" if is_night else "frio_dia"
                    env_anchor_id = "ENV_FRIO_NOCHE" if is_night else "ENV_FRIO_DIA"
                    cluster_scores[key] = cluster_scores.get(key, 0) + 15.0
                    
                if env_anchor_id:
                    c.execute("""
                        SELECT p.product_id, vec_distance_cosine(p.embedding, a.embedding) AS distance,
                               s.id, s.type, s.storeId, s.name, s.category, s.description,
                               s.price, s.icon, s.imageUrl, s.onSale, s.salePrice, s.likes, s.views, s.purchases,
                               st.name as storeName, a_meta.title, a_meta.subtitle
                        FROM product_vectors p
                        JOIN anchor_vectors a ON a.anchor_id = ?
                        JOIN anchor_metadata a_meta ON a_meta.anchor_id = a.anchor_id
                        JOIN search_index s ON p.product_id = s.id AND s.type = 'product'
                        LEFT JOIN (SELECT id, name, isOpen FROM search_index WHERE type='store') st ON st.id = s.storeId
                        WHERE CAST(s.available AS INTEGER) = 1 AND CAST(st.isOpen AS INTEGER) = 1
                        ORDER BY distance ASC
                        LIMIT 20
                    """, (env_anchor_id,))
                    
                    env_items = c.fetchall()
                    filtered_env = []
                    store_counts = {}
                    env_title = "Para ti"
                    env_subtitle = ""
                    
                    env_candidates = []
                    for raw_row in env_items:
                        row = dict(raw_row)
                        if row["distance"] > 0.8: continue
                        rid, sid = row["id"], row["storeId"]
                        if rid in global_seen_ids: continue
                        
                        env_title = row.get("title", env_title)
                        env_subtitle = row.get("subtitle", env_subtitle)
                        
                        purchases = float(row.get("purchases") or 0)
                        views = float(row.get("views") or 0)
                        likes = float(row.get("likes") or 0)
                        
                        cr_boost = 0.0
                        if views >= 10:
                            cr = purchases / views
                            if cr > 0.1: cr_boost = cr * 2.0
                            
                        C, m = 20.0, 1.0
                        bayes_purchases = (views * (purchases / (views + 0.1)) + C * m) / (views + C)
                        
                        popularity = math.log1p(bayes_purchases * 5.0 + likes * 0.5) / 10.0
                        affinity = max(0.0, 1.0 - row["distance"])
                        
                        row["final_score"] = (affinity * 0.6) + (popularity * 0.2) + cr_boost
                        env_candidates.append(row)
                        
                    env_candidates.sort(key=lambda x: x["final_score"], reverse=True)
                    
                    for row in env_candidates:
                        rid, sid = row["id"], row["storeId"]
                        if store_counts.get(sid, 0) >= 3: continue
                        filtered_env.append(row)
                        global_seen_ids.add(rid)
                        store_counts[sid] = store_counts.get(sid, 0) + 1
                        if len(filtered_env) >= 6: break
                        
                    if len(filtered_env) >= 1:
                        feed_sections.insert(0, {
                            "id": f"dyn_env_{env_anchor_id}",
                            "type": "products",
                            "title": env_title,
                            "subtitle": env_subtitle,
                            "items": filtered_env
                        })
            except Exception as e:
                logger.error(f"[Weather/Env Vector] Error: {e}")
                    
        sorted_clusters = sorted([k for k, v in cluster_scores.items() if v > 0], key=lambda k: cluster_scores[k], reverse=True)
        top_clusters = sorted_clusters[:2]
        selected_clusters = top_clusters.copy()
        
        # 5. Anti-Bubble: Exploración Estricta de Categorías no visitadas
        try:
            user_cats = { (act.get('category') or '').lower() for act in req.activities }
            c.execute("SELECT DISTINCT category FROM search_index WHERE type='product' AND CAST(available AS INTEGER) = 1")
            all_cats = [row["category"] for row in c.fetchall() if row["category"]]
            unseen_cats = [cat for cat in all_cats if cat.lower() not in user_cats and cat.lower() != 'general']
            
            if unseen_cats:
                exp_cat = random.choice(unseen_cats)
                c.execute("""
                    SELECT p.id, p.type, p.storeId, p.name, p.category, p.description,
                           p.price, p.icon, p.imageUrl, p.onSale, p.salePrice, p.likes, p.views, p.purchases,
                           s.name as storeName
                    FROM search_index p
                    LEFT JOIN (SELECT id, name FROM search_index WHERE type='store') s ON s.id = p.storeId
                    WHERE p.type = 'product' AND p.category = ? AND CAST(p.available AS INTEGER) = 1
                    ORDER BY RANDOM()
                    LIMIT 15
                """, (exp_cat,))
                
                exp_items = c.fetchall()
                if len(exp_items) >= 1:
                    filtered_exp = []
                    store_counts = {}
                    for raw_row in exp_items:
                        row = dict(raw_row)
                        rid, sid = row["id"], row["storeId"]
                        if rid in global_seen_ids or store_counts.get(sid, 0) >= 4: continue
                        filtered_exp.append(row)
                        global_seen_ids.add(rid)
                        store_counts[sid] = store_counts.get(sid, 0) + 1
                        if len(filtered_exp) >= 5: break
                    
                    if len(filtered_exp) >= 1:
                        feed_sections.append({
                            "id": f"dyn_antibubble_{exp_cat.replace(' ', '_')}",
                            "type": "products",
                            "title": f"¿Has probado {exp_cat}?",
                            "subtitle": "Descubre algo totalmente nuevo",
                            "items": filtered_exp
                        })
        except Exception as e:
            logger.error(f"[Anti-Bubble] Error: {e}")
            
        if not selected_clusters:
            selected_clusters = ["comida_rapida", "mercado", random.choice(list(MACRO_CLUSTERS_CACHE.keys()))]
            
        for cluster in selected_clusters:
            if cluster not in MACRO_CLUSTERS_CACHE: continue
            fts_query = build_cluster_fts_query(cluster, MACRO_CLUSTERS_CACHE[cluster], True)
            if not fts_query: continue
            
            title = random.choice(MACRO_CLUSTERS_CACHE[cluster].get("titles", ["Para ti"]))
            subtitle = "Basado en tus intereses"
            
            try:
                c.execute("""
                    SELECT p.id, p.type, p.storeId, p.name, p.category, p.description,
                           p.price, p.icon, p.imageUrl, p.onSale, p.salePrice, p.likes, p.views, p.purchases,
                           s.name as storeName
                    FROM search_index p
                    LEFT JOIN (SELECT id, name FROM search_index WHERE type='store') s ON s.id = p.storeId
                    WHERE p.type = 'product' AND search_index MATCH ?
                    ORDER BY RANDOM()
                    LIMIT 20
                """, (fts_query,))
                
                raw_items = c.fetchall()
                candidate_items = []
                
                for raw_row in raw_items:
                    row = dict(raw_row)
                    rid = row["id"]
                    if rid in global_seen_ids: continue
                    
                    purchases = float(row.get("purchases") or 0)
                    likes = float(row.get("likes") or 0)
                    views = float(row.get("views") or 0)
                    
                    popularity = math.log1p(purchases + likes * 0.5) / 10.0
                    novelty = 0.2 if (purchases == 0 and views <= 15) else (-0.3 if purchases == 0 and views > 50 else 0.0)
                    sale_boost = 0.15 if str(row.get("onSale", "0")) == "1" else 0.0
                    random_noise = (abs(hash(rid)) % 100) / 1000.0
                    
                    row["final_score"] = popularity + novelty + sale_boost + random_noise
                    candidate_items.append(row)
                    
                candidate_items.sort(key=lambda x: x["final_score"], reverse=True)
                
                store_counts = {}
                filtered_items = []
                
                for row in candidate_items:
                    rid = row["id"]
                    sid = row["storeId"]
                    if store_counts.get(sid, 0) >= 4: continue
                    filtered_items.append(row)
                    global_seen_ids.add(rid)
                    store_counts[sid] = store_counts.get(sid, 0) + 1
                    if len(filtered_items) >= 5: break
                        
                if len(filtered_items) >= 1:
                    feed_sections.append({
                        "id": f"dyn_fts_{cluster}",
                        "type": "products",
                        "title": title,
                        "subtitle": subtitle,
                        "items": filtered_items
                    })
            except Exception as e:
                logger.error(f"[FTS Fallback] Error en cluster {cluster}: {e}")

        # 6. Tiendas Recomendadas
        try:
            if user_vector:
                c.execute("""
                    SELECT s.store_id, vec_distance_cosine(s.embedding, ?) AS distance,
                           st.name, st.category, st.description, st.imageUrl, st.likes, st.isOpen
                    FROM store_vectors s
                    JOIN search_index st ON st.id = s.store_id AND st.type = 'store'
                    WHERE CAST(st.isOpen AS INTEGER) = 1
                    ORDER BY distance ASC
                    LIMIT 200
                """, (user_vector,))
            else:
                c.execute("""
                    SELECT s.store_id, 1.0 AS distance,
                           st.name, st.category, st.description, st.imageUrl, st.likes, st.isOpen
                    FROM store_vectors s
                    JOIN search_index st ON st.id = s.store_id AND st.type = 'store'
                    WHERE CAST(st.isOpen AS INTEGER) = 1
                    ORDER BY CAST(st.likes AS INTEGER) DESC
                    LIMIT 200
                """)
                
            store_rows = c.fetchall()
            category_counts = {}
            recommended_stores = []
            
            candidate_stores = []
            for raw_row in store_rows:
                row = dict(raw_row)
                likes_val = int(row["likes"] or 0)
                distance = row.get("distance", 1.0)
                
                affinity = max(0.0, 1.0 - distance)
                novelty = 0.2 if likes_val < 5 else 0.0
                random_noise = random.uniform(0.0, 0.15)
                
                final_score = affinity + (math.log1p(likes_val) / 20.0) + novelty + random_noise
                row["final_score"] = final_score
                candidate_stores.append(row)
                
            candidate_stores.sort(key=lambda x: x["final_score"], reverse=True)
            
            for row in candidate_stores:
                cat = row["category"]
                if category_counts.get(cat, 0) < 3:
                    likes_val = int(row["likes"] or 0)
                    rating_val = round(min(5.0, 4.0 + (likes_val / 100)), 1)
                    recommended_stores.append({
                        "id": row["store_id"],
                        "name": row["name"],
                        "category": cat,
                        "description": row["description"],
                        "imageUrl": row["imageUrl"],
                        "logoUrl": row["imageUrl"],
                        "likes": likes_val,
                        "time": "15-25 min",
                        "rating": rating_val,
                        "deliveryFee": 0,
                        "open": True,
                        "type": "store"
                    })
                    category_counts[cat] = category_counts.get(cat, 0) + 1
                    
            if recommended_stores:
                insert_pos = min(2, len(feed_sections))
                feed_sections.insert(insert_pos, {
                    "id": "dyn_recommended_stores",
                    "type": "stores",
                    "title": "Puntos para ti",
                    "subtitle": "Basado en tus gustos",
                    "items": recommended_stores
                })
        except Exception as e:
            logger.error(f"[Stores Vector] Error: {e}")

    except Exception as e:
        logger.error(f"[Home Feed] Unhandled error: {e}")
        return {"sections": []}
    finally:
        conn.close()
        
    return {"sections": feed_sections}
