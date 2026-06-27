import logging
import math
import random
import json
from collections import defaultdict
from datetime import datetime, timezone
import numpy as np
from fastapi import APIRouter
from pydantic import BaseModel
from typing import List, Optional

from core.database import get_db_connection
from data.clusters import MACRO_CLUSTERS_CACHE, TIME_RULES_CACHE
from services.recommender import get_or_calculate_user_vector, find_similar_users_products
from services.context_engine import (
    get_weather, compute_context_weights, build_context_vector,
    score_product, concept_distance,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/home", tags=["Home"])

# Títulos amigables para las filas ambientales (concepto -> opciones)
ENV_TITLES = {
    "ENV_CALOR": ["Refréscate", "Para este calorcito", "Algo bien frío"],
    "ENV_FRIO": ["Para el frío", "Algo calientico", "Entra en calor"],
    "ENV_MANANA": ["Buenos días", "Para empezar el día", "Desayuno a la vista"],
    "ENV_MEDIODIA": ["Hora de almorzar", "Para el almuerzo", "Llegó el hambre"],
    "ENV_NOCHE": ["Plan nocturno", "Para esta noche", "Antojo de noche"],
}

class HomeFeedRequest(BaseModel):
    activities: List[dict] = []
    lat: Optional[float] = None
    lng: Optional[float] = None
    override_hour: Optional[int] = None
    override_weather_temp: Optional[float] = None
    override_weather_code: Optional[int] = None


def build_cluster_fts_query(cluster_name: str, c_val: dict, include_cluster_name: bool = True) -> str:
    """Construye una query FTS5 a partir de un cluster (usado por el buscador)."""
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
    """Home Feed por Vector de Contexto: rankea todo el catálogo ponderando gusto + clima + hora."""
    feed_sections = []
    conn = get_db_connection()
    try:
        c = conn.cursor()

        now = datetime.now(timezone.utc)
        base_hour = (now.hour - 5) % 24  # Colombia approx
        current_hour = req.override_hour if req.override_hour is not None else base_hour

        # 1. Vector de usuario (gusto)
        user_vector = None
        user_vec_np = None
        if req.activities:
            user_vector = get_or_calculate_user_vector(uid, req.activities, current_hour)
            if user_vector:
                user_vec_np = np.frombuffer(user_vector, dtype=np.float32)

        # 2. Clima + pesos continuos + vector de contexto
        temp, code = get_weather(req.lat, req.lng, req.override_weather_temp, req.override_weather_code)
        weights = compute_context_weights(temp, code, current_hour)
        ctx = build_context_vector(user_vec_np, weights)

        global_seen_ids = set()

        def take_from_pool(candidates, n, store_cap=2):
            out = []
            store_counts = {}
            for row in candidates:
                rid = row["id"]
                sid = row.get("storeId", "")
                if rid in global_seen_ids:
                    continue
                if store_counts.get(sid, 0) >= store_cap:
                    continue
                row.pop("embedding", None)  # binario, no serializable a JSON
                out.append(row)
                global_seen_ids.add(rid)
                store_counts[sid] = store_counts.get(sid, 0) + 1
                if len(out) >= n:
                    break
            return out

        # 3. Pool maestro: KNN del catálogo completo contra el contexto
        pool = []
        if ctx:
            c.execute("""
                SELECT p.product_id, p.embedding, vec_distance_cosine(p.embedding, ?) AS distance,
                       s.id, s.type, s.storeId, s.name, s.category, s.description,
                       s.price, s.icon, s.imageUrl, s.onSale, s.salePrice, s.likes, s.views, s.purchases,
                       st.name as storeName
                FROM product_vectors p
                JOIN search_index s ON p.product_id = s.id AND s.type = 'product'
                LEFT JOIN (SELECT id, name, isOpen FROM search_index WHERE type='store') st ON st.id = s.storeId
                WHERE CAST(s.available AS INTEGER) = 1 AND CAST(st.isOpen AS INTEGER) = 1
                ORDER BY distance ASC
                LIMIT 200
            """, (ctx,))
            for raw in c.fetchall():
                row = dict(raw)
                row["final_score"] = score_product(row, row["distance"])
                pool.append(row)
        else:
            # Sin señal (usuario nuevo, sin clima/hora marcada) → popularidad
            c.execute("""
                SELECT s.id, s.type, s.storeId, s.name, s.category, s.description,
                       s.price, s.icon, s.imageUrl, s.onSale, s.salePrice, s.likes, s.views, s.purchases,
                       st.name as storeName
                FROM search_index s
                LEFT JOIN (SELECT id, name, isOpen FROM search_index WHERE type='store') st ON st.id = s.storeId
                WHERE s.type = 'product' AND CAST(s.available AS INTEGER) = 1 AND CAST(st.isOpen AS INTEGER) = 1
                ORDER BY CAST(s.purchases AS INTEGER) DESC, CAST(s.likes AS INTEGER) DESC
                LIMIT 200
            """)
            for raw in c.fetchall():
                row = dict(raw)
                row["distance"] = 1.0
                row["final_score"] = score_product(row, 1.0)
                pool.append(row)

        pool.sort(key=lambda x: x["final_score"], reverse=True)

        # 4a. "Para ti ahora" (featured) — top del pool
        featured = take_from_pool(pool, 6)
        if featured:
            personalized = user_vector is not None
            feed_sections.append({
                "id": "dyn_for_you",
                "type": "products",
                "title": "Para ti ahora" if personalized else "Lo mejor ahora",
                "subtitle": "Según tu gusto, la hora y el clima" if personalized else "Lo más popular para este momento",
                "items": featured,
                "isPersonalized": personalized,
                "layout": "featured",
            })

        # 4b. Filas ambientales (solo si el peso continuo es significativo)
        env_rows = 0
        for concept_id, w in sorted(weights.items(), key=lambda kv: kv[1], reverse=True):
            if w < 0.5 or concept_id not in ENV_TITLES or env_rows >= 2:
                continue
            ranked = sorted(
                [r for r in pool if r["id"] not in global_seen_ids],
                key=lambda r: concept_distance(r.get("embedding"), concept_id)
            )
            items = take_from_pool(ranked, 6)
            if len(items) >= 3:
                feed_sections.append({
                    "id": f"dyn_env_{concept_id}",
                    "type": "products",
                    "title": random.choice(ENV_TITLES[concept_id]),
                    "subtitle": "Ideal para este momento",
                    "items": items,
                    "layout": "scroll",
                })
                env_rows += 1

        # 4c. Collaborative filtering — "Otros como tú pidieron"
        if user_vector:
            try:
                collab = find_similar_users_products(uid, user_vector)
                collab = [it for it in collab if it.get("id") not in global_seen_ids]
                if len(collab) >= 2:
                    for it in collab:
                        global_seen_ids.add(it["id"])
                    collab_titles = [
                        "Otros como tú pidieron",
                        "Popular entre usuarios similares",
                        "Tendencia entre perfiles similares",
                    ]
                    feed_sections.append({
                        "id": "dyn_collab_filtering",
                        "type": "products",
                        "title": random.choice(collab_titles),
                        "subtitle": "Basado en usuarios con gustos similares",
                        "items": collab,
                        "layout": "scroll",
                    })
            except Exception as e:
                logger.error(f"[Collaborative Filtering] Error: {e}")

        # 4d. Filas por categoría (anclas como etiquetas/títulos)
        anchor_title_map = {}
        try:
            c.execute("SELECT allowed_categories, titles, title FROM anchor_metadata")
            for arow in c.fetchall():
                arow = dict(arow)
                try:
                    cats = json.loads(arow.get("allowed_categories") or "[]")
                except Exception:
                    cats = []
                try:
                    titles = json.loads(arow.get("titles") or "[]")
                except Exception:
                    titles = []
                if not titles and arow.get("title"):
                    titles = [arow["title"]]
                for cat in cats:
                    if cat:
                        anchor_title_map.setdefault(cat.lower(), titles)
        except Exception as e:
            logger.error(f"[Anchor Titles] Error: {e}")

        cat_groups = defaultdict(list)
        for r in pool:
            if r["id"] in global_seen_ids:
                continue
            cat = r.get("category") or "general"
            cat_groups[cat].append(r)

        ordered_cats = sorted(cat_groups.keys(), key=lambda cat: cat_groups[cat][0]["final_score"], reverse=True)
        for cat in ordered_cats:
            if len(feed_sections) >= 12:
                break
            items = take_from_pool(cat_groups[cat], 5)
            if len(items) >= 2:
                titles = anchor_title_map.get(cat.lower())
                title = random.choice(titles) if titles else cat
                feed_sections.append({
                    "id": f"dyn_cat_{str(cat).replace(' ', '_')}",
                    "type": "products",
                    "title": title,
                    "subtitle": "Basado en tus intereses",
                    "items": items,
                    "layout": "grid" if len(items) >= 4 else "scroll",
                })

        # 4e. Anti-Bubble: categoría no visitada por el usuario
        try:
            user_cats = {(act.get('category') or '').lower() for act in req.activities}
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
                exp_items = [dict(r) for r in c.fetchall()]
                filtered_exp = take_from_pool(exp_items, 5)
                if len(filtered_exp) >= 1:
                    feed_sections.append({
                        "id": f"dyn_antibubble_{exp_cat.replace(' ', '_')}",
                        "type": "products",
                        "title": f"¿Has probado {exp_cat}?",
                        "subtitle": "Descubre algo totalmente nuevo",
                        "items": filtered_exp,
                        "isExploratory": True,
                        "layout": "grid" if len(filtered_exp) >= 4 else "scroll",
                    })
        except Exception as e:
            logger.error(f"[Anti-Bubble] Error: {e}")

        # 4f. Tiendas recomendadas
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

            candidate_stores = []
            for raw_row in c.fetchall():
                row = dict(raw_row)
                likes_val = int(row["likes"] or 0)
                distance = row.get("distance", 1.0)
                affinity = max(0.0, 1.0 - distance)
                novelty = 0.2 if likes_val < 5 else 0.0
                row["final_score"] = affinity + (math.log1p(likes_val) / 20.0) + novelty + random.uniform(0.0, 0.15)
                candidate_stores.append(row)

            candidate_stores.sort(key=lambda x: x["final_score"], reverse=True)

            category_counts = {}
            recommended_stores = []
            for row in candidate_stores:
                cat = row["category"]
                if category_counts.get(cat, 0) < 3:
                    likes_val = int(row["likes"] or 0)
                    recommended_stores.append({
                        "id": row["store_id"],
                        "name": row["name"],
                        "category": cat,
                        "description": row["description"],
                        "imageUrl": row["imageUrl"],
                        "logoUrl": row["imageUrl"],
                        "likes": likes_val,
                        "time": "15-25 min",
                        "rating": round(min(5.0, 4.0 + (likes_val / 100)), 1),
                        "deliveryFee": 0,
                        "open": True,
                        "type": "store",
                    })
                    category_counts[cat] = category_counts.get(cat, 0) + 1

            if recommended_stores:
                insert_pos = min(2, len(feed_sections))
                feed_sections.insert(insert_pos, {
                    "id": "dyn_recommended_stores",
                    "type": "stores",
                    "title": "Puntos para ti",
                    "subtitle": "Basado en tus gustos",
                    "items": recommended_stores,
                })
        except Exception as e:
            logger.error(f"[Stores Vector] Error: {e}")

    except Exception as e:
        logger.error(f"[Home Feed] Unhandled error: {e}")
        return {"sections": []}
    finally:
        conn.close()

    return {"sections": feed_sections}
