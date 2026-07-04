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

from core.database import get_db_connection, sqlite_lock
from data.clusters import MACRO_CLUSTERS_CACHE, TIME_RULES_CACHE
from services.recommender import get_or_calculate_user_vector, find_similar_users_products
from services.context_engine import (
    get_weather, compute_context_weights, build_context_vector,
    score_product, concept_distance, haversine_km, proximity_boost,
)
from data.curation import curation_action

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

# Subtítulos creativos para variar (en vez de repetir siempre "Basado en tus intereses")
SECTION_SUBTITLES = [
    "Elegidos para ti", "Porque te puede gustar", "Hecho a tu medida",
    "Lo que va contigo", "Seleccionados para ti", "Pensado en tus gustos",
    "Esto te va a encantar", "Va con tu estilo", "No te lo pierdas",
]

# Pools de títulos creativos por categoría (fallback cuando no hay título de admin).
# Evita mostrar el nombre plano de la categoría; rotan para sentirse frescos.
CATEGORY_TITLES = {
    "restaurante": ["Para sentarte a comer", "Sabores de verdad", "Como en casa", "Un buen plato"],
    "comidas rápidas": ["Antojo rápido", "Para hoy sin esperar", "Lo que se te antoja ya", "Rápido y rico"],
    "panadería": ["Recién horneado", "El olorcito a pan", "Para el cafecito", "Dulce y salado"],
    "mercados": ["Llena la nevera", "El mercado en casa", "Surte tu despensa", "Frescura a domicilio"],
    "spa & belleza": ["Consiéntete", "Tiempo para ti", "Date un gusto", "Belleza a un toque"],
    "farmacia": ["Tu botiquín en casa", "Salud al instante", "Lo que necesitas, ya", "Cuídate"],
    "barbería": ["Renueva tu look", "Hora de cortarse", "Estilo fresco", "Para verte bien"],
    "licores": ["Para la celebración", "Prende el plan", "Salud y buena vibra", "Tus favoritos"],
    "ferretería": ["Manos a la obra", "Arregla lo que falta", "Para tu proyecto", "Todo para la casa"],
    "ropa & moda": ["Renueva tu clóset", "Tu nuevo look", "Tendencias para ti", "Vístete a tu estilo"],
    "tecnología": ["Gadgets para ti", "Lo último en tech", "Conéctate", "Tus accesorios"],
    "servicios": ["Resuelve hoy", "A tu servicio", "Lo que necesitas resolver", "Sin complicarte"],
    "otros": ["Descubre algo nuevo", "Para ti", "Échale un ojo", "Quizás te guste"],
}

# Plantillas "basadas en gusto" cuando la categoría coincide con tus intereses ({c} en minúscula).
TASTE_TITLE_TEMPLATES = ["Porque te gusta {c}", "Más de {c} para ti", "Sigue con {c}", "{c} para ti"]

# Títulos manuales por categoría (editables desde el admin, SIN vectores). Se SUMAN al pool creativo.
SECTION_TITLES_OVERRIDE = {}  # {categoria_lower: [titulos]}


def load_section_titles(db):
    """Carga los títulos manuales por categoría desde Firestore (config/section_titles)."""
    if not db:
        return
    try:
        doc = db.collection('config').document('section_titles').get()
        if doc.exists:
            data = (doc.to_dict() or {}).get('titles') or {}
            SECTION_TITLES_OVERRIDE.clear()
            for k, v in data.items():
                if isinstance(v, list):
                    clean = [str(t).strip() for t in v if str(t).strip()]
                    if clean:
                        SECTION_TITLES_OVERRIDE[k.lower()] = clean
            print(f"[Títulos] {len(SECTION_TITLES_OVERRIDE)} categorías con títulos manuales cargadas.")
    except Exception as e:
        logger.warning(f"[Títulos] No se pudieron cargar: {e}")


def titles_for_category(cat_l: str):
    """Pool combinado: títulos manuales (admin) + creativos por defecto."""
    return (SECTION_TITLES_OVERRIDE.get(cat_l, []) + CATEGORY_TITLES.get(cat_l, [])) or None


def rerank_by_coherence(rows, anchor_k: int = 4):
    """Coherencia de sección: reordena una fila por cercanía a su 'tema dominante'.

    El tema lo define el centroide (vector promedio) de los productos más afines al concepto.
    Los outliers (ej. audífonos en una fila de comida) caen al fondo — NO se borran, solo bajan.
    Es semántico: el hielo se queda en una fila de licores (está cerca del tema), aunque sea de otra
    categoría; y una comida para preparar convive con la de restaurante porque ambas son 'comida'.

    `rows` debe venir ya ordenado por cercanía al concepto (los primeros = ancla del tema)."""
    if len(rows) < 4:
        return rows  # muy pocos para hablar de un 'tema'
    vecs = []
    for r in rows:
        e = r.get("embedding")
        if not e:
            return rows  # si falta algún vector, no arriesgamos el reordenado
        vecs.append(np.frombuffer(e, dtype=np.float32))

    def _centroid(vlist):
        c = np.mean(vlist, axis=0)
        n = float(np.linalg.norm(c))
        return (c / n) if n > 0 else c

    def _cos_dist(v, c):
        vn = float(np.linalg.norm(v))
        return (1.0 - float(np.dot(v, c) / vn)) if vn > 0 else 1.0

    # 1) Centroide inicial de TODOS: la mayoría define el tema.
    c0 = _centroid(vecs)
    if float(np.linalg.norm(c0)) == 0:
        return rows
    # 2) Centroide ROBUSTO: recalcularlo con la mayoría más cercana, descartando los outliers
    #    (así un audífono colado arriba no contamina el 'tema' de comida).
    order = sorted(range(len(vecs)), key=lambda i: _cos_dist(vecs[i], c0))
    keep_n = max(3, int(round(len(vecs) * 0.7)))
    c1 = _centroid([vecs[i] for i in order[:keep_n]])

    # 3) Reordenar todo por cercanía al tema dominante → los outliers caen al fondo.
    return [r for r, _ in sorted(zip(rows, vecs), key=lambda p: _cos_dist(p[1], c1))]

class HomeFeedRequest(BaseModel):
    activities: List[dict] = []
    lat: Optional[float] = None
    lng: Optional[float] = None
    override_hour: Optional[int] = None
    override_weather_temp: Optional[float] = None
    override_weather_code: Optional[int] = None
    sim_prompt: Optional[str] = None  # solo simulador admin: describe el gusto y se embebe como vector de usuario
    preferred_categories: List[str] = []  # gustos del onboarding (cold-start)


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

        # ¿Arranque en frío? = aún no hay gusto por comportamiento
        cold_start = user_vec_np is None

        # 1.05 Cold-start: construir un VECTOR inicial desde los gustos del onboarding,
        # promediando productos reales de esas categorías → feed personalizado desde el registro.
        if cold_start and req.preferred_categories:
            try:
                import sqlite_vec as _sv
                cats_lower = [c2.strip().lower() for c2 in req.preferred_categories if c2]
                if cats_lower:
                    ph = ",".join(["?"] * len(cats_lower))
                    # Match por la categoría del COMERCIO (la del onboarding es de la app),
                    # con respaldo a la categoría del producto.
                    rows_cs = c.execute(f"""
                        SELECT pv.embedding FROM product_vectors pv
                        JOIN search_index s ON s.id = pv.product_id AND s.type = 'product'
                        LEFT JOIN search_index st ON st.id = s.storeId AND st.type = 'store'
                        WHERE (LOWER(st.category) IN ({ph}) OR LOWER(s.category) IN ({ph}))
                        AND CAST(s.available AS INTEGER) = 1
                        LIMIT 80
                    """, tuple(cats_lower) + tuple(cats_lower)).fetchall()
                    if rows_cs:
                        acc = np.zeros(768, dtype=np.float32)
                        for r_cs in rows_cs:
                            acc += np.frombuffer(r_cs["embedding"], dtype=np.float32)
                        nrm = np.linalg.norm(acc)
                        if nrm > 0:
                            user_vec_np = acc / nrm
                            user_vector = _sv.serialize_float32(user_vec_np.tolist())
            except Exception as e:
                logger.warning(f"[Cold-start] No se pudo construir vector de gustos: {e}")

        # 1.1 Simulador admin: convertir el prompt de gusto en vector de usuario
        if req.sim_prompt and user_vec_np is None:
            try:
                import sqlite_vec
                from core.genai_client import embed_text
                v = np.array(embed_text(req.sim_prompt, task_type="retrieval_query"), dtype=np.float32)
                n = np.linalg.norm(v)
                if n > 0:
                    user_vec_np = v / n
                    user_vector = sqlite_vec.serialize_float32(user_vec_np.tolist())
            except Exception as e:
                logger.warning(f"[Sim Prompt] No se pudo embeber el prompt: {e}")

        # 1.2 Alimentar user_activity_cache desde la actividad reciente (fuente del collaborative filtering)
        if req.activities and not req.sim_prompt:
            score_map = {'purchase': 5.0, 'like': 4.0, 'cart': 3.0, 'search': 2.0, 'click': 1.0,
                         'view': 1.0, 'view_product': 1.0, 'ignored': -0.5}
            try:
                with sqlite_lock:
                    for act in req.activities:
                        pid = act.get('productId')
                        if pid:
                            atype = act.get('type', 'view')
                            c.execute(
                                "INSERT OR REPLACE INTO user_activity_cache (user_id, product_id, activity_type, score, timestamp) "
                                "VALUES (?, ?, ?, ?, ?)",
                                (uid, pid, atype, score_map.get(atype, 1.0), act.get('timestamp') or '')
                            )
                    conn.commit()
            except Exception as e:
                logger.warning(f"[Activity Cache] Error: {e}")

        # 2. Clima + pesos continuos + vector de contexto
        temp, code, tmax, tmin = get_weather(req.lat, req.lng, req.override_weather_temp, req.override_weather_code)
        weights = compute_context_weights(temp, code, current_hour, tmax, tmin)
        ctx = build_context_vector(user_vec_np, weights)

        global_seen_ids = set()

        # Ubicaciones de comercios para el ranking por cercanía
        store_loc = {}
        if req.lat is not None and req.lng is not None:
            try:
                for lr in c.execute("SELECT store_id, lat, lng FROM store_locations").fetchall():
                    store_loc[lr["store_id"]] = (lr["lat"], lr["lng"])
            except Exception as e:
                logger.warning(f"[Geo] No se pudieron cargar ubicaciones: {e}")

        # Engagement real por producto (CTR aprendido + exploración bandit)
        item_stats = {}
        try:
            for sr in c.execute("SELECT product_id, impressions, clicks, purchases FROM item_stats").fetchall():
                item_stats[sr["product_id"]] = (sr["impressions"], sr["clicks"], sr["purchases"])
        except Exception as e:
            logger.warning(f"[ItemStats] Error: {e}")

        # Cold-start: además del vector inicial, empujamos las categorías del onboarding
        # (solo mientras no haya comportamiento real; luego el gusto real manda).
        pref_set = set()
        if cold_start and req.preferred_categories:
            pref_set = {c.strip().lower() for c in req.preferred_categories if c}

        # Afinidad de tienda: cuántas veces visitó cada comercio (de la actividad reciente)
        store_visits = {}
        # Categorías que le interesan al usuario (para títulos "basados en gusto")
        interest_counts = {}
        POSITIVE = {'view_product', 'view', 'click', 'like', 'cart', 'purchase', 'search'}
        for act in req.activities:
            if act.get('type') == 'view_store' and act.get('storeId'):
                store_visits[act['storeId']] = store_visits.get(act['storeId'], 0) + 1
            cat_a = (act.get('category') or '').lower()
            if cat_a and cat_a != 'general' and act.get('type') in POSITIVE:
                interest_counts[cat_a] = interest_counts.get(cat_a, 0) + 1
        # Solo categorías con interés real (2+ interacciones) llevan título "basado en gusto"
        user_interest_cats = {c for c, n in interest_counts.items() if n >= 2}

        def add_proximity(row):
            """Suma boost por cercanía + cold-start de gustos + afinidad de tienda al final_score."""
            row_cat = str(row.get("store_category") or row.get("category") or "").lower()
            if pref_set and row_cat in pref_set:
                row["final_score"] = row.get("final_score", 0) + 0.4  # empuje de gustos declarados
            visits = store_visits.get(row.get("storeId"))
            if visits:
                row["final_score"] = row.get("final_score", 0) + min(0.35, 0.12 * visits)  # tiendas que frecuentas
            if not store_loc or req.lat is None:
                return
            loc = store_loc.get(row.get("storeId"))
            if loc:
                dist = haversine_km(req.lat, req.lng, loc[0], loc[1])
                row["distance_km"] = round(dist, 1)
                row["_prox"] = proximity_boost(dist)
                row["final_score"] = row.get("final_score", 0) + row["_prox"]

        def take_interleaved_by_store(candidates, n):
            """Llena alternando entre negocios (variedad). Si solo hay uno, llena con ese."""
            from collections import OrderedDict
            buckets = OrderedDict()
            for row in candidates:
                if row["id"] in global_seen_ids:
                    continue
                buckets.setdefault(row.get("storeId", ""), []).append(row)
            out = []
            while len(out) < n and any(buckets.values()):
                for sid in list(buckets.keys()):
                    if buckets[sid]:
                        row = buckets[sid].pop(0)
                        row.pop("embedding", None)
                        out.append(row)
                        global_seen_ids.add(row["id"])
                        if len(out) >= n:
                            break
            return out

        def take_from_pool(candidates, n, store_cap=2, cat_cap=None):
            out = []
            store_counts = {}
            cat_counts = {}
            for row in candidates:
                rid = row["id"]
                sid = row.get("storeId", "")
                cat = str(row.get("category", "")).lower()
                if rid in global_seen_ids:
                    continue
                if store_counts.get(sid, 0) >= store_cap:
                    continue
                if cat_cap is not None and cat_counts.get(cat, 0) >= cat_cap:
                    continue  # diversidad: no saturar de una sola categoría
                row.pop("embedding", None)  # binario, no serializable a JSON
                out.append(row)
                global_seen_ids.add(rid)
                store_counts[sid] = store_counts.get(sid, 0) + 1
                cat_counts[cat] = cat_counts.get(cat, 0) + 1
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
                       st.name as storeName, st.category as store_category
                FROM product_vectors p
                JOIN search_index s ON p.product_id = s.id AND s.type = 'product'
                LEFT JOIN (SELECT id, name, isOpen, category FROM search_index WHERE type='store') st ON st.id = s.storeId
                WHERE CAST(s.available AS INTEGER) = 1 AND CAST(st.isOpen AS INTEGER) = 1
                ORDER BY distance ASC
                LIMIT 200
            """, (ctx,))
            for raw in c.fetchall():
                row = dict(raw)
                row["final_score"] = score_product(row, row["distance"], item_stats)
                add_proximity(row)
                pool.append(row)
        else:
            # Sin señal (usuario nuevo, sin clima/hora marcada) → popularidad
            c.execute("""
                SELECT s.id, s.type, s.storeId, s.name, s.category, s.description,
                       s.price, s.icon, s.imageUrl, s.onSale, s.salePrice, s.likes, s.views, s.purchases,
                       st.name as storeName, st.category as store_category
                FROM search_index s
                LEFT JOIN (SELECT id, name, isOpen, category FROM search_index WHERE type='store') st ON st.id = s.storeId
                WHERE s.type = 'product' AND CAST(s.available AS INTEGER) = 1 AND CAST(st.isOpen AS INTEGER) = 1
                ORDER BY CAST(s.purchases AS INTEGER) DESC, CAST(s.likes AS INTEGER) DESC
                LIMIT 200
            """)
            for raw in c.fetchall():
                row = dict(raw)
                row["distance"] = 1.0
                row["final_score"] = score_product(row, 1.0, item_stats)
                add_proximity(row)
                pool.append(row)

        pool.sort(key=lambda x: x["final_score"], reverse=True)

        # 4a. "Para ti ahora" (featured)
        # Si hay gusto del usuario, ordenamos por AFINIDAD PURA (cercanía al gusto), no por
        # popularidad — así no se cuelan productos populares pero irrelevantes (p.ej. papel higiénico).
        personalized = user_vector is not None
        if personalized:
            # Afinidad (cercanía semántica al gusto) y, para empates, cercanía física
            featured_candidates = sorted(pool, key=lambda x: x.get("distance", 1.0) - x.get("_prox", 0.0))
        else:
            featured_candidates = pool  # usuario nuevo → ya está ordenado por popularidad/contexto
        featured = take_from_pool(featured_candidates, 6, cat_cap=3)  # máx 3 de una misma categoría
        if featured:
            feed_sections.append({
                "id": "dyn_for_you",
                "type": "products",
                "title": "Para ti ahora" if personalized else "Lo mejor ahora",
                "subtitle": "Según tu gusto, la hora y el clima" if personalized else "Lo más popular para este momento",
                "items": featured,
                "isPersonalized": personalized,
                "layout": "featured",
            })

        # 4b. Filas ambientales (solo si el peso es significativo Y hay productos realmente afines)
        env_rows = 0
        for concept_id, w in sorted(weights.items(), key=lambda kv: kv[1], reverse=True):
            if w < 0.5 or concept_id not in ENV_TITLES or env_rows >= 2:
                continue
            # Distancia de cada producto al concepto (una sola vez)
            scored = []
            for r in pool:
                if r["id"] in global_seen_ids:
                    continue
                d = concept_distance(r.get("embedding"), concept_id)
                scored.append((d, r))
            scored.sort(key=lambda x: x[0])
            if not scored:
                continue
            # Corte de relevancia: solo productos cerca del mejor (evita rellenar con cosas no afines)
            best = scored[0][0]
            cutoff = min(best + 0.12, 0.6)
            relevant = [r for d, r in scored if d <= cutoff]
            # Coherencia de sección: el tema dominante manda; los outliers (audífonos en una fila
            # de comida) caen al fondo automáticamente. No borra nada, solo reordena.
            relevant = rerank_by_coherence(relevant)
            # Curación manual: quitar los que marcaste como "no van aquí" y mandar al final los "demote"
            _kept, _demoted = [], []
            for r in relevant:
                act = curation_action("concept", concept_id, r["id"])
                if act == "exclude":
                    continue
                (_demoted if act == "demote" else _kept).append(r)
            relevant = _kept + _demoted
            items = take_from_pool(relevant, 6)
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

        # Agrupamos por la categoría del COMERCIO (app: Restaurante, Comidas rápidas...),
        # no por la del producto (texto libre tipo "Tacos") → filas limpias y consistentes.
        cat_groups = defaultdict(list)
        for r in pool:
            if r["id"] in global_seen_ids:
                continue
            cat = r.get("store_category") or r.get("category") or "general"
            cat_groups[cat].append(r)

        ordered_cats = sorted(cat_groups.keys(), key=lambda cat: cat_groups[cat][0]["final_score"], reverse=True)
        for cat in ordered_cats:
            if len(feed_sections) >= 12:
                break
            # Carrusel por categoría: alterna entre negocios (variedad); si solo hay uno, llena con ese
            items = take_interleaved_by_store(cat_groups[cat], 10)
            if len(items) >= 2:
                cat_l = cat.lower()
                pool = titles_for_category(cat_l)  # manuales (admin) + creativos, combinados
                if cat_l in user_interest_cats:
                    title = random.choice(TASTE_TITLE_TEMPLATES).format(c=cat_l)  # basado en gusto
                elif pool:
                    title = random.choice(pool)
                elif anchor_title_map.get(cat_l):
                    title = random.choice(anchor_title_map[cat_l])               # legacy (anclas IA)
                else:
                    title = cat                                                  # último recurso
                feed_sections.append({
                    "id": f"dyn_cat_{str(cat).replace(' ', '_')}",
                    "type": "products",
                    "title": title,
                    "subtitle": random.choice(SECTION_SUBTITLES),
                    "items": items,
                    "layout": "scroll",
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
                    LEFT JOIN (SELECT id, name, isOpen FROM search_index WHERE type='store') s ON s.id = p.storeId
                    WHERE p.type = 'product' AND p.category = ? AND CAST(p.available AS INTEGER) = 1
                    AND CAST(s.isOpen AS INTEGER) = 1
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

        # 4e.bis RED DE SEGURIDAD: garantizar que el catálogo se muestre aunque falten vectores.
        # El pool vectorial solo incluye productos ya vectorizados; si la vectorización
        # va atrasada o falló, el feed quedaría vacío. Esto trae productos directo del índice.
        total_product_items = sum(len(s.get("items", [])) for s in feed_sections if s.get("type") == "products")
        if total_product_items < 4:
            try:
                c.execute("""
                    SELECT p.id, p.type, p.storeId, p.name, p.category, p.description,
                           p.price, p.icon, p.imageUrl, p.onSale, p.salePrice, p.likes, p.views, p.purchases,
                           s.name as storeName
                    FROM search_index p
                    LEFT JOIN (SELECT id, name, isOpen FROM search_index WHERE type='store') s ON s.id = p.storeId
                    WHERE p.type = 'product' AND CAST(p.available AS INTEGER) = 1 AND CAST(s.isOpen AS INTEGER) = 1
                    ORDER BY CAST(p.purchases AS INTEGER) DESC, CAST(p.likes AS INTEGER) DESC, RANDOM()
                    LIMIT 40
                """)
                catalog = [dict(r) for r in c.fetchall()]
                fallback_items = take_from_pool(catalog, 8, store_cap=3)
                if fallback_items:
                    feed_sections.append({
                        "id": "dyn_catalog",
                        "type": "products",
                        "title": "Explora el catálogo",
                        "subtitle": "Descubre lo que hay cerca de ti",
                        "items": fallback_items,
                        "layout": "grid" if len(fallback_items) >= 4 else "scroll",
                    })
            except Exception as e:
                logger.error(f"[Catalog Fallback] Error: {e}")

        # 4f. Tiendas recomendadas
        try:
            if user_vector:
                c.execute("""
                    SELECT s.store_id, vec_distance_cosine(s.embedding, ?) AS distance,
                           st.name, st.category, st.description, st.imageUrl, st.likes, st.isOpen,
                           fs.featured_until
                    FROM store_vectors s
                    JOIN search_index st ON st.id = s.store_id AND st.type = 'store'
                    LEFT JOIN featured_stores fs ON fs.store_id = s.store_id
                    WHERE CAST(st.isOpen AS INTEGER) = 1
                    ORDER BY distance ASC
                    LIMIT 200
                """, (user_vector,))
            else:
                c.execute("""
                    SELECT s.store_id, 1.0 AS distance,
                           st.name, st.category, st.description, st.imageUrl, st.likes, st.isOpen,
                           fs.featured_until
                    FROM store_vectors s
                    JOIN search_index st ON st.id = s.store_id AND st.type = 'store'
                    LEFT JOIN featured_stores fs ON fs.store_id = s.store_id
                    WHERE CAST(st.isOpen AS INTEGER) = 1
                    ORDER BY CAST(st.likes AS INTEGER) DESC
                    LIMIT 200
                """)

            now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
            candidate_stores = []
            for raw_row in c.fetchall():
                row = dict(raw_row)
                likes_val = int(row["likes"] or 0)
                distance = row.get("distance", 1.0)
                affinity = max(0.0, 1.0 - distance)
                novelty = 0.2 if likes_val < 5 else 0.0
                row["final_score"] = affinity + (math.log1p(likes_val) / 20.0) + novelty + random.uniform(0.0, 0.15)
                loc = store_loc.get(row["store_id"])
                if loc:
                    d = haversine_km(req.lat, req.lng, loc[0], loc[1])
                    row["distance_km"] = round(d, 1)
                    row["final_score"] += proximity_boost(d)
                # Comercio Destacado (pagado): marca para el badge. En el home ya tiene su
                # propia sección "Comercios Destacados", así que aquí NO forzamos el ranking
                # (evita duplicarlo arriba); la prioridad fuerte va en las búsquedas.
                fu = row.get("featured_until") or 0
                row["is_featured"] = bool(fu and fu > now_ms)
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
                        "distance_km": row.get("distance_km"),
                        "isFeatured": row.get("is_featured", False),
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

        # 4.z Ciclo de aprendizaje por CTR: reordenar secciones por su tasa de clic histórica.
        # "Para ti" queda fija arriba y las tiendas segundas; el resto sube/baja según cuánto se toca.
        try:
            stats = {}
            for sr in c.execute("SELECT section_id, impressions, clicks FROM section_stats").fetchall():
                stats[sr["section_id"]] = (sr["impressions"], sr["clicks"])

            def ctr_score(sid):
                imp, clk = stats.get(sid, (0, 0))
                if imp < 5:
                    return 0.5  # poca data → prior neutro alto para darle oportunidad
                return (clk + 1) / (imp + 5)

            featured_secs = [s for s in feed_sections if s.get("id") == "dyn_for_you"]
            store_secs = [s for s in feed_sections if s.get("type") == "stores"]
            rest_secs = [s for s in feed_sections if s not in featured_secs and s not in store_secs]
            rest_secs.sort(key=lambda s: ctr_score(s.get("id", "")), reverse=True)
            feed_sections = featured_secs + store_secs + rest_secs
        except Exception as e:
            logger.warning(f"[CTR Reorder] Error: {e}")

        # 5. Registrar impresiones por sección (denominador del CTR)
        if not req.sim_prompt:
            try:
                shown_pids = set()
                for sec in feed_sections:
                    if sec.get("type") == "products":
                        for it in sec.get("items", []):
                            if it.get("id"):
                                shown_pids.add(it["id"])
                with sqlite_lock:
                    for sec in feed_sections:
                        c.execute(
                            "INSERT INTO section_stats (section_id, impressions, clicks) VALUES (?, 1, 0) "
                            "ON CONFLICT(section_id) DO UPDATE SET impressions = impressions + 1, updated_at = datetime('now')",
                            (sec["id"],)
                        )
                    for pid in shown_pids:
                        c.execute(
                            "INSERT INTO item_stats (product_id, impressions, clicks, purchases) VALUES (?, 1, 0, 0) "
                            "ON CONFLICT(product_id) DO UPDATE SET impressions = impressions + 1, updated_at = datetime('now')",
                            (pid,)
                        )
                    conn.commit()
            except Exception as e:
                logger.warning(f"[Impressions] Error: {e}")

    except Exception as e:
        logger.error(f"[Home Feed] Unhandled error: {e}")
        return {"sections": []}
    finally:
        conn.close()

    return {"sections": feed_sections}
