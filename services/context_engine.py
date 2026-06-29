"""
Motor de Recomendaciones por Vector de Contexto.

En lugar de anclas fijas, se construye un único "vector de contexto" por petición
que mezcla el gusto del usuario con pesos CONTINUOS de clima y hora. Todo el catálogo
se rankea por cercanía a ese vector + señales de popularidad/novedad/oferta.
"""
import math
import time
import logging
import requests
import numpy as np
import sqlite_vec

from data.concepts import DICCIONARIO_CONCEPTOS
from services.weather import WEATHER_CACHE_STORE

logger = logging.getLogger(__name__)

# Pesos GLOBALES del ranking. Se auto-ajustan con el comportamiento real (ver tune_ranking_weights).
# affinity y exploration se mantienen estructurales; el resto aprende de los datos.
RANK_WEIGHTS = {
    "affinity": 0.55,
    "popularity": 0.18,
    "ctr": 0.25,
    "exploration": 0.12,
    "novelty": 0.08,
    "sale": 0.10,
}
# Topes de seguridad para que el auto-ajuste nunca se vaya a un extremo.
_WEIGHT_BOUNDS = {
    "popularity": (0.08, 0.30),
    "ctr": (0.15, 0.45),
    "novelty": (0.03, 0.20),
    "sale": (0.03, 0.35),
}


def _clamp_bounds(key, val):
    lo, hi = _WEIGHT_BOUNDS.get(key, (0.0, 1.0))
    return max(lo, min(hi, val))


def load_ranking_weights(db):
    """Carga los pesos guardados (config/ranking) si existen."""
    if not db:
        return
    try:
        doc = db.collection('config').document('ranking').get()
        if doc.exists:
            saved = (doc.to_dict() or {}).get('weights') or {}
            for k, v in saved.items():
                if k in RANK_WEIGHTS and isinstance(v, (int, float)):
                    RANK_WEIGHTS[k] = float(v)
            print(f"[Ranking] Pesos cargados desde Firestore: {RANK_WEIGHTS}")
    except Exception as e:
        print(f"[Ranking] No se pudieron cargar pesos: {e}")


def save_ranking_weights(db, new_weights: dict) -> dict:
    """Guarda pesos editados manualmente: valida tipos, aplica topes de seguridad y persiste.
    Devuelve el diccionario final de pesos."""
    for k, v in (new_weights or {}).items():
        if k in RANK_WEIGHTS and isinstance(v, (int, float)):
            RANK_WEIGHTS[k] = float(_clamp_bounds(k, float(v)))
    if db:
        try:
            db.collection('config').document('ranking').set({"weights": RANK_WEIGHTS}, merge=True)
        except Exception as e:
            logger.error(f"[Ranking] Error guardando pesos manuales: {e}")
    print(f"[Ranking] Pesos guardados manualmente: {RANK_WEIGHTS}")
    return dict(RANK_WEIGHTS)


def tune_ranking_weights(db):
    """Auto-ajuste de pesos GLOBALES desde el engagement real (CTR por producto).
    Suavizado (mueve 25% hacia el objetivo) y con topes → estable, sin sobresaltos."""
    from statistics import mean
    from core.database import get_db_connection

    conn = get_db_connection()
    try:
        rows = conn.execute("""
            SELECT i.impressions AS imp, i.clicks AS clk, s.onSale AS onsale, i.purchases AS pur
            FROM item_stats i
            JOIN search_index s ON s.id = i.product_id AND s.type = 'product'
            WHERE i.impressions >= 5
        """).fetchall()
    finally:
        conn.close()

    if len(rows) < 20:
        return {"status": "skipped", "reason": "datos insuficientes", "rows": len(rows)}

    def ctr(r):
        return r["clk"] / max(r["imp"], 1)

    all_ctr = mean(ctr(r) for r in rows) or 0.0
    sale_rows = [r for r in rows if str(r["onsale"]) in ("1", "True", "true")]
    nosale_rows = [r for r in rows if r not in sale_rows]
    sale_ctr = mean(ctr(r) for r in sale_rows) if sale_rows else all_ctr
    nosale_ctr = mean(ctr(r) for r in nosale_rows) if nosale_rows else all_ctr

    def smooth(cur, target):
        return round(cur + 0.25 * (target - cur), 4)

    # 1) Oferta: si los productos en oferta convierten mejor, sube el peso de "sale".
    sale_lift = (sale_ctr + 1e-6) / (nosale_ctr + 1e-6)
    RANK_WEIGHTS["sale"] = smooth(RANK_WEIGHTS["sale"], _clamp_bounds("sale", 0.10 * sale_lift))

    # 2) CTR vs popularidad: a más datos, confiamos más en el CTR aprendido y menos en likes (adivinado).
    total_clicks = sum(r["clk"] for r in rows)
    RANK_WEIGHTS["ctr"] = smooth(RANK_WEIGHTS["ctr"], _clamp_bounds("ctr", 0.15 + min(total_clicks, 1000) / 1000 * 0.25))
    RANK_WEIGHTS["popularity"] = smooth(RANK_WEIGHTS["popularity"], _clamp_bounds("popularity", 0.28 - min(total_clicks, 1000) / 1000 * 0.15))

    if db:
        try:
            db.collection('config').document('ranking').set({"weights": RANK_WEIGHTS}, merge=True)
        except Exception as e:
            logger.error(f"[Ranking] Error persistiendo pesos: {e}")

    print(f"[Ranking] Pesos auto-ajustados: {RANK_WEIGHTS}")
    return {"status": "ok", "weights": dict(RANK_WEIGHTS), "sale_lift": round(sale_lift, 2), "total_clicks": total_clicks}


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def haversine_km(lat1, lng1, lat2, lng2) -> float:
    """Distancia en km entre dos coordenadas."""
    try:
        r = 6371.0
        p1, p2 = math.radians(lat1), math.radians(lat2)
        dphi = math.radians(lat2 - lat1)
        dlmb = math.radians(lng2 - lng1)
        a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
        return 2 * r * math.asin(math.sqrt(a))
    except Exception:
        return 9999.0


def proximity_boost(dist_km: float, scale: float = 5.0, weight: float = 0.4) -> float:
    """Boost que premia la cercanía: máximo `weight` a 0 km, decae con la distancia."""
    if dist_km is None or dist_km >= 9999:
        return 0.0
    return weight * math.exp(-dist_km / scale)


def _time_bump(hour: int, center: float, spread: float) -> float:
    """Curva gaussiana de proximidad horaria con wraparound de medianoche."""
    diff = abs(hour - center)
    if diff > 12:
        diff = 24 - diff
    return math.exp(-(diff ** 2) / (2 * spread ** 2))


def get_weather(lat, lng, override_temp=None, override_code=None):
    """Devuelve (temp, code, tmax, tmin). Soporta overrides del simulador y caché por zona.
    tmax/tmin = máx/mín del día local (para umbrales relativos a la ciudad)."""
    if override_temp is not None:
        return float(override_temp), int(override_code or 0), None, None
    if lat is None or lng is None:
        return None, None, None, None
    try:
        lat_key, lng_key = round(lat, 1), round(lng, 1)
        loc_key = f"{lat_key}_{lng_key}"
        now_ts = time.time()
        cached = WEATHER_CACHE_STORE.get(loc_key)
        if cached and (now_ts - cached["time"] < 3600):
            return cached["temp"], cached["code"], cached.get("tmax"), cached.get("tmin")
        w_res = requests.get(
            f"https://api.open-meteo.com/v1/forecast?latitude={lat_key}&longitude={lng_key}"
            f"&current_weather=true&daily=temperature_2m_max,temperature_2m_min&timezone=auto&forecast_days=1",
            timeout=3,
        ).json()
        if "current_weather" in w_res:
            temp = w_res["current_weather"].get("temperature", 20)
            code = w_res["current_weather"].get("weathercode", 0)
            daily = w_res.get("daily", {})
            tmax = (daily.get("temperature_2m_max") or [None])[0]
            tmin = (daily.get("temperature_2m_min") or [None])[0]
            WEATHER_CACHE_STORE[loc_key] = {"temp": temp, "code": code, "tmax": tmax, "tmin": tmin, "time": now_ts}
            return temp, code, tmax, tmin
    except Exception as e:
        logger.warning(f"[Weather] Error obteniendo clima: {e}")
    return None, None, None, None


def compute_context_weights(temp, code, hour: int, tmax=None, tmin=None) -> dict:
    """Pesos CONTINUOS 0..1 de cada concepto ambiental/horario.
    Si hay tmax/tmin del día, el calor/frío es RELATIVO a la ciudad (lo que aquí se siente caluroso)."""
    weights = {}

    if temp is not None:
        # Absoluto = "¿hace calor/frío de verdad?". Es la base que NO se puede inventar.
        abs_hot = _clamp((temp - 22) / 10)      # 0 a 22°, 1 a 32°+
        abs_cold = _clamp((20 - temp) / 12)     # 0 a 20°, 1 a 8°
        if tmax is not None and tmin is not None and (tmax - tmin) >= 3:
            # Lo relativo a la ciudad solo MODULA lo absoluto (no crea calor en un día frío).
            rel = _clamp((temp - tmin) / (tmax - tmin))
            hot = abs_hot * (0.7 + 0.3 * rel)
            cold = abs_cold * (0.7 + 0.3 * (1 - rel))
        else:
            hot, cold = abs_hot, abs_cold
        # Lluvia/niebla/tormenta: la gente quiere algo calientito, NO refrescarse.
        if code is not None and code >= 51:
            cold = max(cold, 0.6)
            hot *= 0.25
        weights["ENV_CALOR"] = hot
        weights["ENV_FRIO"] = cold

    if hour is not None:
        weights["ENV_MANANA"] = _time_bump(hour, 7.5, 2.5)
        weights["ENV_MEDIODIA"] = _time_bump(hour, 13.0, 2.5)
        weights["ENV_NOCHE"] = _time_bump(hour, 21.0, 3.0)

    return {k: v for k, v in weights.items() if v > 0.05}


def build_context_vector(user_vec_np, weights: dict, Wu: float = 2.5):
    """
    Mezcla Wu·usuario + Σ(peso·concepto) y normaliza.
    - user_vec_np: np.ndarray(768) o None
    - weights: {concept_id: peso}
    Wu alto (2.5) hace que el GUSTO del usuario domine; el clima/hora solo empuja suave.
    Retorna bytes serializados para sqlite-vec, o None si no hay señal.
    """
    ctx = np.zeros(768, dtype=np.float32)
    has_signal = False

    if user_vec_np is not None:
        ctx += Wu * user_vec_np
        has_signal = True

    for concept_id, w in weights.items():
        vec = DICCIONARIO_CONCEPTOS.get(concept_id)
        if vec is not None:
            ctx += w * vec
            has_signal = True

    if not has_signal:
        return None

    norm = np.linalg.norm(ctx)
    if norm <= 0:
        return None
    ctx = ctx / norm
    return sqlite_vec.serialize_float32(ctx.tolist())


def score_product(row: dict, distance: float, stats: dict | None = None) -> float:
    """Scoring unificado: afinidad + popularidad + CTR aprendido + exploración (bandit) + novedad + oferta.

    `stats`: {product_id: (impressions, clicks, purchases)} con engagement REAL del feed.
    El CTR aprendido crece en influencia a medida que hay datos; la exploración (UCB) premia
    a los productos poco mostrados para descubrir cuáles gustan.
    """
    affinity = max(0.0, 1.0 - distance)
    purchases = float(row.get("purchases") or 0)
    likes = float(row.get("likes") or 0)
    views = float(row.get("views") or 0)

    cr_boost = 0.0
    if views >= 10:
        cr = purchases / views
        if cr > 0.1:
            cr_boost = cr * 2.0

    C, m = 20.0, 1.0
    bayes_purchases = (views * (purchases / (views + 0.1)) + C * m) / (views + C)
    popularity = math.log1p(bayes_purchases * 5.0 + likes * 0.5) / 10.0

    novelty = 0.2 if (purchases == 0 and views <= 15) else (-0.3 if purchases == 0 and views > 50 else 0.0)
    sale_boost = 0.15 if str(row.get("onSale", "0")) == "1" else 0.0

    # --- Señal APRENDIDA (CTR/conversión real del feed) + EXPLORACIÓN (bandit UCB) ---
    learned = 0.0
    explore = 0.30  # default alto: producto nunca mostrado → máxima curiosidad
    if stats is not None:
        st = stats.get(str(row.get("id", "")))
        if st:
            imp, clk, pur = st
            ctr = (clk + 1.0) / (imp + 10.0)          # suavizado bayesiano
            conv = (pur + 0.5) / (imp + 10.0)
            learned = ctr + conv * 2.0                 # comprar pesa más que clicar
            explore = 0.30 / math.sqrt(imp + 1.0)      # menos exploración cuanto más se ha mostrado

    noise = (abs(hash(str(row.get("id", "")))) % 50) / 1000.0

    W = RANK_WEIGHTS
    return (affinity * W["affinity"]) + (popularity * W["popularity"]) + cr_boost \
        + (learned * W["ctr"]) + (explore * W["exploration"]) + (novelty * W["novelty"]) + (sale_boost * W["sale"]) + noise


def concept_distance(product_emb_bytes: bytes, concept_id: str) -> float:
    """Distancia coseno de un producto a un concepto ambiental (para filas temáticas)."""
    vec = DICCIONARIO_CONCEPTOS.get(concept_id)
    if vec is None or not product_emb_bytes:
        return 1.0
    p = np.frombuffer(product_emb_bytes, dtype=np.float32)
    denom = (np.linalg.norm(p) * np.linalg.norm(vec)) + 1e-10
    return float(1.0 - np.dot(p, vec) / denom)
