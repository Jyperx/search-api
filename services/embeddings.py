import numpy as np
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor
import sqlite_vec
from core.config import LLM_MODEL
from core.genai_client import embed_text, embed_texts, generate_text
from data.concepts import CATEGORY_WEIGHTS, DICCIONARIO_CONCEPTOS

logger = logging.getLogger(__name__)
_embed_executor = ThreadPoolExecutor(max_workers=6)

def _serialize(vector: np.ndarray) -> bytes:
    return sqlite_vec.serialize_float32(vector.tolist())

def _find_closest_concept(vector: np.ndarray) -> tuple[float, str]:
    """Retorna (similitud_maxima, id_del_concepto_mas_cercano)"""
    best_sim = -1.0
    best_id = ""
    for concept_id, concept_vec in DICCIONARIO_CONCEPTOS.items():
        sim = float(np.dot(vector, concept_vec) / (np.linalg.norm(vector) * np.linalg.norm(concept_vec) + 1e-10))
        if sim > best_sim:
            best_sim = sim
            best_id = concept_id
    return best_sim, best_id

async def _call_gemini_embedding(text: str) -> np.ndarray | None:
    loop = asyncio.get_event_loop()
    try:
        def _blocking_call():
            embedding = np.array(embed_text(text), dtype=np.float32)  # ya truncado a 768 (Matryoshka)
            norm = np.linalg.norm(embedding)
            return embedding / norm if norm > 0 else embedding

        return await loop.run_in_executor(_embed_executor, _blocking_call)
    except Exception as e:
        logger.warning(f"Error en la API de embeddings de Gemini: {e}")
        return None

async def generate_product_embedding(name: str, category: str, description: str) -> tuple[bytes | None, str]:
    cat_key = category.lower().strip()
    anchor_weight = CATEGORY_WEIGHTS.get(cat_key, CATEGORY_WEIGHTS["default"])

    raw_text = f"Producto: {name}. Categoría: {category}. Descripción: {description}"
    vector_base = await _call_gemini_embedding(raw_text)
    if vector_base is None:
        return None, 'error'

    if anchor_weight == 0.0:
        return _serialize(vector_base), 'raw_only'

    if DICCIONARIO_CONCEPTOS:
        best_sim, best_anchor_id = _find_closest_concept(vector_base)
        if best_sim >= 0.62:
            anchor_vec = DICCIONARIO_CONCEPTOS[best_anchor_id]
            vector_final = (vector_base * (1 - anchor_weight)) + (anchor_vec * anchor_weight)
            vector_final = vector_final / np.linalg.norm(vector_final)
            return _serialize(vector_final), 'concept_anchoring'

    intent_str = ""
    try:
        prompt = (
            f"Producto: '{name}', categoría: '{category}', descripción: '{description}'. "
            f"Responde SOLO con 4 palabras clave separadas por comas que describan "
            f"el momento ideal para consumirlo (clima, hora, estado de ánimo)."
        )
        
        def _blocking_llm_call():
            return generate_text(prompt, model=LLM_MODEL, temperature=0.2, max_output_tokens=25)

        loop = asyncio.get_event_loop()
        res = await loop.run_in_executor(_embed_executor, _blocking_llm_call)

        if res and res.strip():
            intent_str = f" Contexto: {res.strip()}"
    except Exception as e:
        logger.warning(f"Falló el enriquecimiento Flash Lite para '{name}': {e}")

    enriched_text = raw_text + intent_str
    vector_enriched = await _call_gemini_embedding(enriched_text)
    if vector_enriched is not None:
        return _serialize(vector_enriched), 'flash_lite_enrichment'

    return _serialize(vector_base), 'raw_only'


def _llm_intent(name: str, category: str, description: str) -> str:
    """Pide al LLM Flash-Lite palabras de contexto (clima/hora/ánimo) para enriquecer el texto.
    Mismo prompt que el path de a-uno. Devuelve ' Contexto: ...' o '' si falla."""
    try:
        prompt = (
            f"Producto: '{name}', categoría: '{category}', descripción: '{description}'. "
            f"Responde SOLO con 4 palabras clave separadas por comas que describan "
            f"el momento ideal para consumirlo (clima, hora, estado de ánimo)."
        )
        res = generate_text(prompt, model=LLM_MODEL, temperature=0.2, max_output_tokens=25)
        return f" Contexto: {res.strip()}" if res and res.strip() else ""
    except Exception as e:
        logger.warning(f"[Batch] Falló enriquecimiento LLM: {e}")
        return ""


# Se desactiva solo si el modelo de embeddings demuestra que NO soporta lote (devuelve un número
# de vectores distinto al de textos enviados). A partir de ahí vectorizamos uno-por-uno.
_batch_supported = True


def _embed_items_one_by_one(items: list) -> list:
    """Fallback: cada producto por el path completo de a-uno (raw + anclaje + LLM). Calidad idéntica."""
    out = []
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        for (pid, name, category, description) in items:
            try:
                blob, source = loop.run_until_complete(
                    generate_product_embedding(name, category, description)
                )
                out.append((pid, blob, source))
            except Exception as e:
                logger.warning(f"[Single] Error vectorizando {pid}: {e}")
                out.append((pid, None, 'error'))
    finally:
        loop.close()
    return out


def _process_batch_vectors(items: list, raw_vectors: list) -> list:
    """Procesa los vectores base de un lote: anclaje a concepto + enriquecimiento LLM para los débiles.
    Garantiza una lista alineada 1:1 con items (ningún slot queda en None)."""
    results: list = [None] * len(items)
    weak: list = []  # (idx, item, vector_base) → no anclaron, necesitan enriquecimiento LLM

    for idx, (item, raw) in enumerate(zip(items, raw_vectors)):
        pid, name, category, description = item
        try:
            v = np.array(raw, dtype=np.float32)
            norm = np.linalg.norm(v)
            vector_base = v / norm if norm > 0 else v
            cat_key = (category or '').lower().strip()
            anchor_weight = CATEGORY_WEIGHTS.get(cat_key, CATEGORY_WEIGHTS["default"])

            if anchor_weight == 0.0 or not DICCIONARIO_CONCEPTOS:
                results[idx] = (pid, _serialize(vector_base), 'raw_only')
                continue

            best_sim, best_anchor_id = _find_closest_concept(vector_base)
            if best_sim >= 0.62:
                anchor_vec = DICCIONARIO_CONCEPTOS[best_anchor_id]
                vector_final = (vector_base * (1 - anchor_weight)) + (anchor_vec * anchor_weight)
                vector_final = vector_final / np.linalg.norm(vector_final)
                results[idx] = (pid, _serialize(vector_final), 'concept_anchoring')
            else:
                weak.append((idx, item, vector_base))
        except Exception as e:
            logger.warning(f"[Batch] Error procesando {pid}: {e}")
            results[idx] = (pid, None, 'error')

    if weak:
        intents = list(_embed_executor.map(lambda w: _llm_intent(w[1][1], w[1][2], w[1][3]), weak))
        enriched_texts = [
            f"Producto: {item[1]}. Categoría: {item[2]}. Descripción: {item[3]}{intent}"
            for (_idx, item, _vb), intent in zip(weak, intents)
        ]
        try:
            enriched_vectors = embed_texts(enriched_texts)
            if len(enriched_vectors) != len(weak):
                enriched_vectors = [None] * len(weak)
        except Exception as e:
            logger.warning(f"[Batch] Falló el embed enriquecido en lote: {e}")
            enriched_vectors = [None] * len(weak)

        for (idx, item, vb), ev in zip(weak, enriched_vectors):
            pid = item[0]
            if ev is not None:
                v = np.array(ev, dtype=np.float32)
                norm = np.linalg.norm(v)
                v = v / norm if norm > 0 else v
                results[idx] = (pid, _serialize(v), 'flash_lite_enrichment')
            else:
                results[idx] = (pid, _serialize(vb), 'raw_only')

    # Defensa: ningún slot debe quedar en None.
    for idx in range(len(items)):
        if results[idx] is None:
            results[idx] = (items[idx][0], None, 'error')
    return results


def generate_product_embeddings_batch(items: list) -> list:
    """Vectoriza una TANDA de productos. Intenta en lote (1 llamada de embeddings); si el modelo
    NO soporta lote (devuelve un conteo distinto), cae automáticamente a uno-por-uno con calidad
    idéntica. Devuelve lista de (product_id, bytes|None, source) alineada con items."""
    global _batch_supported
    if not items:
        return []

    if _batch_supported:
        raw_texts = [f"Producto: {n}. Categoría: {c}. Descripción: {d}" for (_id, n, c, d) in items]
        try:
            raw_vectors = embed_texts(raw_texts)
            if len(raw_vectors) == len(items):
                return _process_batch_vectors(items, raw_vectors)
            # Conteo distinto = el modelo no batchea de verdad → desactivar y caer a uno-por-uno.
            logger.warning(
                f"[Batch] El modelo de embeddings devolvió {len(raw_vectors)}/{len(items)} vectores; "
                f"no soporta lote. Cambiando a uno-por-uno de aquí en adelante."
            )
            _batch_supported = False
        except Exception as e:
            # Error (posiblemente transitorio): caer a uno-por-uno SOLO este lote, sin desactivar.
            logger.warning(f"[Batch] Falló el embed en lote ({e}); proceso este lote uno-por-uno.")

    return _embed_items_one_by_one(items)
