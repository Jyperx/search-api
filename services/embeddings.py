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


def _anchor_vector(vector_base: np.ndarray, category: str):
    """Aplica anclaje a concepto si corresponde. Devuelve (bytes, source). Sin LLM."""
    cat_key = (category or '').lower().strip()
    anchor_weight = CATEGORY_WEIGHTS.get(cat_key, CATEGORY_WEIGHTS["default"])
    if anchor_weight == 0.0 or not DICCIONARIO_CONCEPTOS:
        return _serialize(vector_base), 'raw_only'
    best_sim, best_anchor_id = _find_closest_concept(vector_base)
    if best_sim >= 0.62:
        anchor_vec = DICCIONARIO_CONCEPTOS[best_anchor_id]
        vector_final = (vector_base * (1 - anchor_weight)) + (anchor_vec * anchor_weight)
        vector_final = vector_final / np.linalg.norm(vector_final)
        return _serialize(vector_final), 'concept_anchoring'
    # En batch NO llamamos al LLM (esa rama queda para el path de crear/editar 1 producto).
    return _serialize(vector_base), 'raw_only'


def generate_product_embeddings_batch(items: list) -> list:
    """Vectoriza una TANDA de productos con UNA sola llamada de embeddings (raw en lote) + anclaje a
    concepto por producto. Mucho más rápido para sync masivo. Omite el enriquecimiento Flash-Lite.

    items: lista de (product_id, name, category, description).
    Devuelve: lista de (product_id, bytes|None, source).
    """
    if not items:
        return []
    raw_texts = [f"Producto: {n}. Categoría: {c}. Descripción: {d}" for (_id, n, c, d) in items]
    try:
        raw_vectors = embed_texts(raw_texts, task_type="retrieval_document")
    except Exception as e:
        logger.warning(f"[Batch] Falló el embed en lote ({len(items)} items): {e}")
        return [(it[0], None, 'error') for it in items]

    out = []
    for item, raw in zip(items, raw_vectors):
        pid, _name, category, _desc = item
        try:
            v = np.array(raw, dtype=np.float32)
            norm = np.linalg.norm(v)
            vector_base = v / norm if norm > 0 else v
            blob, source = _anchor_vector(vector_base, category)
            out.append((pid, blob, source))
        except Exception as e:
            logger.warning(f"[Batch] Error procesando {pid}: {e}")
            out.append((pid, None, 'error'))
    return out
