"""Cliente centralizado de Gemini usando el paquete nuevo `google-genai`.

Reemplaza al paquete deprecado `google.generativeai`. Expone dos helpers:
- embed_text: genera embeddings truncados a 768 dims (formato Matryoshka) para las tablas vec0.
- generate_text: genera texto con un modelo generativo.
"""
import time
import logging
from typing import Optional, List
from google import genai
from google.genai import types
from core.config import GOOGLE_API_KEY, EMBEDDING_MODEL

logger = logging.getLogger(__name__)

client = genai.Client(api_key=GOOGLE_API_KEY)


def _with_retry(fn, attempts: int = 3, base_delay: float = 0.6):
    """Reintenta una llamada a Gemini con backoff exponencial (ayuda con rate-limits transitorios).
    Reintenta sobre todo cuando el error parece de cuota/rate-limit; en el último intento, propaga."""
    last_exc = None
    for i in range(attempts):
        try:
            return fn()
        except Exception as e:
            last_exc = e
            msg = str(e).lower()
            transient = any(k in msg for k in ("429", "resource", "exhaust", "rate", "quota", "unavailable", "503", "deadline", "timeout"))
            if i == attempts - 1 or not transient:
                raise
            time.sleep(base_delay * (2 ** i))  # 0.6s, 1.2s, ...
    raise last_exc


def embed_text(text: str, task_type: Optional[str] = None, model: Optional[str] = None) -> list:
    """Genera un embedding y lo trunca a 768 dims para encajar en las tablas vectoriales."""
    # El SDK nuevo valida task_type en mayúsculas (RETRIEVAL_QUERY, RETRIEVAL_DOCUMENT, ...)
    config = types.EmbedContentConfig(task_type=task_type.upper()) if task_type else None
    res = _with_retry(lambda: client.models.embed_content(
        model=model or EMBEDDING_MODEL,
        contents=text,
        config=config,
    ))
    return list(res.embeddings[0].values)[:768]


def embed_texts(texts: List[str], task_type: Optional[str] = None, model: Optional[str] = None) -> List[list]:
    """Embebe VARIOS textos en UNA sola llamada (batchEmbedContents). Devuelve una lista de
    vectores (768 dims), alineada 1:1 con `texts`.

    IMPORTANTE: con gemini-embedding-2, pasar una lista de STRINGS planos los FUSIONA en un solo
    vector. Para obtener un embedding por texto hay que enviar cada uno como un Content separado;
    así el SDK usa batchEmbedContents y devuelve N embeddings."""
    if not texts:
        return []
    config = types.EmbedContentConfig(task_type=task_type.upper()) if task_type else None
    contents = [types.Content(parts=[types.Part(text=t)]) for t in texts]
    res = _with_retry(lambda: client.models.embed_content(
        model=model or EMBEDDING_MODEL,
        contents=contents,
        config=config,
    ))
    return [list(e.values)[:768] for e in res.embeddings]


def generate_text(prompt: str, model: str, temperature: Optional[float] = None,
                  max_output_tokens: Optional[int] = None) -> str:
    """Genera texto con un modelo generativo. Devuelve el string de respuesta (o vacío)."""
    config = None
    if temperature is not None or max_output_tokens is not None:
        config = types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )
    res = client.models.generate_content(model=model, contents=prompt, config=config)
    return res.text or ""
