"""Cliente centralizado de Gemini usando el paquete nuevo `google-genai`.

Reemplaza al paquete deprecado `google.generativeai`. Expone dos helpers:
- embed_text: genera embeddings truncados a 768 dims (formato Matryoshka) para las tablas vec0.
- generate_text: genera texto con un modelo generativo.
"""
import logging
from typing import Optional
from google import genai
from google.genai import types
from core.config import GOOGLE_API_KEY, EMBEDDING_MODEL

logger = logging.getLogger(__name__)

client = genai.Client(api_key=GOOGLE_API_KEY)


def embed_text(text: str, task_type: Optional[str] = None, model: Optional[str] = None) -> list:
    """Genera un embedding y lo trunca a 768 dims para encajar en las tablas vectoriales."""
    # El SDK nuevo valida task_type en mayúsculas (RETRIEVAL_QUERY, RETRIEVAL_DOCUMENT, ...)
    config = types.EmbedContentConfig(task_type=task_type.upper()) if task_type else None
    res = client.models.embed_content(
        model=model or EMBEDDING_MODEL,
        contents=text,
        config=config,
    )
    return list(res.embeddings[0].values)[:768]


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
