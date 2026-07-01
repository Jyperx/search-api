"""Capa de CURACIÓN humana (relevance feedback).

Permite corregir a mano los resultados del motor sin tocar el auto-aprendizaje:
- kind 'concept': "este producto NO va en la sección de tal concepto" (home feed).
- kind 'query':   "este producto NO es relevante para tal búsqueda".

Cada corrección tiene una acción:
- 'exclude': el producto no vuelve a aparecer ahí.
- 'demote':  el producto aparece menos / más abajo (no desaparece del todo).

Se persiste en Firestore (config/curation) para sobrevivir a los redeploys, y se
mantiene una copia en memoria para lookups O(1) en cada petición.
"""

# CURATION[kind][key][product_id] = 'exclude' | 'demote'
CURATION = {"concept": {}, "query": {}}

DEMOTE_PENALTY = 0.7  # cuánto se le resta al score en el home cuando es 'demote'


def _norm(key: str) -> str:
    return (key or "").strip().lower()


def load_curation(db) -> None:
    """Carga las correcciones desde Firestore a memoria."""
    CURATION["concept"].clear()
    CURATION["query"].clear()
    if not db:
        return
    try:
        doc = db.collection("config").document("curation").get()
        if not doc.exists:
            return
        data = doc.to_dict() or {}
        for kind in ("concept", "query"):
            for entry in data.get(kind, []) or []:
                key = _norm(entry.get("key"))
                pid = entry.get("product_id")
                if key and pid:
                    CURATION[kind].setdefault(key, {})[pid] = entry.get("action", "exclude")
        total = sum(len(v) for k in CURATION for v in CURATION[k].values())
        print(f"[Curación] {total} correcciones cargadas.")
    except Exception as e:
        print(f"[Curación] No se pudieron cargar: {e}")


def _persist(db) -> None:
    if not db:
        return
    out = {"concept": [], "query": []}
    for kind in ("concept", "query"):
        for key, prods in CURATION[kind].items():
            for pid, action in prods.items():
                out[kind].append({"key": key, "product_id": pid, "action": action})
    try:
        db.collection("config").document("curation").set(out)
    except Exception as e:
        print(f"[Curación] Error persistiendo: {e}")


def set_curation(db, kind: str, key: str, product_id: str, action: str = "exclude") -> None:
    if kind not in CURATION:
        raise ValueError("kind inválido")
    if action not in ("exclude", "demote"):
        action = "exclude"
    CURATION[kind].setdefault(_norm(key), {})[product_id] = action
    _persist(db)


def delete_curation(db, kind: str, key: str, product_id: str) -> None:
    key = _norm(key)
    if kind in CURATION and key in CURATION[kind]:
        CURATION[kind][key].pop(product_id, None)
        if not CURATION[kind][key]:
            CURATION[kind].pop(key, None)
    _persist(db)


def curation_action(kind: str, key: str, product_id: str):
    """Devuelve 'exclude' | 'demote' | None para un producto en un concepto/query."""
    return CURATION.get(kind, {}).get(_norm(key), {}).get(product_id)
