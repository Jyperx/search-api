SYNONYMS = {
    "cerveza": ["pola", "biela", "chela", "fría"],
    "hamburguesa": ["burger", "burguer"],
    "perro caliente": ["perro", "hot dog", "hotdog"],
    "gaseosa": ["refresco", "soda"],
    "tinto": ["cafe", "cafecito", "tintico"],
    "aguardiente": ["guaro", "chorro"],
    "panaderia": ["pan", "pandebono", "buñuelo", "almojabana"]
}
REVERSE_SYNONYMS = {}

# Inicializar REVERSE_SYNONYMS base
for root, alts in SYNONYMS.items():
    for a in alts:
        REVERSE_SYNONYMS[a.lower().strip()] = root

def _rebuild_reverse():
    REVERSE_SYNONYMS.clear()
    for root, alts in SYNONYMS.items():
        for a in alts:
            REVERSE_SYNONYMS[a.lower().strip()] = root


def load_synonyms_from_firestore(db):
    if not db:
        return
    try:
        doc = db.collection('config').document('synonyms').get()
        if doc.exists:
            auto_syns = doc.to_dict().get('auto_synonyms', {})
            if auto_syns:
                # Firestore es la fuente de verdad una vez editado (permite borrar base)
                SYNONYMS.clear()
                for root, alts in auto_syns.items():
                    SYNONYMS[root.lower().strip()] = [a.lower().strip() for a in alts]
                _rebuild_reverse()
        print(f"Cargados {len(SYNONYMS)} sinónimos (base + auto)")
    except Exception as e:
        print(f"Error cargando sinónimos: {e}")


def _persist(db):
    if db:
        try:
            db.collection('config').document('synonyms').set({"auto_synonyms": SYNONYMS}, merge=False)
        except Exception as e:
            print(f"Error persistiendo sinónimos: {e}")


def set_synonym_group(db, root, alternatives):
    """Crea/actualiza un grupo de sinónimos y lo persiste."""
    root = (root or "").lower().strip()
    alts = sorted({a.lower().strip() for a in alternatives if a and a.strip() and a.lower().strip() != root})
    if not root or not alts:
        return SYNONYMS
    SYNONYMS[root] = alts
    _rebuild_reverse()
    _persist(db)
    return SYNONYMS


def delete_synonym_group(db, root):
    """Elimina un grupo de sinónimos."""
    SYNONYMS.pop((root or "").lower().strip(), None)
    _rebuild_reverse()
    _persist(db)
    return SYNONYMS


def learn_synonyms_from_clicks(db, min_support=3, jaccard_threshold=0.25, max_groups=40):
    """Aprende sinónimos por CO-CLICS (puro comportamiento, sin IA).

    Si dos búsquedas de una sola palabra llevan a clics en los mismos productos,
    son candidatas a sinónimos. Se mide con índice de Jaccard sobre los productos clicados.
    """
    from collections import defaultdict
    from core.database import get_db_connection

    conn = get_db_connection()
    try:
        rows = conn.execute(
            "SELECT query, clicked_id FROM search_logs WHERE query != '' AND clicked_id IS NOT NULL AND clicked_id != ''"
        ).fetchall()
    finally:
        conn.close()

    q_products = defaultdict(set)
    q_count = defaultdict(int)
    for r in rows:
        q = (r["query"] or "").strip().lower()
        if not q or " " in q:  # solo términos de una palabra
            continue
        q_products[q].add(r["clicked_id"])
        q_count[q] += 1

    terms = [q for q in q_products if q_count[q] >= min_support and len(q_products[q]) >= 2]

    # Pares con Jaccard alto
    parent = {t: t for t in terms}
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(a, b):
        parent[find(a)] = find(b)

    pairs_found = 0
    for i in range(len(terms)):
        for j in range(i + 1, len(terms)):
            a, b = terms[i], terms[j]
            sa, sb = q_products[a], q_products[b]
            inter = len(sa & sb)
            if inter == 0:
                continue
            jacc = inter / len(sa | sb)
            if jacc >= jaccard_threshold:
                union(a, b)
                pairs_found += 1

    # Agrupar por componente; root = término más frecuente
    groups = defaultdict(list)
    for t in terms:
        groups[find(t)].append(t)

    learned = {}
    for members in groups.values():
        if len(members) < 2:
            continue
        members.sort(key=lambda t: q_count[t], reverse=True)
        root, alts = members[0], members[1:]
        learned[root] = alts
        if len(learned) >= max_groups:
            break

    # Fusionar con lo existente y persistir
    for root, alts in learned.items():
        existing = set(SYNONYMS.get(root, []))
        SYNONYMS[root] = sorted(existing | set(alts))
    _rebuild_reverse()
    _persist(db)
    return learned
