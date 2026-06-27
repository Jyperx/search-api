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

def load_synonyms_from_firestore(db):
    if not db: 
        return
    try:
        doc = db.collection('config').document('synonyms').get()
        if doc.exists:
            auto_syns = doc.to_dict().get('auto_synonyms', {})
            for root, alts in auto_syns.items():
                # Sobrescribir o añadir
                SYNONYMS[root] = [a.lower().strip() for a in alts]
                for a in alts:
                    REVERSE_SYNONYMS[a.lower().strip()] = root
        print(f"Cargados {len(SYNONYMS)} sinónimos (base + auto)")
    except Exception as e:
        print(f"Error cargando sinónimos: {e}")
