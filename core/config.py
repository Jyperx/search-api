import os
import json

# --- CONFIGURACIONES GLOBALES Y ENTORNO ---
VOLUME_PATH = os.getenv('RAILWAY_VOLUME_MOUNT_PATH', '')
if VOLUME_PATH:
    SQLITE_DB = os.path.join(VOLUME_PATH, 'search_index.db')
else:
    SQLITE_DB = 'search_index.db'

EMBEDDING_MODEL = "models/gemini-embedding-2"
LLM_MODEL = "models/gemini-3.1-flash-lite"

# --- ESTADO GLOBAL Y CACHÉS EN MEMORIA ---
global_sync_state = {
    "is_syncing": False,
    "total_products": 0,
    "completed_products": 0,
    "status": "idle"
}

WEATHER_CACHE_STORE = {}

TIME_RULES_CACHE = [
    {"startHour": 5, "endHour": 10, "cluster": "desayuno", "scoreBoost": 5.0},
    {"startHour": 5, "endHour": 10, "cluster": "comida_rapida", "scoreBoost": -3.0},
    {"startHour": 11, "endHour": 14, "cluster": "almuerzo", "scoreBoost": 5.0},
    {"startHour": 18, "endHour": 23, "cluster": "comida_rapida", "scoreBoost": 3.0},
    {"startHour": 18, "endHour": 23, "cluster": "licores", "scoreBoost": 2.0},
]

MACRO_CLUSTERS_CACHE = {
    "desayuno": {
        "titles": ["Empieza el día con energía", "Mañanas deliciosas", "Despierta con sabor", "Para el desayuno"],
        "keywords": "desayuno OR arepa OR pan OR cafe OR huevos OR tamal OR calentao OR jugo OR tostada OR pandebono OR almojabana OR empanada OR buñuelo",
        "storeCategories": "Cafetería, Panaderia, Restaurante de desayunos, Desayunos",
        "negativeKeywords": "",
        "relatedClusters": "comida_rapida"
    },
    "almuerzo": {
        "titles": ["Hora de almorzar", "Almuerzos Ejecutivos", "Para el medio día", "Almuerzo Casero"],
        "keywords": "almuerzo OR corrientazo OR sopa OR arroz OR carne OR pollo OR principio OR bandeja OR menu OR ejecutivo",
        "storeCategories": "Restaurante, Comida Casera, Asadero, Almuerzos",
        "negativeKeywords": "desayuno OR pan OR cafe",
        "relatedClusters": "saludable"
    },
    "calor_dia": {
        "titles": ["Para este calorcito ☀️", "Días soleados", "Refresca tu tarde"],
        "keywords": "helado OR jugo OR paleta OR granizado OR frappe OR ensalada OR fruta OR refresco OR gafas OR pantaloneta OR camiseta OR bermuda OR sandalias",
        "storeCategories": "Heladería, Jugos, Ropa, Boutique",
        "negativeKeywords": "sopa OR tinto OR cafe OR caliente OR caldo OR chaqueta OR abrigo",
        "relatedClusters": "postres"
    },
    "calor_noche": {
        "titles": ["Noches cálidas", "Refréscate esta noche", "El calor no para", "Para compartir hoy"],
        "keywords": "helado OR cerveza OR licor OR coctel OR refresco OR frio OR hielo OR bebida",
        "storeCategories": "Heladería, Bar, Licorería",
        "negativeKeywords": "sopa OR tinto OR cafe OR caliente",
        "relatedClusters": "licores"
    },
    "frio_dia": {
        "titles": ["Días fríos 🌧️", "Acompáñalo con café", "Entra en calor"],
        "keywords": "cafe OR tinto OR sopa OR caldo OR chocolate OR empanada OR pan OR postre OR tamal OR changua OR chaqueta OR sueter OR bufanda",
        "storeCategories": "Cafetería, Panaderia, Restaurante, Ropa",
        "negativeKeywords": "helado OR hielo OR cerveza OR pantaloneta",
        "relatedClusters": "desayuno"
    },
    "frio_noche": {
        "titles": ["Noches frías 🌧️", "No salgas de casa", "Pide a domicilio", "Para el frío de hoy"],
        "keywords": "sopa OR caldo OR cobija OR saco OR chaqueta OR domicilio OR pizza OR hamburguesa",
        "storeCategories": "Restaurante, Hogar, Comida Rápida",
        "negativeKeywords": "helado OR jugo OR hielo",
        "relatedClusters": "comida_rapida"
    },
    "comida_rapida": {
        "titles": ["Antojos Rápidos", "Para calmar el hambre", "Tus favoritos"],
        "keywords": "hamburguesa OR pizza OR salchipapa OR frito OR alitas OR nuggets OR shawarma OR wrap OR combo",
        "storeCategories": "Restaurante, Comida Rápida, Hamburgueseria, Pizzeria",
        "negativeKeywords": "",
        "relatedClusters": "licores, saludable"
    },
    "saludable": {
        "titles": ["Cuida tu cuerpo", "Opciones Saludables", "Ligero y delicioso"],
        "keywords": "ensalada OR bowl OR saludable OR vegano OR vegetariano OR light OR dieta OR acai OR proteina OR organico",
        "storeCategories": "Restaurante Saludable, Jugos, Comida Saludable, Vegano",
        "negativeKeywords": "",
        "relatedClusters": "mercado"
    },
    "regalos": {
        "titles": ["Para esa persona especial", "Detalles que enamoran", "Sorpresas únicas"],
        "keywords": "regalo OR flor OR spa OR detalle OR aniversario OR peluche OR amor OR flores OR arreglo OR canasta OR bouquet",
        "storeCategories": "Regalería, Floristería, Spa, Detalles, Perfumeria",
        "negativeKeywords": "chocolate OR torta OR pastel OR cake OR pan",
        "relatedClusters": ""
    },
    "licores": {
        "titles": ["Para la fiesta", "Salud y celebración", "Prende la noche"],
        "keywords": "licor OR cerveza OR aguardiente OR ron OR vodka OR vino OR coctel OR fiesta OR hielo OR tequila OR whisky",
        "storeCategories": "Licorería, Bar, Distribuidora de Licores",
        "negativeKeywords": "",
        "relatedClusters": "comida_rapida, snacks"
    },
    "farmacia": {
        "titles": ["Cuida de tu salud", "Farmacia en casa", "Lo que necesitas, rápido"],
        "keywords": "farmacia OR medicamento OR pastilla OR dolor OR vitamina OR shampoo OR pañal OR crema OR jabon OR desodorante OR curitas OR antiseptico OR alcohol OR suero OR droga",
        "storeCategories": "Farmacia, Drogueía, Cuidado Personal, Salud, Supermercado",
        "negativeKeywords": "",
        "relatedClusters": ""
    },
    "hogar": {
        "titles": ["Mejora tu hogar", "Todo para tu casa", "Remodela tu espacio"],
        "keywords": "mueble OR herramienta OR pintura OR decoracion OR ferreteria OR destornillador OR bombillo OR taladro OR llave OR tornillo OR cable OR electricidad",
        "storeCategories": "Ferreteriía, Hogar, Materiales, Decoración",
        "negativeKeywords": "jabon OR shampoo OR crema OR pañal OR medicamento",
        "relatedClusters": ""
    },
    "mercado": {
        "titles": ["Directo a tu nevera", "Mercado fresco", "Llena tu despensa"],
        "keywords": "mercado OR carne OR verdura OR fruta OR lacteo OR viveres OR abarrotes OR huevo OR arroz OR aceite OR sal OR papa OR platano",
        "storeCategories": "Supermercado, Minimarket, Mercado, Carnicería, Fruver, Tienda",
        "negativeKeywords": "pollo asado OR asadero OR restaurante",
        "relatedClusters": "desayuno"
    },
    "mascotas": {
        "titles": ["Para el rey de la casa", "Mimos para tu peludo", "Cuidado animal"],
        "keywords": "mascota OR concentrado OR veterinaria OR pet OR pulgas OR collar OR juguete OR arena OR raza OR canino OR felino",
        "storeCategories": "Veterinaria, Tienda de Mascotas, Pet Shop",
        "negativeKeywords": "perro caliente OR hot dog OR salchicha",
        "relatedClusters": ""
    },
    "ropa": {
        "titles": ["Completa tu clóset", "Renueva tu estilo", "Moda recomendada"],
        "keywords": "ropa OR camisa OR pantalon OR zapato OR tenis OR moda OR accesorio OR reloj OR gafas OR vestido OR falda OR chaqueta OR sudadera",
        "storeCategories": "Ropa, Moda, Calzado, Boutique, Accesorios",
        "negativeKeywords": "",
        "relatedClusters": ""
    },
    "tecnologia": {
        "titles": ["Gadgets para tu vida", "Tecnología al instante", "Lo último en tech"],
        "keywords": "audifonos OR cargador OR cable OR funda OR celular OR tablet OR powerbank OR bluetooth OR usb OR memoria OR teclado OR mouse",
        "storeCategories": "Tecnología, Electrónicos, Celulares, Accesorios Tech",
        "negativeKeywords": "",
        "relatedClusters": ""
    },
    "postres": {
        "titles": ["Dulce tentación", "Antojos dulces", "El postre que mereces"],
        "keywords": "postre OR helado OR torta OR brownie OR cono OR malteada OR muffin OR cheesecake OR tiramisú OR flan OR crepe OR waffle",
        "storeCategories": "Heladería, Pastelería, Café, Postres",
        "negativeKeywords": "",
        "relatedClusters": "comida_rapida, licores"
    }
}

# --- MOTORES DE SINÓNIMOS ---
SYNONYMS = {
    # Comida Rápida y Restaurantes
    "hamburguesa": ["hamburguesa", "burger", "burguer", "hanburguesa"],
    "gaseosa": ["gaseosa", "coca", "coca-cola", "coca cola", "pepsi", "soda", "sprite", "postobon", "refresco", "bebida"],
    "pizza": ["pizza", "piza", "pissa"],
    "perro": ["perro", "hot dog", "hotdog", "salchicha", "hot-dog", "chori", "chorizo"],
    "pollo": ["pollo", "broaster", "asado", "alitas", "wings", "nuggets", "pechuga"],
    "papas": ["papas", "fritas", "francesa", "cascos", "salchipapa", "papa"],
    "helado": ["helado", "postre", "cono", "paleta", "sundae", "mcflurry", "brownie"],
    "cerveza": ["cerveza", "pola", "biela", "chela", "club colombia", "aguila", "poker", "corona", "heineken"],
    "jugo": ["jugo", "zumo", "batido", "licuado", "limonada", "jugos", "avena"],
    "carne": ["carne", "res", "churrasco", "parrilla", "asado", "picada", "cerdo", "chuzo"],
    "empanada": ["empanada", "pastel", "arepa", "pastelito", "dedito", "tequeno", "tequeño", "pandebono", "buñuelo"],
    "sushi": ["sushi", "maki", "roll", "sashimi", "nigiri"],
    "comida": ["comida", "almuerzo", "cena", "plato", "corrientazo", "ejecutivo", "menu", "restaurante", "sopa", "seco", "bandeja"],
    "almuerzo": ["almuerzo", "corrientazo", "sopa", "seco", "ejecutivo", "bandeja", "menu", "comida"],
    
    # Farmacia / Salud
    "pastillas": ["pastilla", "pildora", "tableta", "medicamento", "droga", "acetaminofen", "ibuprofeno", "aspirina", "dolex", "advil"],
    "jarabe": ["jarabe", "tos"],
    "preservativos": ["preservativo", "condon", "condones", "profilactico", "duo", "today"],
    "alcohol": ["alcohol", "antiseptico", "antibacterial", "desinfectante"],
    "panal": ["pañal", "panales", "pañales", "winny", "huggies", "pequeñin", "pañalitis"],
    "toallas": ["toalla", "toallas", "nosotras", "protectores", "tampones"],
    "crema": ["crema", "pomada", "unguento", "gel"],
    "suero": ["suero", "pedialyte", "electrolit"],
    
    # Ferretería / Hogar
    "taladro": ["taladro", "perforadora", "pulidora", "caladora"],
    "martillo": ["martillo", "mazo", "maceta", "alicate", "pinza", "hombre solo"],
    "destornillador": ["destornillador", "desatornillador", "estrella", "pala"],
    "bombillo": ["bombillo", "foco", "lampara", "luz", "bombilla", "led"],
    "pintura": ["pintura", "esmalte", "vinilo", "brocha", "rodillo", "aerosol", "thinner"],
    "clavos": ["clavo", "clavos", "puntilla", "tornillo", "chazo", "tuerca", "arandela"],
    "cinta": ["cinta", "pegante", "aislante", "enmascarar", "pegamento", "silicona", "boxer"],
    "tubo": ["tubo", "pvc", "tuberia", "codo", "accesorio", "soldadura"],
    "llave": ["llave", "candado", "cerradura", "cerrojo", "chapa"],
    "cable": ["cable", "alambre", "extension", "enchufe", "tomacorriente", "interruptor"],
    
    # Tecnología / Celulares
    "cargador": ["cargador", "cable", "adaptador", "fuente"],
    "audifonos": ["audifonos", "auriculares", "diadema", "airpods", "inpods", "earpods", "headset"],
    "celular": ["celular", "telefono", "smartphone", "iphone", "android", "movil", "xiaomi", "samsung", "motorola", "huawei"],
    "pantalla": ["pantalla", "display", "monitor", "tv", "televisor", "glass", "vidrio templado", "visor"],
    "bateria": ["bateria", "pila", "powerbank"],
    "regalo": ["regalo", "mama", "mamá", "madre", "cumpleaños", "aniversario", "floristeria", "flores", "spa", "chocolates", "detalle", "regalos"],
    "computador": ["computador", "pc", "laptop", "portatil", "computadora", "teclado", "mouse", "raton", "impresora"],
    "memoria": ["memoria", "usb", "microsd", "pendrive", "disco duro", "ssd"],
    "funda": ["funda", "estuche", "carcasa", "forro", "case", "protector"]
}

REVERSE_SYNONYMS = {}
for root, alts in SYNONYMS.items():
    for alt in alts:
        REVERSE_SYNONYMS[alt] = root

# --- LISTENERS Y CARGADORES DE FIREBASE ---
def on_algorithm_config_snapshot(doc_snapshot, changes, read_time):
    global MACRO_CLUSTERS_CACHE
    global TIME_RULES_CACHE
    for doc in doc_snapshot:
        data = doc.to_dict()
        if data:
            if "clusters" in data:
                MACRO_CLUSTERS_CACHE = data["clusters"]
            if "time_rules" in data:
                TIME_RULES_CACHE = data["time_rules"]
            print(f"🔥 Cerebro Híbrido V4.0 RAM Actualizado. Clústeres: {len(MACRO_CLUSTERS_CACHE)} | Reglas: {len(TIME_RULES_CACHE)}")

def load_synonyms_from_firestore():
    global SYNONYMS, REVERSE_SYNONYMS
    from database import db
    if not db: 
        return
    try:
        doc = db.collection('config').document('synonyms').get()
        if doc.exists:
            auto_syns = doc.to_dict().get('auto_synonyms', {})
            count = 0
            for root, alts in auto_syns.items():
                if root not in SYNONYMS:
                    SYNONYMS[root] = []
                for alt in alts:
                    if alt not in SYNONYMS[root]:
                        SYNONYMS[root].append(alt)
                        count += 1
            # Reconstruir reverse
            REVERSE_SYNONYMS.clear()
            for r, a_list in SYNONYMS.items():
                for a in a_list:
                    REVERSE_SYNONYMS[a] = r
            if count > 0:
                print(f"Cargados {count} sinónimos auto-aprendidos desde Firestore.")
    except Exception as e:
        print(f"Error cargando sinónimos dinámicos: {e}")
