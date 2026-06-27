import numpy as np
import asyncio
import google.generativeai as genai
from core.config import GOOGLE_API_KEY, EMBEDDING_MODEL
from core.database import get_db_connection

genai.configure(api_key=GOOGLE_API_KEY)

DICCIONARIO_CONCEPTOS = {}

CATEGORY_WEIGHTS = {
    "restaurante": 0.20,
    "comida rapida": 0.20,
    "heladeria": 0.25,    
    "cafeteria": 0.18,
    "licoreria": 0.15,
    "panaderia": 0.18,
    "jugos": 0.22,
    "postres": 0.20,
    "ropa": 0.12,
    "boutique": 0.12,
    "spa": 0.10,
    "tecnologia": 0.0,
    "ferreteria": 0.0,
    "farmacia": 0.0,
    "mascotas": 0.0,
    "hogar": 0.0,
    "default": 0.10,      
}

CONCEPTOS_SEMILLA = {
    "ENV_CALOR": (
        "Día soleado, mucho calor, sol picante, sed, bochorno, caluroso. "
        "Bebidas frías, helados, jugos naturales, paletas, granizados, gaseosa fría, limonada, "
        "cerveza fría, refrescos, ensalada de frutas. Ropa fresca, pantalonetas, gafas de sol, sandalias."
    ),
    "ENV_FRIO": (
        "Día lluvioso, clima frío, nublado, aguacero, llovizna, fresco. "
        "Cosas calientes para entrar en calor: café, tinto, chocolate caliente, caldo de costilla, "
        "sopa, changua, empanadas recién hechas, tamales, pan, panadería, buñuelos. "
        "Cobijas, chaquetas, suéter, domicilios para no mojarse."
    ),
    "ENV_NOCHE": (
        "Noche, rumba, fin de semana, celebración, amigos, fiesta, madrugada. "
        "Licores, aguardiente, ron, cerveza, cócteles, hielo, pasabocas, "
        "comida rápida para la madrugada, hamburguesas, salchipapas, pizza."
    ),
    "ENV_MANANA": (
        "Mañana, amanecer, despertar, empezar el día, energía, desayuno. "
        "Café, tinto, arepa, huevos, pan, tamal, calentao, jugo de naranja, "
        "almojábana, pandebono, buñuelo."
    ),
    "ENV_MEDIODIA": (
        "Mediodía, almuerzo, hambre fuerte, descanso del trabajo, corrientazo. "
        "Almuerzo ejecutivo, sopa y seco, bandeja paisa, carne, pollo asado, principio, arroz."
    ),
    "ENV_SALUDABLE": (
        "Dieta, gimnasio, cuidar la figura, entrenamiento, fit, sano. "
        "Ensaladas, bowls, vegano, vegetariano, light, batidos de proteína, frutas, orgánico, acaí."
    ),
    "ENV_GUAYABO": (
        "Resaca, guayabo, dolor de cabeza, malestar, deshidratado, mucha sed, cansancio. "
        "Caldo de costilla, suero oral, pedialyte, pastillas para el dolor, "
        "bebidas frías, comida grasosa reconfortante."
    ),
    "ENV_PEREZA": (
        "Domingo, pereza, quedarse en casa, no quiero cocinar, maratón de series, "
        "domicilio, no salir. Pizza, hamburguesa, comida reconfortante, helado, snacks."
    ),
}

def cargar_conceptos_en_memoria():
    conn = get_db_connection()
    try:
        rows = conn.execute("SELECT id, embedding FROM concept_vectors").fetchall()
        for row in rows:
            DICCIONARIO_CONCEPTOS[row['id']] = np.frombuffer(row['embedding'], dtype=np.float32)
        print(f"Loaded {len(DICCIONARIO_CONCEPTOS)} concepts into memory.")
    except Exception as e:
        print(f"No se pudieron cargar conceptos en memoria (quizás falte correr build_concept_dictionary): {e}")
    finally:
        conn.close()

async def _async_build_concept_dictionary():
    print("Building concept dictionary...")
    conn = get_db_connection()
    
    for concept_id, description in CONCEPTOS_SEMILLA.items():
        try:
            res = genai.embed_content(
                model=EMBEDDING_MODEL,
                content=description
            )
            embedding = np.array(res['embedding'], dtype=np.float32)
            embedding = embedding / np.linalg.norm(embedding)
            
            conn.execute(
                "INSERT OR REPLACE INTO concept_vectors (id, embedding) VALUES (?, ?)",
                (concept_id, embedding.tobytes())
            )
            print(f"Successfully generated embedding for {concept_id}")
        except Exception as e:
            print(f"Error generating embedding for {concept_id}: {e}")
            
    conn.commit()
    conn.close()
    print("Concept dictionary generation complete.")

def build_concept_dictionary():
    """Ejecutar solo una vez o por crontab para generar DB de conceptos"""
    asyncio.run(_async_build_concept_dictionary())
