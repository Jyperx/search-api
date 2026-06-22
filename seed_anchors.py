import os
import sqlite3
import sqlite_vec
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv("../admin/.env")

genai.configure(api_key=os.getenv("VITE_GEMINI_API_KEY", os.getenv("GOOGLE_API_KEY", "")))
EMBEDDING_MODEL = "models/gemini-embedding-001"

ANCHORS = [
    {"id": "A1", "title": "Gustos Culposos", "subtitle": "Para pecar sin remordimiento", "desc": "Comida rápida, hamburguesas grasosas, postres dulces, frituras, pizza, donas."},
    {"id": "A2", "title": "Cena Rápida", "subtitle": "Sin complicaciones", "desc": "Comida fácil de preparar o lista para comer en la noche, sándwiches, ensaladas ligeras, sushi, wraps."},
    {"id": "A3", "title": "Desayuno Energético", "subtitle": "Empieza el día con todo", "desc": "Café, huevos, pan, arepas, jugo de naranja, tostadas, tocino."},
]

def seed_anchors():
    conn = sqlite3.connect("search_index.db")
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    c = conn.cursor()
    
    for anchor in ANCHORS:
        res = genai.embed_content(model=EMBEDDING_MODEL, content=anchor["desc"], task_type="retrieval_document")
        vec_bytes = sqlite_vec.serialize_float32(res['embedding'])
        
        c.execute("INSERT OR REPLACE INTO anchor_metadata (anchor_id, title, subtitle, section_type) VALUES (?, ?, ?, ?)",
                  (anchor["id"], anchor["title"], anchor["subtitle"], "generative"))
        c.execute("INSERT OR REPLACE INTO anchor_vectors (anchor_id, embedding) VALUES (?, ?)",
                  (anchor["id"], vec_bytes))
        print(f"Ancla {anchor['title']} vectorizada.")
        
    conn.commit()
    conn.close()

if __name__ == "__main__":
    seed_anchors()
