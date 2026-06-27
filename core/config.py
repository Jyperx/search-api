import os
from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
FIREBASE_SERVICE_ACCOUNT = os.getenv("FIREBASE_SERVICE_ACCOUNT", "")

VOLUME_PATH = os.getenv("RAILWAY_VOLUME_MOUNT_PATH", "")
if VOLUME_PATH:
    SQLITE_DB = os.path.join(VOLUME_PATH, "search_index.db")
else:
    SQLITE_DB = "search_index.db"

OPEN_METEO_CACHE_SECONDS = int(os.getenv("OPEN_METEO_CACHE_SECONDS", "300"))

EMBEDDING_MODEL = "models/gemini-embedding-2"
LLM_MODEL = "models/gemini-3.1-flash-lite"

global_sync_state = {
    "is_syncing": False,
    "total_products": 0,
    "completed_products": 0,
    "status": "idle"
}
