import os
import threading
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from database import init_db, db
from core.config import load_synonyms_from_firestore, on_algorithm_config_snapshot, MACRO_CLUSTERS_CACHE
from background_jobs import delta_sync_loop, cleanup_activity_loop, on_stores_snapshot

from routers.home import router as home_router
from routers.search import router as search_router
from routers.admin import router as admin_router

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Search backend is ready. Iniciando procesos en segundo plano...")
    
    # Inicializar la base de datos (SQLite + VEC)
    init_db()
    
    # Cargar sinónimos desde Firestore
    load_synonyms_from_firestore()
    
    # Iniciar listeners de Firestore si la DB está configurada
    if db:
        doc_ref = db.collection('config').document('algorithm')
        doc_snap = doc_ref.get()
        if not doc_snap.exists:
            doc_ref.set({"clusters": MACRO_CLUSTERS_CACHE})
        doc_ref.on_snapshot(on_algorithm_config_snapshot)
        
        # Listener de comercios para actualizaciones en tiempo real
        db.collection("stores").on_snapshot(on_stores_snapshot)

    # Iniciar procesos pesados (Delta Sync y Cleanup) en hilos separados
    threading.Thread(target=delta_sync_loop, daemon=True).start()
    threading.Thread(target=cleanup_activity_loop, daemon=True).start()
    
    yield
    print("Apagando procesos...")

app = FastAPI(title="Punto Search Engine (Mini-Algolia) Modular", lifespan=lifespan)

# Habilitar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Registrar routers
app.include_router(home_router)
app.include_router(search_router)
app.include_router(admin_router)

@app.get("/")
def read_root():
    return {"message": "Punto Search Engine V4.0 Modular is running", "status": "ok"}
