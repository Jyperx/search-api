from fastapi import FastAPI
from contextlib import asynccontextmanager
import asyncio

from core.database import init_db
from core.firebase import init_firebase, listen_config
from data.clusters import MACRO_CLUSTERS_CACHE, TIME_RULES_CACHE
from data.synonyms import load_synonyms_from_firestore
from data.concepts import cargar_conceptos_en_memoria
import core.firebase
from fastapi.middleware.cors import CORSMiddleware

from routers.home import router as home_router
from routers.events import router as events_router
from routers.search import router as search_router
from routers.admin import router as admin_router

# ── Elección de worker LÍDER para tareas de fondo ──────────────────────────────
# Con `uvicorn --workers N`, el lifespan corre en CADA worker → el scheduler se
# ejecutaría N veces (retry embeddings, reconcile, etc. duplicados). Un flock elige UN
# único líder entre los workers del mismo contenedor: solo ese corre las tareas de fondo.
# Se adquiere DENTRO del lifespan (cada worker abre su propio descriptor) para que un
# posible fork de uvicorn no herede el lock y crea que todos son líderes.
_leader_lock_fh = None  # se conserva abierto para mantener el lock vivo

def _try_become_leader() -> bool:
    global _leader_lock_fh
    try:
        import fcntl
        _leader_lock_fh = open("/tmp/punto_scheduler.lock", "w")
        fcntl.flock(_leader_lock_fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return True
    except Exception:
        return False

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 1. Inicializar base de datos SQLite y tablas vectoriales
    init_db()
    
    # 2. Inicializar Firebase
    init_firebase()
    
    # 3. Arrancar listener de config pasándole las variables a mutar
    listen_config(MACRO_CLUSTERS_CACHE, TIME_RULES_CACHE)
    
    # 4. Cargar sinónimos y conceptos
    load_synonyms_from_firestore(core.firebase.db)
    from data.concepts import load_concept_texts_from_firestore
    load_concept_texts_from_firestore(core.firebase.db)
    cargar_conceptos_en_memoria()
    from services.context_engine import load_ranking_weights
    load_ranking_weights(core.firebase.db)
    from routers.home import load_section_titles
    load_section_titles(core.firebase.db)
    from data.curation import load_curation
    load_curation(core.firebase.db)

    # 4.1 Construir conceptos ambientales si faltan (necesarios para el peso clima/hora).
    # Es lo ÚNICO lento del arranque (llama al LLM) → se hace en SEGUNDO PLANO para no
    # bloquear el servidor. Mientras se construye, el ranking usa peso de clima/hora neutro
    # (resultados válidos, solo un poco menos afinados) hasta que termine (una vez).
    from data.concepts import DICCIONARIO_CONCEPTOS, _async_build_concept_dictionary
    if not DICCIONARIO_CONCEPTOS:
        async def _build_concepts_bg():
            try:
                print("[Conceptos] Vacío. Construyendo en segundo plano...")
                await _async_build_concept_dictionary()
                cargar_conceptos_en_memoria()
                print("[Conceptos] Diccionario construido y cargado.")
            except Exception as e:
                print(f"[Conceptos] No se pudieron construir: {e}")
        asyncio.create_task(_build_concepts_bg())

    # 5. Scheduler: tareas automáticas en background (solo el worker líder lo ARRANCA).
    is_leader = _try_become_leader()
    try:
        from apscheduler.schedulers.background import BackgroundScheduler
        from services.sync import retry_vector_queue_task, reconcile_catalog
        from data.synonyms import learn_synonyms_from_clicks
        from data.clusters import learn_clusters_from_catalog
        from services.context_engine import tune_ranking_weights

        def _auto_tune_weights():
            try:
                tune_ranking_weights(core.firebase.db)
            except Exception as e:
                print(f"[Scheduler] Error ajustando pesos: {e}")

        def _auto_learn_synonyms():
            try:
                learned = learn_synonyms_from_clicks(core.firebase.db)
                if learned:
                    print(f"[Scheduler] Auto-aprendidos {len(learned)} grupos de sinónimos por co-clics.")
            except Exception as e:
                print(f"[Scheduler] Error aprendiendo sinónimos: {e}")

        def _auto_learn_clusters():
            try:
                learned = learn_clusters_from_catalog(core.firebase.db)
                if learned:
                    print(f"[Scheduler] Auto-aprendidos/actualizados {len(learned)} clusters por TF-IDF.")
            except Exception as e:
                print(f"[Scheduler] Error aprendiendo clusters: {e}")

        scheduler = BackgroundScheduler(timezone="UTC")
        # Reintentar embeddings fallidos cada 10 min (rescata productos sin vectorizar)
        scheduler.add_job(retry_vector_queue_task, 'interval', minutes=10, id='retry_vectors', replace_existing=True)
        # Aprender sinónimos por co-clics cada 6 horas
        scheduler.add_job(_auto_learn_synonyms, 'interval', hours=6, id='learn_synonyms', replace_existing=True)
        # Aprender/actualizar clusters por TF-IDF del catálogo cada 24 horas
        scheduler.add_job(_auto_learn_clusters, 'interval', hours=24, id='learn_clusters', replace_existing=True)
        # Auto-ajustar pesos del ranking cada 12 horas
        scheduler.add_job(_auto_tune_weights, 'interval', hours=12, id='tune_weights', replace_existing=True)
        # Reconciliar catálogo (quitar fantasmas) cada 24h. Lee TODO el catálogo de Firestore,
        # así que se hace 1 vez/día para minimizar lecturas (no es urgente: el borrado en la app
        # ya limpia su propio fantasma al instante).
        scheduler.add_job(reconcile_catalog, 'interval', hours=24, id='reconcile_catalog', replace_existing=True)
        # Solo el worker LÍDER arranca el scheduler → evita tareas duplicadas con --workers N.
        if is_leader:
            scheduler.start()
            app.state.scheduler = scheduler
            print("[Scheduler] Iniciado (worker líder): vectores (10 min) + sinónimos (6 h) + clusters (24 h) + pesos (12 h) + reconcile (24 h).")
        else:
            print("[Scheduler] Worker secundario: no arranca tareas de fondo (ya las corre el líder).")
    except Exception as e:
        print(f"[Scheduler] No se pudo iniciar: {e}")

    yield

    # Shutdown
    try:
        if getattr(app.state, "scheduler", None):
            app.state.scheduler.shutdown(wait=False)
    except Exception:
        pass

app = FastAPI(title="Punto Search API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(home_router)
app.include_router(events_router)
app.include_router(search_router)
app.include_router(admin_router)

@app.get("/")
def read_root():
    return {"status": "ok", "service": "search-api"}
