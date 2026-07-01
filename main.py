from fastapi import FastAPI
from contextlib import asynccontextmanager

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

    # 4.1 Construir conceptos ambientales si faltan (necesarios para el peso clima/hora)
    from data.concepts import DICCIONARIO_CONCEPTOS, _async_build_concept_dictionary
    if not DICCIONARIO_CONCEPTOS:
        try:
            print("[Conceptos] Vacío. Construyendo diccionario de conceptos ambientales...")
            await _async_build_concept_dictionary()
            cargar_conceptos_en_memoria()
        except Exception as e:
            print(f"[Conceptos] No se pudieron construir al inicio: {e}")

    # 5. Scheduler: tareas automáticas en background
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
        scheduler.start()
        app.state.scheduler = scheduler
        print("[Scheduler] Iniciado: vectores (10 min) + sinónimos (6 h) + clusters (24 h) + pesos (12 h) + reconcile (6 h).")
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
