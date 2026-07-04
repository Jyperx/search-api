import logging
from fastapi import APIRouter
from pydantic import BaseModel
from typing import List, Optional
from core.database import get_db_connection, sqlite_lock

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/events", tags=["Events"])

class ActivityEvent(BaseModel):
    productId: Optional[str] = None
    type: str # 'view', 'cart', 'purchase', 'search', 'ignored', 'view_store'
    timestamp: str
    sectionId: Optional[str] = None
    storeId: Optional[str] = None   # presente en 'view_store' (alimenta el vector vía recent_activity)
    category: Optional[str] = None

class UserEventsRequest(BaseModel):
    activities: List[ActivityEvent]

@router.post("/{uid}")
def track_user_events(uid: str, req: UserEventsRequest):
    """
    Ingesta de eventos en batch.
    Invalida el caché de vectores del usuario para forzar recálculo.
    """
    if not req.activities:
        return {"status": "ok"}
        
    conn = None
    try:
        score_map = {'purchase': 5.0, 'like': 4.0, 'cart': 3.0, 'search': 2.0, 'click': 1.0, 'view': 1.0, 'view_product': 1.0, 'ignored': -0.5}
        with sqlite_lock:
            conn = get_db_connection()
            for act in req.activities:
                if act.productId and act.type in ('purchase', 'cart', 'click'):
                    conn.execute(
                        "INSERT INTO search_logs (query, clicked_id, clicked_category, result_count) VALUES (?, ?, ?, ?)",
                        ('', act.productId, '', 0)
                    )
                if act.productId:
                    conn.execute(
                        "INSERT OR REPLACE INTO user_activity_cache (user_id, product_id, activity_type, score, timestamp) VALUES (?, ?, ?, ?, ?)",
                        (uid, act.productId, act.type, score_map.get(act.type, 1.0), act.timestamp)
                    )
                if act.sectionId:
                    conn.execute(
                        "INSERT INTO section_stats (section_id, impressions, clicks) VALUES (?, 0, 1) "
                        "ON CONFLICT(section_id) DO UPDATE SET clicks = clicks + 1, updated_at = datetime('now')",
                        (act.sectionId,)
                    )
                # CTR/conversión por producto (señal aprendida + recompensa del bandit)
                if act.productId and act.type in ('view_product', 'click', 'cart', 'like'):
                    conn.execute(
                        "INSERT INTO item_stats (product_id, impressions, clicks, purchases) VALUES (?, 0, 1, 0) "
                        "ON CONFLICT(product_id) DO UPDATE SET clicks = clicks + 1, updated_at = datetime('now')",
                        (act.productId,)
                    )
                if act.productId and act.type == 'purchase':
                    conn.execute(
                        "INSERT INTO item_stats (product_id, impressions, clicks, purchases) VALUES (?, 0, 0, 1) "
                        "ON CONFLICT(product_id) DO UPDATE SET purchases = purchases + 1, updated_at = datetime('now')",
                        (act.productId,)
                    )
            conn.execute("DELETE FROM user_vectors WHERE user_id = ?", (uid,))
            conn.execute("DELETE FROM user_vector_meta WHERE user_id = ?", (uid,))
            conn.commit()
    except Exception as e:
        logger.error(f"Error processing events for {uid}: {e}")
    finally:
        if conn:
            conn.close()  # SIEMPRE cierra (evita conexiones colgadas → 'database is locked')

    return {"status": "ok", "message": "Events processed and cache invalidated"}
