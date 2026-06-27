import logging
from fastapi import APIRouter
from pydantic import BaseModel
from typing import List, Optional
from core.database import get_db_connection, sqlite_lock

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/events", tags=["Events"])

class ActivityEvent(BaseModel):
    productId: Optional[str] = None
    type: str # 'view', 'cart', 'purchase', 'search', 'ignored'
    timestamp: str
    sectionId: Optional[str] = None

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
        
    try:
        score_map = {'purchase': 5.0, 'cart': 3.0, 'search': 2.0, 'click': 1.0, 'view': 1.0, 'view_product': 1.0, 'ignored': -0.5}
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
                        "INSERT INTO section_impressions (section_id, user_id, clicked) VALUES (?, ?, 1)",
                        (act.sectionId, uid)
                    )
            conn.execute("DELETE FROM user_vectors WHERE user_id = ?", (uid,))
            conn.execute("DELETE FROM user_vector_meta WHERE user_id = ?", (uid,))
            conn.commit()
            conn.close()
    except Exception as e:
        logger.error(f"Error processing events for {uid}: {e}")
        
    return {"status": "ok", "message": "Events processed and cache invalidated"}
