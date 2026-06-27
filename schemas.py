from pydantic import BaseModel
from typing import List, Optional

class HomeFeedRequest(BaseModel):
    activities: List[dict] = []
    lat: float = None
    lng: float = None
    override_hour: int = None
    override_weather_temp: float = None
    override_weather_code: int = None

class ManualAnchorRequest(BaseModel):
    title: str
    subtitle: str
    desc: str
    allowed_categories: List[str] = []
    exclude_rules: List[str] = []
    titles: List[str] = []

class SimulateRequest(BaseModel):
    prompt: str

class SearchClickPayload(BaseModel):
    query: str
    clicked_id: str
    clicked_category: Optional[str] = ""
    result_count: Optional[int] = 0

class ProductPayload(BaseModel):
    id: str
    storeId: str
    name: str
    category: Optional[str] = ""
    description: Optional[str] = ""
    price: Optional[float] = 0
    icon: Optional[str] = ""
    imageUrl: Optional[str] = ""
    isOpen: Optional[bool] = True
    onSale: Optional[bool] = False
    salePrice: Optional[float] = None
    likes: Optional[int] = 0
    views: Optional[int] = 0
    purchases: Optional[int] = 0
    available: Optional[bool] = True

class StorePayload(BaseModel):
    id: str
    name: str
    category: Optional[str] = ""
    imageUrl: Optional[str] = ""
    isOpen: Optional[bool] = True
