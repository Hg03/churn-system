from pydantic import BaseModel
from typing import Literal, Optional

class RawDataValidator(BaseModel):
    user_id: Optional[str]
    subscription_type: Optional[Literal['Free', 'Premium']]
    country: Optional[str]
    avg_daily_minutes: Optional[float]
    number_of_playlists: Optional[int]
    top_genre: Optional[str]
    skips_per_day: Optional[int]
    support_tickets: Optional[int]
    days_since_last_login: Optional[int]
    churned: Optional[int]