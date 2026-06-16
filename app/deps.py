from fastapi import Header, HTTPException
from .settings import settings

def require_service_key(x_api_key: str | None = Header(default=None)) -> None:
    if not x_api_key or x_api_key != settings.SERVICE_API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")
