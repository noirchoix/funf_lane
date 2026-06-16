from fastapi import APIRouter

router = APIRouter(tags=["health"])

@router.get("/healthz")
def healthz():
    return {"status": "ok"}

@router.get("/readyz")
def readyz():
    # later you can add checks: qdrant reachable, collections exist, etc.
    return {"status": "ready"}