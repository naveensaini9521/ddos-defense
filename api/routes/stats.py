from fastapi import APIRouter

router = APIRouter()


@router.get("/")
def stats():
    return {"rps": 0, "blocked": 0}
