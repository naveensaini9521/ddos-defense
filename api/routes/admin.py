from fastapi import APIRouter

router = APIRouter()


@router.post("/reload-model")
def reload_model():
    return {"ok": True}
