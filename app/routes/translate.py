from fastapi import APIRouter
from app.services.openai_service import generate_openai_response

router = APIRouter()

@router.post("/openai")
async def openai_translate(prompt: str):
    return {"response": generate_openai_response(prompt)}