from fastapi import APIRouter

router = APIRouter()  # APIRouter 객체 생성

@router.post("/")
async def handle_openai_request(prompt: str):
    return {"response": f"OpenAI received: {prompt}"}