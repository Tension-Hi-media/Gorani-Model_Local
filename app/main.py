from fastapi import FastAPI
from app.models.schemas import TranslateRequest


app = FastAPI()


@app.post("/translate")
async def translate(request: TranslateRequest):
    # 받은 JSON 데이터 처리
    translated_text = f"Translated: {request.text}"
    return {"translated_text": translated_text}
