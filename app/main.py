from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class TranslateRequest(BaseModel):
    text: str

@app.post("/translate")
async def translate(request: TranslateRequest):
    # 받은 JSON 데이터 처리
    translated_text = f"Translated: {request.text}"
    return {"translated_text": translated_text}
