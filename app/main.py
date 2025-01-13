from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from models.translation import setup_translation_chain
import logging

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 필요에 따라 특정 도메인으로 제한 가능
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TranslateRequest(BaseModel):
    text: str

class TranslateResponse(BaseModel):
    answer: str

@app.post("/translate")
async def translate(request: TranslateRequest):
    # 받은 JSON 데이터 처리
    translated_text = f"Translated: {request.text}"
    return {"translated_text": translated_text}

@app.post("/translate/onlygpt")
async def translateWithGPT(request: TranslateRequest):
    try:
        print(request)
        chain = setup_translation_chain()

        response = chain.invoke({"text": request.text})

        return TranslateResponse(
            answer=response
        )
    
    except Exception as e:
        logging.error(f"Translation error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))