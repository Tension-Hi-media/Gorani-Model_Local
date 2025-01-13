from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from app.models.translation import setup_translation_chain
import logging

app = FastAPI()

class TranslateRequest(BaseModel):
    text: str

class TranslateResponse(BaseModel):
    answer: str

@app.post("/translate")
async def translate(request: TranslateRequest):
    # 받은 JSON 데이터 처리
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
