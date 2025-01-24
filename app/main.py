from fastapi import FastAPI, HTTPException, Request
from models.schemas import TranslateRequest, TranslateResponse
from fastapi.middleware.cors import CORSMiddleware
from models.translation import setup_translation_chain
import logging
from routes.glossary_router import router as glossary_router


app = FastAPI()
app.include_router(glossary_router)
translation_chain = setup_translation_chain()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 필요에 따라 특정 도메인으로 제한 가능
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/translate")
async def translate(request: TranslateRequest):
    # 받은 JSON 데이터 처리
    translated_text = f"Translated: {request.text}"
    return {"translated_text": translated_text}

@app.post("/translate/onlygpt")
async def translate(request: Request):
    try:
        body = await request.json()
        text = body.get("text", "")
        source_lang = body.get("source_lang", "ko")
        target_lang = body.get("target_lang", "en")

        # 체인에 전달할 입력 데이터 구성
        result = translation_chain.invoke({
            "text": text,
            "source_lang": source_lang,
            "target_lang": target_lang,
        })

        return {"answer": result}
    except Exception as e:
        logging.error(f"Translation error: {e}")
        return {"error": str(e)}
    