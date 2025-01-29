from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from app.models.translation import setup_translation_chain
import logging
from app.routes.glossary_router import router as glossary_router

# 로깅 설정
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# FastAPI 앱 초기화
app = FastAPI(title="Translation Service")

# 라우터 포함
app.include_router(glossary_router)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 필요 시 특정 도메인으로 제한
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 번역 체인 초기화 (한 번만 실행)
translation_chain = setup_translation_chain()


@app.post("/translate", tags=["Translation"])
async def translate(request: Request):
    """
    단순 번역 엔드포인트
    """
    try:
        body = await request.json()
        text = body.get("text", "")
        return {"translated_text": f"Translated: {text}"}
    except Exception as e:
        logger.error(f"Error in dummy translation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/translate/onlygpt", tags=["Translation"])
async def translate_with_gpt(request: Request):
    """
    GPT 기반 번역 엔드포인트
    """
    try:
        # 요청 데이터 로깅
        body = await request.json()
        logger.info(f"Received request: {body}")

        # JSON에서 필요한 데이터 추출
        text = body.get("text", "")
        source_lang = body.get("source_lang", "ko")  # 기본값 "ko"
        target_lang = body.get("target_lang", "en")  # 기본값 "en"

        # 체인 실행
        result = translation_chain.invoke({
            "text": text,
            "source_lang": source_lang,
            "target_lang": target_lang,
        })

        return {"answer": result}

    except Exception as e:
        logger.error(f"Translation error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
