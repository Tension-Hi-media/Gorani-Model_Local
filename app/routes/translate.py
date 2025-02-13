import httpx
import asyncio
from fastapi import APIRouter, HTTPException
import logging
from app.models.schemas import TranslateRequest, TranslateResponse
from app.services.translation_service import translate_text  # ✅ FastAPI에서 사용 가능하도록 임포트

router = APIRouter()
logger = logging.getLogger(__name__)

# ✅ Runpod Gorani(Llama) 모델 주소 (Runpod Proxy URL 사용)
RUNPOD_MODEL_URL = "https://3m392zclj3zw79-5000.proxy.runpod.net/translate"

# ✅ **번역을 명확히 요청하는 프롬프트 설정**
TRANSLATION_PROMPT = """You are a professional translator. 
Translate the given text from {source_lang} to {target_lang}.
Do not explain or provide additional information. 

### Source Text:
{text}

### Translated Text:
"""

async def translate_with_gorani(text: str, source_lang: str = "ko", target_lang: str = "en", model: str = "Gorani") -> str:
    """
    Runpod의 Gorani(Llama) 모델을 호출하여 번역 수행
    """
    # ✅ **수정된 프롬프트 적용**
    prompt = TRANSLATION_PROMPT.format(source_lang=source_lang, target_lang=target_lang, text=text)

    payload = {
        "text": prompt,
        "source_lang": source_lang,
        "target_lang": target_lang,
        "model": model
    }

    async with httpx.AsyncClient(timeout=60.0) as client:  # ✅ 타임아웃 60초 설정
        try:
            response = await client.post(RUNPOD_MODEL_URL, json=payload)

            # ✅ 응답 상태 코드 체크
            if response.status_code != 200:
                logger.error(f"❌ Runpod 응답 오류: {response.status_code} - {response.text}")
                raise HTTPException(status_code=502, detail=f"Runpod 응답 오류: {response.status_code}")

            # ✅ JSON 응답 구조 확인 및 예외 처리
            try:
                response_data = response.json()
                logger.info(f"✅ Runpod 응답: {response_data}")
                return response_data.get("output", "번역 실패").strip()
            except ValueError:
                logger.error(f"❌ JSON 파싱 오류 - 응답 내용: {response.text}")
                raise HTTPException(status_code=502, detail="Runpod 응답 JSON 파싱 오류")

        except httpx.RequestError as e:
            logger.error(f"❌ HTTP 통신 오류 발생: {str(e)}")
            raise HTTPException(status_code=502, detail="Runpod 서버 연결 오류")

        except Exception as e:
            logger.error(f"❌ 알 수 없는 오류 발생: {str(e)}")
            raise HTTPException(status_code=500, detail="번역 요청 중 알 수 없는 오류 발생")

@router.post("/translate/onlygpt", response_model=TranslateResponse, tags=["Translation"])
async def translate_with_gpt(request: TranslateRequest):
    """
    GPT 또는 Runpod의 Gorani(Llama) 모델을 사용하여 번역 수행
    """
    try:
        logger.info(f"📥 Received request: {request.dict()}")

        model = request.model if request.model else "OpenAI"

        if model == "OpenAI":
            result = translate_text(request.text, request.source_lang, request.target_lang)
        elif model == "Gorani":
            logger.info("🚀 Runpod Gorani 모델로 요청 중...")
            # ✅ `source_lang`, `target_lang`, `model`도 함께 전달
            result = await translate_with_gorani(request.text, request.source_lang, request.target_lang, request.model)
            logger.info(f"✅ Runpod 응답: {result}")
        else:
            raise HTTPException(status_code=400, detail="지원되지 않는 모델입니다.")

        return TranslateResponse(answer=result)

    except Exception as e:
        logger.error(f"❌ Translation error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
