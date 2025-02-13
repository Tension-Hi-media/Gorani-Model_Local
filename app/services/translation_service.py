import logging
from app.models.translation import setup_translation_chain

# 로깅 설정
logger = logging.getLogger(__name__)

# GPT 번역 체인 초기화
translation_chain = setup_translation_chain()

def translate_text(text: str, source_lang: str = "ko", target_lang: str = "en") -> str:
    """
    GPT 번역 체인을 실행하여 번역 수행
    """
    try:
        result = translation_chain.invoke({
            "text": text,
            "source_lang": source_lang,
            "target_lang": target_lang,
        })
        return result
    except Exception as e:
        logger.error(f"Translation error: {str(e)}")
        return "Translation failed"
