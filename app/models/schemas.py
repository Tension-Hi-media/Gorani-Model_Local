# 요청 응답 및 응답 데이터
from pydantic import BaseModel, Field
from typing import Optional

class TranslateRequest(BaseModel):
    text: str
    source_lang: str
    target_lang: str
    model: Optional[str] = Field(None, description="번역 모델 (gpt 또는 llama)")

class TranslateResponse(BaseModel):
    answer: str
