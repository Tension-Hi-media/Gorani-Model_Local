# 요청 응답 및 응답 데이터

from pydantic import BaseModel

class TranslateRequest(BaseModel):
    text: str
    source_lang: str
    target_lang: str
    model: str

class TranslateResponse(BaseModel):
    answer: str