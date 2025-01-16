# 요청 응답 및 응답 데이터

from pydantic import BaseModel

class TranslateRequest(BaseModel):
    text: str

class TranslateResponse(BaseModel):
    answer: str