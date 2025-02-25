# 요청 응답 및 응답 데이터
from pydantic import BaseModel, Field
from typing import Optional, TypedDict, List
from langchain.schema import SystemMessage, HumanMessage, AIMessage

class TranslateRequest(BaseModel):
    text: str
    source_lang: str
    target_lang: str
    model: Optional[str] = Field(None, description="번역 모델 (gpt 또는 llama)")

class TranslateResponse(BaseModel):
    answer: str

class State(TypedDict):
    messages: List[HumanMessage | AIMessage | SystemMessage]
    originStr: str
    source_lang: str
    translateStr: str
    targetLanguage: str
    evaluation: str
    glossary: str