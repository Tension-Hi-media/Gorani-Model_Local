from fastapi import APIRouter, HTTPException, Query
from typing import List, Optional
from pydantic import BaseModel
from typing import List
from pymongo import MongoClient
import os

router = APIRouter()

# MongoDB Atlas 연결 문자열 (예시)
# 실제 DB 사용자, 비밀번호, 클러스터 정보를 바꿔주세요.
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
client = MongoClient(MONGO_URI)

db = client["glossary_db"]  # 'glossary_db'라는 데이터베이스 선택
collection = db["glossaries"]

class WordPair(BaseModel):
    start: str
    arrival: str

class Glossary(BaseModel):
    name: str
    userId: Optional[int] = None
    words: List[WordPair]

@router.post("/api/glossary")
async def save_glossary(glossary: Glossary):
    try:
        collection.insert_one(glossary.dict())
        return {"message": "용어집 저장 성공"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/glossary", tags=["Glossary"])
def get_glossaries(userId: int):
    """
    userId가 일치하는 문서만 가져옴.
    예: GET /api/glossary?userId=1
    """
    try:
        cursor = collection.find({"userId": userId})
        results = []
        for doc in cursor:
            doc["_id"] = str(doc["_id"])  # ObjectId → 문자열
            results.append(doc)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))