from fastapi import APIRouter, HTTPException, Query
from typing import List, Optional
from pydantic import BaseModel
from typing import List
from pymongo import MongoClient
from bson import ObjectId 
import os
import logging
from fastapi.responses import JSONResponse

# 로깅 설정
logging.basicConfig(level=logging.INFO)  # INFO 레벨 이상의 로그 출력
logger = logging.getLogger(__name__)
router = APIRouter()

# MongoDB Atlas 연결 문자열 (예시)
# 실제 DB 사용자, 비밀번호, 클러스터 정보를 바꿔주세요.
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
client = MongoClient(MONGO_URI)

db = client["glossary_db"]  # 'glossary_db'라는 데이터베이스 선택
collection = db["glossaries"]

class WordPair(BaseModel):
    _id: Optional[str] = None  # 새 단어쌍은 _id가 없음
    start: str
    arrival: str

class Glossary(BaseModel):
    name: str
    userId: Optional[int] = None
    words: List[WordPair]

class UpdateGlossaryNameRequest(BaseModel):
    name: str

from fastapi import HTTPException

#용어집 생성
@router.post("/api/glossary", tags=["Glossary"])
async def save_glossary(glossary: Glossary):
    if glossary.userId is None:
        raise HTTPException(status_code=400, detail="userId는 필수입니다.")
    try:
        inserted_id = collection.insert_one(glossary.dict()).inserted_id
        saved_glossary = collection.find_one({"_id": inserted_id})
        saved_glossary["_id"] = str(saved_glossary["_id"])  # ObjectId를 문자열로 변환
        return saved_glossary  # 생성된 용어집 데이터를 반환
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

#용어집 이름 변경
@router.put("/api/glossary/{id}", tags=["Glossary"])
async def update_glossary_name(id: str, request: UpdateGlossaryNameRequest):
    if not request.name:
        raise HTTPException(status_code=400, detail="이름은 비어 있을 수 없습니다.")
    
    glossary = collection.find_one({"_id": ObjectId(id)})
    if not glossary:
        raise HTTPException(status_code=404, detail="용어집을 찾을 수 없습니다.")

    collection.update_one({"_id": ObjectId(id)}, {"$set": {"name": request.name}})
    return {"message": "용어집 이름 업데이트 성공", "updatedName": request.name}

#용어집 조회
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

#용어집 삭제
@router.delete("/api/glossary/{id}", tags=["Glossary"])
async def delete_glossary(id: str):
    try:
        result = collection.delete_one({"_id": ObjectId(id)})
        if result.deleted_count == 0:
            raise HTTPException(status_code=404, detail="용어집을 찾을 수 없습니다.")
        return {"message": "용어집 삭제 성공"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"용어집 삭제 실패: {str(e)}")

# 기본 용어집 설정 후 반환
@router.put("/api/v1/glossary/{user_id}/default", tags=["Glossary"])
async def set_default_glossary(user_id: int, glossary_id: str = Query(...)):
    logger.info(f"Received user_id: {user_id}, glossary_id: {glossary_id}")

    # 기존 기본 용어집을 false로 리셋
    reset_default_glossary(user_id)  # 기존 기본 용어집을 false로 설정

    # 새로운 기본 용어집을 찾아서 isDefault를 True로 설정
    glossary = collection.find_one({"_id": ObjectId(glossary_id), "userId": user_id})
    if not glossary:
        raise HTTPException(status_code=404, detail="해당 유저의 용어집을 찾을 수 없습니다.")

    # 기본 용어집 설정
    collection.update_one(
        {"_id": ObjectId(glossary_id), "userId": user_id},
        {"$set": {"isDefault": True}}
    )

    # 기본 용어집만 반환
    default_glossary = collection.find_one({"_id": ObjectId(glossary_id), "userId": user_id})
    if default_glossary:
        default_glossary["_id"] = str(default_glossary["_id"])  # ObjectId를 문자열로 변환
        return JSONResponse(content={"message": "Default glossary set successfully", "glossary": default_glossary})

    raise HTTPException(status_code=500, detail="기본 용어집 설정에 실패했습니다.")

# 기존 기본 용어집을 리셋하는 엔드포인트 정의
@router.put("/api/v1/glossary/{user_id}/reset-default", tags=["Glossary"])
async def reset_default_glossary(user_id: int):
    logger.info(f"Received request to reset default glossary for user_id: {user_id}")

    try:
        # 해당 유저의 모든 용어집에서 기본 용어집을 false로 설정
        result = collection.update_many(
            {"userId": user_id}, 
            {"$set": {"isDefault": False}}
        )

        # 업데이트된 문서 수를 로그로 출력
        logger.info(f"Reset {result.modified_count} glossaries' default status to False.")

        return {"message": "All glossaries' default status reset successfully"}
    
    except Exception as e:
        logger.error(f"Error resetting default glossary: {str(e)}")
        raise HTTPException(status_code=500, detail="기본 용어집 리셋 중 오류 발생")

# add_word_pair에서 _id 추가
@router.post("/api/glossary/{id}/word-pair", tags=["WordPair"])
async def add_word_pair(id: str, word_pair: WordPair):
    try:
        word_pair_dict = word_pair.dict()
        word_pair_dict["_id"] = str(ObjectId())  # 새로 생성된 _id 추가
        result = collection.update_one(
            {"_id": ObjectId(id)},
            {"$push": {"words": word_pair_dict}}
        )
        if result.matched_count == 0:
            raise HTTPException(status_code=404, detail="용어집을 찾을 수 없습니다.")
        return word_pair_dict  # 새로 추가된 단어쌍 반환
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.put("/api/glossary/{glossaryId}/word-pair/{wordPairId}", tags=["WordPair"])
async def update_word_pair(glossaryId: str, wordPairId: str, word_pair: WordPair):
    try:
        logger.info(f"Received update request: glossaryId={glossaryId}, wordPairId={wordPairId}, data={word_pair}")

        # MongoDB에서 용어집 찾기
        glossary = collection.find_one({"_id": ObjectId(glossaryId)})
        if not glossary:
            logger.warning("Glossary not found")
            raise HTTPException(status_code=404, detail="용어집을 찾을 수 없습니다.")

        # 배열 내 단어쌍 수정
        updated = False
        for word in glossary["words"]:
            if str(word["_id"]) == wordPairId:
                word["start"] = word_pair.start
                word["arrival"] = word_pair.arrival
                updated = True
                break

        if not updated:
            logger.warning("Word pair not found in glossary")
            raise HTTPException(status_code=404, detail="단어쌍을 찾을 수 없습니다.")

        # MongoDB 업데이트
        collection.update_one(
            {"_id": ObjectId(glossaryId)},
            {"$set": {"words": glossary["words"]}}
        )

        logger.info("Word pair updated successfully")
        return {"message": "단어쌍 수정 성공", "updatedWordPair": word_pair.dict()}
    except Exception as e:
        logger.error(f"Failed to update word pair: {e}")
        raise HTTPException(status_code=500, detail=f"단어쌍 수정 실패: {str(e)}")

@router.delete("/api/glossary/{id}/word-pair/{index}", tags=["WordPair"])
async def delete_word_pair(id: str, index: int):
    try:
        glossary = collection.find_one({"_id": ObjectId(id)})  # ObjectId 변환 추가
        if not glossary:
            raise HTTPException(status_code=404, detail="용어집을 찾을 수 없습니다.")

        if index < 0 or index >= len(glossary["words"]):
            raise HTTPException(status_code=400, detail="잘못된 단어쌍 인덱스입니다.")

        # 단어쌍 삭제
        glossary["words"].pop(index)
        collection.update_one({"_id": ObjectId(id)}, {"$set": {"words": glossary["words"]}})
        return {"message": "단어쌍 삭제 성공"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"단어쌍 삭제 실패: {str(e)}")
    
@router.get("/api/glossary/{id}/word-pair", tags=["WordPair"])
async def get_word_pairs(id: str):
    try:
        logger.info(f"Fetching glossary with id: {id}")  # 용어집 ID 로그
        glossary = collection.find_one({"_id": ObjectId(id)})
        if not glossary:
            logger.warning(f"Glossary not found for id: {id}")  # 경고 로그
            raise HTTPException(status_code=404, detail="용어집을 찾을 수 없습니다.")

        # 단어쌍 리스트에서 `_id`를 문자열로 변환
        words = glossary.get("words", [])
        for word in words:
            if "_id" in word:
                word["_id"] = str(word["_id"])  # ObjectId → 문자열

        logger.info(f"Fetched word pairs: {words}")  # 단어쌍 데이터 로그
        return words  # 변환된 단어쌍 리스트 반환
    except Exception as e:
        logger.error(f"Error fetching word pairs for glossary {id}: {str(e)}")  # 오류 로그
        raise HTTPException(status_code=500, detail=f"단어쌍 조회 실패: {str(e)}")