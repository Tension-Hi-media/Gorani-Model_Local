from pymongo import MongoClient
from pymongo.database import Database
from pymongo.collection import Collection
import json
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import os
import pandas as pd
import numpy as np
from transformers import XLMRobertaTokenizer, XLMRobertaModel
# torch 임포트 제거
from transformers import pipeline
from models.llama import model, tokenizer

# 몽고DB-용어사전 연결
MONGO_URI = os.environ.get("MONGODB_ATLAS_CLUSTER_URI")
DATABASE_NAME = "TestDB"
COLLECTION_NAME = "test_bert"

# MongoDB 클라이언트 생성
client = MongoClient(MONGO_URI)
db: Database = client[DATABASE_NAME]
collection: Collection = db[COLLECTION_NAME]

# 토크나이저와 모델 로드
embedding_tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-base")
embedding_model = XLMRobertaModel.from_pretrained("xlm-roberta-base")

# CPU로 설정 (GPU 사용하지 않음)
device = "cpu"
embedding_model = embedding_model.to(device)

# 하이브리드 검색 가중치 설정
BM25_MAX_VALUE = 10.0  # 설정 필요
BM25_MIN_VALUE = 0.0   # 설정 필요
VECTOR_SCORE_WEIGHT = 0.5
TEXT_SCORE_WEIGHT = 0.5

def setup_translation_chain_llama():
    prompt = setPrompt()

    # Hugging Face 파이프라인 설정
    hf_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        return_full_text=False,
        device=-1
    )

    # LLMChain 설정
    chain = LLMChain(llm=hf_pipeline, prompt=prompt)

    return chain

def setPrompt():
    prompt_template_str = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
    - You are required to translate the user message into the target language strictly, even if the text is in the form of a question.
    - Your response must contain only the translated text of the original message and nothing else. Do not include explanations, clarifications, or any additional information.
    - Adhere to the glossary for applicable terms.
    - Apply proper capitalization rules for general nouns, ensuring they are lowercase in the middle of a sentence unless they are proper nouns or explicitly marked as capitalized in the glossary.
    - For terms like Scourge, use lowercase and pluralize if the context suggests multiple entities.

    ### target language ###
    {target_language}

    ### glossary ###
    {glossary}

    <|eot_id|><|start_header_id|>user<|end_header_id|>
    {user_message}

    <|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

    # PromptTemplate 정의
    prompt_template = PromptTemplate(
        input_variables=["target_language", "glossary", "user_message"],
        template=prompt_template_str,
    )

    return prompt_template

def create_metadata_array(query, limit=10):
    """
    Hybrid Search를 수행하여 metadata를 array 형태로 묶은 JSON string 반환
    """
    # Hybrid Search 수행
    search_results = hybrid_search(query, limit)

    # metadata array 생성
    metadata_array = []
    for result in search_results[:limit]:
        metadata = result.get("metadata", {})
        metadata_array.append(metadata)

    # metadata array를 JSON string으로 변환
    metadata_json = json.dumps(metadata_array, ensure_ascii=False)

    return metadata_json

def hybrid_search(query, length=10, model=embedding_model, tokenizer=embedding_tokenizer):
    # 벡터 및 텍스트 검색 수행
    embedding = get_embedding_from_xlm_roberta(query, model, tokenizer)
    vector_results = vector_search(embedding, "vector_index")
    text_results = text_search(query, "text_index")

    # 결과 병합
    combined_results = {}
    for result in vector_results:
        doc_id = result["_id"]
        vector_score = result.get("vectorScore", 0)
        combined_results[doc_id] = {
            **result,
            "vectorScore": vector_score,
            "score": calculate_convex_score(vector_score, 0)
        }

    for result in text_results:
        doc_id = result["_id"]
        text_score = result.get("textScore", 0)
        if doc_id not in combined_results:
            combined_results[doc_id] = {
                **result,
                "vectorScore": 0,
                "score": calculate_convex_score(0, text_score)
            }
        else:
            vector_score = combined_results[doc_id]["vectorScore"]
            combined_results[doc_id]["textScore"] = text_score
            combined_results[doc_id]["score"] = calculate_convex_score(vector_score, text_score)

    # 결과 정렬
    sorted_results = sorted(combined_results.values(), key=lambda x: x["score"], reverse=True)
    return sorted_results[0:length]

def get_embedding_from_xlm_roberta(text, model, tokenizer):
    """
    XLM-RoBERTa 모델을 사용해 텍스트 임베딩 생성
    """

    # 입력 텍스트가 문자열인 경우 리스트로 변환
    if isinstance(text, str):
        text = [text]

    # 텍스트 토큰화
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

    # 모델로 임베딩 생성 (CPU에서 실행)
    outputs = model(**inputs)

    # CLS 토큰 임베딩 사용
    embeddings = outputs.last_hidden_state[:, 0, :].detach().numpy()  # NumPy 배열로 변환

    # 입력 텍스트가 단일 문장이었다면 첫 번째 임베딩만 반환
    if len(embeddings) == 1:
        return embeddings[0].tolist()  # NumPy 배열을 리스트로 변환하여 반환
    return embeddings.tolist()  # 전체 임베딩을 리스트로 변환하여 반환


# @title
def normalize_vector_score(vector_score):
    return (vector_score + 1) / 2.0

def normalize_bm25_score(bm25_score):
    return min((bm25_score - BM25_MIN_VALUE) / (BM25_MAX_VALUE - BM25_MIN_VALUE), 1.0)

def calculate_convex_score(vector_score, bm25_score):
    tmm_vector_score = normalize_vector_score(vector_score)
    tmm_bm25_score = normalize_bm25_score(bm25_score)
    return VECTOR_SCORE_WEIGHT * tmm_vector_score + TEXT_SCORE_WEIGHT * tmm_bm25_score

# @title
def vector_search(query_vector, vector_index_name, num_candidates=64, limit=25):
    """
    벡터 검색 수행
    """
    pipeline = [
        {
            "$vectorSearch": {
                "index": vector_index_name,
                "path": "embedding",
                "queryVector": query_vector,
                "numCandidates": num_candidates,
                "limit": limit
            }
        },
        {
            "$project": {
                "metadata": 1,
                "content": 1,
                "vectorScore": {"$meta": "vectorSearchScore"},
                "score": {"$meta": "vectorSearchScore"}
            }
        },
        {
            "$sort": {"score": -1}
        },
        {
            "$limit": limit
        }
    ]

    results = collection.aggregate(pipeline)
    return list(results)

# @title
def text_search(query, text_index_name, limit=25):
    """
    텍스트 검색 수행
    """
    pipeline = [
        {
            "$search": {
                "index": text_index_name,
                "text": {
                    "query": query,
                    "path": ["content", "metadata.KO", "metadata.ENG", "metadata.JPN"]
                }
            }
        },
        {
            "$project": {
                "metadata": 1,
                "content": 1,
                "textScore": {"$meta": "searchScore"},
                "score": {"$meta": "searchScore"}
            }
        },
        {
            "$sort": {"score": -1}
        },
        {
            "$limit": limit
        }
    ]

    results = collection.aggregate(pipeline)
    return list(results)