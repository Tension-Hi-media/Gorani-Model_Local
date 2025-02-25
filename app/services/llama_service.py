import json
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from transformers import XLMRobertaTokenizer, XLMRobertaModel
from langchain_huggingface import HuggingFacePipeline
import torch
from transformers import pipeline
from app.models.llama import model, tokenizer
from dotenv import load_dotenv
# from app.services.MongoDB import collection

# .env 파일 로드
load_dotenv()

import os
from pymongo import MongoClient
from pymongo.database import Database
from pymongo.collection import Collection

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
embedding_model = embedding_model.to(device)

# 하이브리드 검색 가중치 설정
BM25_MAX_VALUE = 20.0  # 설정 필요
BM25_MIN_VALUE = 8.0   # 설정 필요
VECTOR_SCORE_WEIGHT = 0.3
TEXT_SCORE_WEIGHT = 0.7

def setup_translation_chain_llama():

    prompt = setPrompt()

    # Hugging Face 파이프라인 설정
    hf_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        return_full_text=False,
    )

    # LangChain에서 Hugging Face 파이프라인 Wrapping
    llm = HuggingFacePipeline(pipeline=hf_pipeline)

    # LLMChain 설정
    chain = LLMChain(llm=llm, prompt=prompt)

    return chain

def setChain1():
    prompt = setPrompt()

    # Hugging Face 파이프라인 설정
    hf_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        return_full_text=False,
    )

    llm = HuggingFacePipeline(pipeline=hf_pipeline)

    chain = LLMChain(llm=llm, prompt=prompt)

    return chain

def setPrompt():

    eos_token = tokenizer.eos_token if tokenizer.eos_token is not None else "<EOS>"

    alpaca_prompt = """
    ### Task: Translation with Context and Glossary
    You will be provided with a Input Text that needs to be translated accurately.
    The source language of the Input Text is given under Source Language. Translate the Input Text into target language given under Target Language.
    Use the glossary for reference. If any of the words  is not found in the glossary, translate it normally. 
    Respond with only the translation of the Input Text. Do not provide any explanations.

    ### Example 1:
    **Source Language:** Korean
    **Target Language:** English
    **Glossary:**
    ["KO": "고라니", "ENG": "gorani", "JPN": "キバノロ"]
    **Input Text:** 고라니는 한국의 토종 동물입니까?
    **Translated Output:** Is gorani a native animal of Korea?

    ### Example 2:
    **Source Language:** Korean
    **Target Language:** Japanese
    **Glossary:**
    ["KO": "고라니", "ENG": "gorani", "JPN": "キバノロ"]
    **Input Text:** 고라니는 한국의 토종 동물입니다.
    **Translated Output:** ゴラニは韓国の在来動物です。

    ### Example 3:
    **Source Language:** English
    **Target Language:** Korean
    **Glossary:**
    ["KO": "사과", "ENG": "apple", "JPN": "リンゴ"]
    **Input Text:** Please explain about the apple.
    **Translated Output:** 사과에 대해 설명해주세요.

    ### Example 4:
    **Source Language:** English
    **Target Language:** Japanese
    **Glossary:**
    ["KO": "사과", "ENG": "apple", "JPN": "リンゴ"]
    **Input Text:** The apple is delicious.
    **Translated Output:** リンゴは美味しいです。

    ### Example 5:
    **Source Language:** Japanese
    **Target Language:** Korean
    **Glossary:**
    ["KO": "학교", "ENG": "school", "JPN": "学校"]
    **Input Text:** 学生が学校に行きます。
    **Translated Output:** 학생들이 학교에 갑니다.

    ### Example 6:
    **Source Language:** Japanese
    **Target Language:** English
    **Glossary:**
    ["KO": "학교", "ENG": "school", "JPN": "学校"]
    **Input Text:** 学生が学校に行きます。
    **Translated Output:** The students go to school.

    ### Source Language:
    {src_lang}

    ### Target Language:
    {target_language}

    ### Glossary:
    {glossary_text}

    ### Input Text:
    {input_text}

    ### Translated Output:""" + eos_token + "<|start_header_id|>assistant<|end_header_id|>"

    # 프롬프트 템플릿 생성
    prompt_template = PromptTemplate(
        input_variables=["src_lang", "input_text", "target_language", "glossary_text"],
        template=alpaca_prompt
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

        # 배열 추가
        metadata_array.append(metadata)

    # metadata array를 JSON string으로 변환
    metadata_json = json.dumps(metadata_array, ensure_ascii=False)

    return metadata_json

def hybrid_search(query, length=10, model=embedding_model, tokenizer=embedding_tokenizer):
    # 벡터 및 텍스트 검색 수행
    embedding = get_embedding_from_xlm_roberta(query, model, tokenizer)
    vector_results = vector_search(embedding, "vector_index")
    text_results = text_search(query,"text_index")

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

    # score가 0.3 미만인 결과는 제외
    filtered_results = [result for result in combined_results.values() if result["score"] >= 0.5]

    # 결과 정렬 (score 높은 순으로 내림차순)
    sorted_results = sorted(filtered_results, key=lambda x: x["score"], reverse=True)

    # 상위 length개의 결과만 반환
    return sorted_results[:length]

    # # 결과 정렬
    # sorted_results = sorted(combined_results.values(), key=lambda x: x["score"], reverse=True)
    # return sorted_results[0:length]

def get_embedding_from_xlm_roberta(text, model, tokenizer):
    """
    XLM-RoBERTa 모델을 사용해 텍스트 임베딩 생성

    Args:
        text (str or List[str]): 임베딩을 생성할 텍스트 또는 텍스트 리스트
        model: 사전 학습된 XLM-RoBERTa 모델
        tokenizer: 사전 학습된 XLM-RoBERTa 토크나이저

    Returns:
        List[float] or List[List[float]]: 입력 텍스트의 임베딩
    """

    # 입력 텍스트가 문자열인 경우 리스트로 변환
    if isinstance(text, str):
        text = [text]

    # if model is not uploaded on device
    if not model.device.type == device:
        model = model.to(device)

    # 텍스트 토큰화
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # 모델로 임베딩 생성
    with torch.no_grad():
        outputs = model(**inputs)

    # CLS 토큰 임베딩 사용
    embeddings = outputs.last_hidden_state[:, 0, :].cpu().tolist()

    # 입력 텍스트가 단일 문장이었다면 첫 번째 임베딩만 반환
    if len(embeddings) == 1:
        return embeddings[0]
    return embeddings

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

