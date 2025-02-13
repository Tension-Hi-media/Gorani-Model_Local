from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging
from app.routes.glossary_router import router as glossary_router
from app.routes.translate import router as translate_router  # ✅ 추가

# 로깅 설정
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# FastAPI 앱 초기화
app = FastAPI(title="Translation Service")

# 라우터 포함
app.include_router(glossary_router)
app.include_router(translate_router)  # ✅ 번역 라우트 추가

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 필요 시 특정 도메인으로 제한
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
