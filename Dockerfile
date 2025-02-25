# 최신 Python 3.10 Slim 버전 사용
FROM python:3.10-slim

# 작업 디렉터리 설정
WORKDIR /app

# 필수 패키지 설치 (pip 업그레이드 포함)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libffi-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# (1) requirements.txt만 먼저 COPY
COPY requirements.txt .

# (2) pip install (requirements가 바뀌지 않으면 여기서 캐시 사용)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# (3) 나머지 애플리케이션 코드 복사
COPY . .

# FastAPI 실행 (Uvicorn)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7000"]
