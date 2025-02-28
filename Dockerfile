# (1) PyTorch + CUDA 포함된 컨테이너 (GPU 지원)
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

# (2) 작업 디렉토리 설정
WORKDIR /app

# (3) 필수 패키지 설치
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libffi-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# (4) requirements.txt 먼저 COPY
COPY requirements.txt .

# (5) Python 패키지 설치
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# (6) 로컬의 모든 코드 복사
COPY . .

# (7) FastAPI 실행 (Uvicorn)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7000"]
