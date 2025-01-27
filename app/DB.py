from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base

# DB_URL = ".env mongoDB url"



# engine = create_engine(DB_URL, connect_args={"check_same_thread": False})


# # 데이터 베이스 연결 세션 설정
# session_local = sessionmaker(autoflush=False, autocommit = False, bind=engine)

# # 모델 클래스가 상속받을 기본 모델 클래스 지정
# Base = declarative_base()

# def get_db():
#     db = session_local()
#     try:
#         yield db
#     finally:
#         db.close()