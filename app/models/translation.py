from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()

def setup_translation_chain():
    """
    GPT 번역 체인 설정 함수
    """
    # 번역 프롬프트 정의
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", """
            # Task:
            - Translation

            # Instructions:
            - Translate the text from {source_lang} to {target_lang}.
            - Ensure the translation is contextually accurate.
            - Just return the translated result only.
            """),
            ("human", "Text: {text}")
        ]
    )

    # OpenAI LLM 모델 설정 (gpt-4o-mini 사용)
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.5)

    # 체인 설정
    chain = (
            {
                "text": lambda x: x["text"],  # 올바른 lambda 구문
                "source_lang": lambda x: x.get("source_lang", "ko"),  # 기본값 "ko"
                "target_lang": lambda x: x.get("target_lang", "en"),  # 기본값 "en"
            }
            | prompt
            | llm
            | StrOutputParser()
    )

    return chain
