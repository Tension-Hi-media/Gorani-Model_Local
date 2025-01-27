from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

def setup_translation_chain():
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

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.5)

    # 체인에 필요한 모든 입력 데이터 정의
    chain = (
        {
            "text": lambda x: x["text"],
            "source_lang": lambda x: x.get("source_lang", "ko"),  # 기본값 설정
            "target_lang": lambda x: x.get("target_lang", "en"),  # 기본값 설정
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain
