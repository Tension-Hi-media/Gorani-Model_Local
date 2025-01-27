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
            - Translate the text given in Korean that best fits with the context.
            - Translate the words in english.
            - Just return the translated result only

            """),
            ("human", "Text: {text}")
        ]
    )

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.5)

    chain = (
        {
            "text": lambda x: x["text"]
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain