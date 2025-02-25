from langchain.schema import AIMessage
from langgraph.graph import StateGraph, START, END
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import os
from langchain_openai import ChatOpenAI
from app.models.schemas import State
from app.services.llama_service import setChain1, create_metadata_array

def setup_translation_graph_LangGorani():

    chain1 = setChain1()

    chain2 = setchain2()

    def node_translate_text(state: State) -> State:
        user_message = state["messages"][-1].content
        target_language = state["targetLanguage"]
        source_language = state["source_lang"]
        glossary = create_metadata_array(user_message, 10)
        translated_text = chain1.run({"src_lang":source_language,"target_language": target_language,"glossary_text": glossary,"input_text": user_message}).lstrip("\n")
        state["translateStr"] = translated_text
        state["originStr"] = user_message
        state["glossary"] = glossary
        state["messages"].append(AIMessage(content=translated_text))
        return state

    # 번역 검토 노드
    def node_evaluate_translation(state: State) -> State:
        target_language = state["targetLanguage"]
        user_message = state["originStr"]
        glossary = state["glossary"]
        translated_text = state["translateStr"]
        evaluation = chain2.run({"target_language": target_language,"glossary": glossary,"original_text": user_message,"translated_text": translated_text})
        state["evaluation"] = evaluation
        state["messages"].append(AIMessage(content=evaluation))
        return state
    
    # LangGraph Workflow
    graph_builder = StateGraph(State)
    graph_builder.add_node("translate", node_translate_text)
    graph_builder.add_node("evaluate", node_evaluate_translation)
    graph_builder.add_edge(START, "translate")
    graph_builder.add_edge("translate", "evaluate")
    graph_builder.add_edge("evaluate", END)

    graph = graph_builder.compile()
        
    return graph

def setchain2():
    
    # OpenAI 모델 초기화
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, openai_api_key = os.getenv('OPENAI_API_KEY'))
    
    prompt = setPrompt2()

    chain = LLMChain(llm=llm, prompt=prompt)

    return chain

def setPrompt2():
    prompt = """
    ### Task
    - Evaluate the quality of the translation by reviewing the Original Text and its corresponding Translated Text.
    - Check if Translated Text is the explanation or answer of the Original Text which is wrong. The Translated Text should be strictly a translation of Original Text.
    - If the Translated Text is accurately translated from the Original Text into the Target Language, write only the Translated Text.
    - If the Translated Text is not well translated, provide a new translation of the Original Text into the Target Language and respond with that new translation; Do not answer from Orignial Text; Just translate no matter what.
    - Refer to the glossary if any of the words in Original Text is relevant.

    ### Original Text:
    {original_text}

    ### Target Language:
    {target_language}

    ### gloassary:
    {glossary}

    ### Translated Text:
    {translated_text}

    ### Example 1
    **Original Text:** 사과는 맛있을까?

    **Target Language:** english

    **gloassary:**
    ["KO": "사과", "ENG": "apple", "JPN": "リンゴ"]

    **Translated Text:** Do you think apples are delicious?

    **Response:** Do you think apples are delicious?

    ### Response:"""

    # PromptTemplate 정의
    prompt_template2 = PromptTemplate(
        input_variables=["original_text", "target_language" "glossary", "translated_text"],  # 사용자가 입력할 변수
        template=prompt,
    )

    return prompt_template2