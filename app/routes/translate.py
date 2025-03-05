# import re
from fastapi import APIRouter, HTTPException
# from langchain.schema import HumanMessage
import logging
from app.models.schemas import TranslateRequest, TranslateResponse, State
# from app.services.llama_service import setup_translation_chain_llama, create_metadata_array
# from app.services.langGorani_service import setup_translation_graph_LangGorani
from app.services.openAI_service import setup_translation_chain

router = APIRouter()
logger = logging.getLogger(__name__)
    
# @router.post("/translate/Gorani", tags=["Translation"])
# async def translate(request: TranslateRequest):

#     try:
#         print(request)

#         glossary = create_metadata_array(request.text, 10)

#         chain = setup_translation_chain_llama()

#         response = chain.invoke({
#             "src_lang": request.source_lang,
#             "input_text": request.text,
#             "target_language": request.target_lang,
#             "glossary_text": glossary
#         })

#         print(glossary)
#         print(response)

#         # translated_text = response['text'].strip()

#         translated_text = response['text']
#         cleaned_text = re.sub(r'^assistant\s*', '', translated_text).strip()

#         return TranslateResponse(
#             answer=cleaned_text
#         )
    
#     except Exception as e:
#         logging.error(f"Translation error: {str(e)}")
#         raise HTTPException(status_code=500, detail=str(e))

# @router.post("/translate/LangGorani", tags=["Translation"])
# async def translate(request: TranslateRequest):

#     try:
#         print("번역 요청 발생!! -> ",request)

#         graph = setup_translation_graph_LangGorani()

#         initial_state = {"messages": [HumanMessage(content=request.text)], "targetLanguage" : request.target_lang, "source_lang" : request.source_lang}
        
#         response = graph.invoke(initial_state)

#         print("Response : ",response)

#         translated_text = response["messages"][-1].content

#         print("번역 결과 : ",translated_text)

#         return TranslateResponse(
#             answer=translated_text
#         )
    
#     except Exception as e:
#         logging.error(f"Translation error: {str(e)}")
#         raise HTTPException(status_code=500, detail=str(e))
    
@router.post("/translate/OpenAI", tags=["Translation"])
async def translateWithGPT(request: TranslateRequest):
    try:
        print(request)
        chain = setup_translation_chain()

        response = chain.invoke({"text": request.text})

        return TranslateResponse(
            answer=response
        )

    except Exception as e:
        logging.error(f"Translation error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))