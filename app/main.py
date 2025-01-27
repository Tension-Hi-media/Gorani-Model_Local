from fastapi import FastAPI, HTTPException
from models.schemas import TranslateRequest, TranslateResponse
from fastapi.middleware.cors import CORSMiddleware
from models.translation import setup_translation_chain
from services.llama_service import setup_translation_chain_llama, create_metadata_array
import logging
from routes.glossary_router import router as glossary_router


app = FastAPI()
app.include_router(glossary_router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 필요에 따라 특정 도메인으로 제한 가능
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/translate/onlygpt", tags=["Translation"])
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
    
@app.post("/translate", tags=["Translation"])
async def translate(request: TranslateRequest):

    try:
        print(request)

        if request.target_lang == '한국어':
            request.target_lang = 'korean'
        elif request.target_lang == '영어':
            request.target_lang = 'english'
        else:
            request.target_lang = 'Japanese'

        chain = setup_translation_chain_llama()

        response = chain.run({
            "target_language": request.target_lang,
            "glossary": create_metadata_array(request.text, 10),
            "user_message": request.text
        }).lstrip("\n")

        return TranslateResponse(
            answer=response
        )
    
    except Exception as e:
        logging.error(f"Translation error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    
    