import openai
import os

# API 키 설정
openai.api_key = os.getenv("OPENAI_API_KEY")

def generate_openai_response(prompt: str) -> str:
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    return response["choices"][0]["message"]["content"]