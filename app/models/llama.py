from transformers import AutoTokenizer, AutoModelForCausalLM

# 사용할 모델 이름
model_name = "meta-llama/Llama-3.1-8B-Instruct"  # 모델 이름

# 토크나이저와 모델 로드
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)