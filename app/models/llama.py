from unsloth import FastLanguageModel

max_seq_length = 1024 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="taeyoon12321421/gorani-8b-finetuned", # YOUR MODEL YOU USED FOR TRAINING
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)

FastLanguageModel.for_inference(model) # Enable native 2x faster inference

# taeyoon12321421/gorani-8b-finetuned 최신파인튜닝
# haeun0420/gorani-1B-4bit
# unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit
# Bring2It2On/gorani-9B-4bit
# lucian1120/gorani-3.1-8B-4bit