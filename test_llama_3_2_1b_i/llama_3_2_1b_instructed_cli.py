import torch
import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

# base_model = "meta-llama/Llama-3.2-1B-Instruct"
base_model = "meta-llama/Meta-Llama-3-8B-Instruct"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, # Use 4-bit quantization for memory and time efficency
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=False,
)

# Loads the model using HF libraries
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    quantization_config=bnb_config,
    device_map="auto",
)

# Loads the tokenizer corresponding to the loaded model. 
tokenizer = AutoTokenizer.from_pretrained(base_model)

messages = [
    {"role": "system", "content": "You are a generative AI assistant. Be formal, brief and concise."},
    # {"role": "user", "content": "Who are you?"},
]

# Use HF pipeline
pipe = transformers.pipeline(
    model=model,
    tokenizer=tokenizer,
    return_full_text=False, # Langchain expects the full text
    task='text-generation',
    max_new_tokens=256, # Small number of max out tokens for time efficiency
    temperature=1,  # Temperature for more or less creative answers
    do_sample=True,
    top_p=0.95,
)

# Main Conversation Loop, read and ingest user prompt
while(True):
    user_prompt = input("Q:\t")
    messages.append({"role": "user", "content": user_prompt})
    gen_seqs = pipe(messages)
    segn_seq_str = ""
    for seq in gen_seqs:
        segn_seq_str = segn_seq_str + seq['generated_text']
    messages.append({"role": "assistant", "content": segn_seq_str})
    print(f"A:\t{segn_seq_str}")
