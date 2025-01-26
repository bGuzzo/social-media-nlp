import gc
import tkinter as tk
from tkinter import scrolledtext, Label, Entry
import torch
import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
import logging

# Clean VRAM cahce
gc.collect()
torch.cuda.empty_cache()

# Create & configure logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO) 
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO) 
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

system_prompt = """
You are a large language model, trained to be informative and concise.
Do not provide explanations or elaborations unless explicitly requested.
Focus on delivering the most direct and accurate answer to the user's query.
"""

# Gemma
# base_model = "google/gemma-2-9b-it"

# 1B
base_model = "meta-llama/Llama-3.2-1B-Instruct"

# 3B
# base_model = "meta-llama/Llama-3.2-3B-Instruct"

# 8B
# base_model = "meta-llama/Llama-3.1-8B-Instruct"


# LLaMA 3.2 3B SDG 5 epochs
# base_model = "/home/bruno/Documents/GitHub/social-media-nlp/training/outputs/llama-3-2-3B-SDG-5e"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=False,
)

try:
    model = AutoModelForCausalLM.from_pretrained(
        base_model, 
        quantization_config=bnb_config, 
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model)
except Exception as e:
    logger.error(f"Error loading model: {e}")
    exit(1)

# System prompt used as 0-time message in the chat-template.
messages = [
    {"role": "system", "content": system_prompt},
]

# --- GUI Implementation using Tkinter ---
root = tk.Tk()
root.title("LLaMA 3.2 Chatbot")

# --- Parameter Input Boxes ---
temp_label = Label(root, text="Temperature:")
temp_label.pack()
temp_entry = Entry(root)
temp_entry.insert(0, "1.0")  # Default value
temp_entry.pack()

top_p_label = Label(root, text="Top-p:")
top_p_label.pack()
top_p_entry = Entry(root)
top_p_entry.insert(0, "0.95")  # Default value
top_p_entry.pack()

top_k_label = Label(root, text="Top-k:")
top_k_label.pack()
top_k_entry = Entry(root)
top_k_entry.insert(0, "100")  # Default value.  Set to a reasonable default.
top_k_entry.pack()

new_token_label = Label(root, text="Max New Tokens:")
new_token_label.pack()
new_token_entry = Entry(root)
new_token_entry.insert(0, "8192")  # Default value.  Set to a reasonable default.
new_token_entry.pack()

# --- Text Box ---
chat_log = scrolledtext.ScrolledText(root, wrap=tk.WORD)
chat_log.pack(expand=True, fill="both")
chat_log.config(state=tk.DISABLED)  # Make it initially read-only

entry = scrolledtext.ScrolledText(root, wrap=tk.WORD, height=3)
entry.pack(expand=True, fill=tk.X, side=tk.BOTTOM)

def send_message():
    user_input = entry.get("1.0", tk.END).strip()
    if not user_input:
        return
    entry.delete("1.0", tk.END)
    logger.info(f"New user message: {user_input}")
    
    # Add user message to chat history
    messages.append({"role": "user", "content": user_input})
    try:
        temperature = float(temp_entry.get())
        top_p = float(top_p_entry.get())
        top_k = int(top_k_entry.get()) 
        new_token = int(new_token_entry.get())
        
        # Re-intialize HF Pipeline
        pipe = transformers.pipeline(
            model=model,
            tokenizer=tokenizer,
            return_full_text=False,
            task="text-generation",
            max_new_tokens=new_token,
            temperature=temperature,
            do_sample=True,
            top_p=top_p,
            top_k=top_k
        )
        
        logger.info(f"Intialized HF pipeline with: temp={temperature}, top-p={top_p}, top-k={top_k}, new_token={new_token}")

        gen_seqs = pipe(messages)
        response = "".join([seq["generated_text"] for seq in gen_seqs])
        messages.append({"role": "assistant", "content": response})
        chat_log.config(state=tk.NORMAL)
        chat_log.insert(tk.END, f"You:\t{user_input}\n")
        chat_log.insert(tk.END, f"AI:\t{response}\n")
        chat_log.insert(tk.END, "\n")
        chat_log.see(tk.END)  # Scroll to bottom
        chat_log.config(state=tk.DISABLED)
        
        logger.debug(f"Chat history: {messages}")
        
        # Prevent OOM
        gc.collect()
        torch.cuda.empty_cache()
    except Exception as e:
        chat_log.config(state=tk.NORMAL)
        chat_log.insert(tk.END, f"Error: {e}\n")
        chat_log.config(state=tk.DISABLED)

entry.bind("<Return>", lambda event: send_message())  # Enter key sends message

send_button = tk.Button(root, text="Send", command=send_message)
send_button.pack(side=tk.BOTTOM)

def reset_chat_history():
    global messages
    messages = [
        {"role": "system", "content": system_prompt},
    ]
    chat_log.config(state=tk.NORMAL)  # Make the chat log editable
    chat_log.delete("1.0", tk.END)  # Clear the chat log
    chat_log.config(state=tk.DISABLED) # Make it read-only again
    
send_button = tk.Button(root, text="Reset Chat History", command=reset_chat_history)
send_button.pack(side=tk.BOTTOM)

def on_closing():
    root.destroy()
    exit(0)

root.protocol("WM_DELETE_WINDOW", on_closing)

root.mainloop()