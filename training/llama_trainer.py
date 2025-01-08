import gc
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
from unsloth import FastLanguageModel
import torch
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments
import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

# Clean VRAM cahce
gc.collect()
torch.cuda.empty_cache()

"""
    Sets the maximum sequence length for the model. 
    If an input sequence exceeds max_seq_length, it will be truncated to that length. 
    This means the model will only see the first max_seq_length tokens of the input.
"""
max_seq_length = 2048 # Sets the maximum sequence length for the model
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.

# 1B for testing 
# model_name = "meta-llama/Llama-3.2-1B-Instruct"

# 3B for semi-real applications
model_name = "meta-llama/Llama-3.2-3B-Instruct"

print("Loading Model... \t\t START")

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_name,
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit
)


"""
    Use Parameter-Efficient Fine-Tuning (PEFT) techniques to the model. 
    This optimizes the fine-tuning process by only training a small subset of parameters, 
    reducing computational cost and memory usage.
    
    Refrence: https://huggingface.co/blog/peft
"""
model = FastLanguageModel.get_peft_model(
    model,
    r = 8, # Specifies the rank for LoRA (Low Rank Adaptation)
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 8, # Specifies alpha paramters for LoRA
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing = True, # True or "unsloth" for very long context
    random_state = 42,
    use_rslora = True,  # We support rank stabilized LoRA, refrence: https://huggingface.co/blog/damjan-k/rslora
    loftq_config = None, # And LoftQ
)

print("Loading Model... \t\t COMPLETE")

print("\nLoading Dataset... \t\t START")

# # Shuffles the dataset to ensure randomness and prevent any biases during training.
# dataset = load_dataset("wikimedia/wikipedia", "20231101.it", split="train")
# dataset = dataset.shuffle(seed=42)  # Set a seed for reproducibility
# n_samples = int(len(dataset) * 1.0) # Use 100% of the dataset
# dataset = dataset.select(range(n_samples))

# Load the dataset from text the folder
dataset = load_dataset("text", data_dir="/home/bruno/Documents/GitHub/social-media-nlp/training/dataset_txt", split="train")

print(f"Dataset loaded with size {len(dataset)}")
print(dataset)

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 10, # Uses 10 processes (thread) for dataset processing, potentially speeding up data loading.
    packing = False, # Can make training 5x faster for short sequences.
    args = TrainingArguments(
        num_train_epochs=10, # Only One full pass through the dataset
        per_device_train_batch_size = 8,
        gradient_accumulation_steps = 4,
        warmup_steps = 1,
        learning_rate = 2e-4, #smaller steps for DPO and ORPO - standard 2e-4 for finetune
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 1,
        optim = "paged_adamw_8bit",#"adamw_torch", #"paged_adamw_8bit",#paged_adamw_32bit
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 42,
        output_dir = "/home/bruno/Documents/GitHub/social-media-nlp/training/outputs",
        do_eval=False, # Do not evaluate the model during training, save time and resources.
        torch_empty_cache_steps=1,
        save_strategy="steps",
        save_total_limit=3, # Retain the latest 3 checkpoint
        save_steps=100 # Save a checkpoint every 100 steps
    ),
)

print("\nLoading Dataset... \t\t COMPLETE")

print('Training... \t\t START')
trainer_stats = trainer.train()
print('Training... \t\t END')
