import gc
import logging
import os
import torch
from unsloth import FastLanguageModel
import torch
from trl import SFTTrainer
from transformers import TrainingArguments
from dataset_loader import load_text_folder_dataset

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Create & configure logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# Clean VRAM cahce
gc.collect()
torch.cuda.empty_cache()

"""
    Sets the maximum sequence length for the model. 
    If an input sequence exceeds max_seq_length, it will be truncated to that length. 
    This means the model will only see the first max_seq_length tokens of the input.
"""
max_seq_length = 1024  # Sets the maximum sequence length for the model
dtype = (
    None  # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
)
load_in_4bit = True  # Use 4bit quantization to reduce memory usage. Can be False.

# 1B for testing
# model_name = "meta-llama/Llama-3.2-1B-Instruct"

# 3B for semi-real applications
model_name = "meta-llama/Llama-3.2-3B-Instruct"

logger.info(f"Loading base model {model_name}")

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)

"""
    Use Parameter-Efficient Fine-Tuning (PEFT) techniques to the model. 
    This optimizes the fine-tuning process by only training a small subset of parameters, 
    reducing computational cost and memory usage.
    
    Refrence: https://huggingface.co/blog/peft
"""
model = FastLanguageModel.get_peft_model(
    model,
    # Rank=64 max for 8GB VRAM
    r=64,  # Specifies the rank for LoRA (Low Rank Adaptation)
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    lora_alpha=8,  # Specifies alpha paramters for LoRA
    lora_dropout=0,  # Supports any, but = 0 is optimized
    bias="none",  # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing=True,  # True or "unsloth" for very long context
    random_state=42,
    use_rslora=True,  # We support rank stabilized LoRA, refrence: https://huggingface.co/blog/damjan-k/rslora
    loftq_config=None,  # And LoftQ
)

logger.info(f"Model {model_name} and tokenizer loaded successfully")

logger.info("Loading dataset for training")

dataset = load_text_folder_dataset(
    data_dir="/home/bruno/Documents/GitHub/social-media-nlp/training/dataset_txt",
    chunk_size=256,
    overlap=64,
    field_name="text",
    split_name="train",
    tokenizer=tokenizer,
    max_tokens=max_seq_length,
)

logger.info(f"Loaded dataset with {len(dataset)} samples")

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    dataset_num_proc=10,  # Uses 10 processes (thread) for dataset processing, potentially speeding up data loading.
    packing=False,  # Can make training 5x faster for short sequences.
    args=TrainingArguments(
        num_train_epochs=5,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=4,
        warmup_steps=1,
        learning_rate=2e-4,  # smaller steps for DPO and ORPO - standard 2e-4 for finetune
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=1,
        optim="paged_adamw_8bit",  # "adamw_torch", #"paged_adamw_8bit",#paged_adamw_32bit
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=42,
        output_dir="/home/bruno/Documents/GitHub/social-media-nlp/training/outputs",
        do_eval=False,  # Do not evaluate the model during training, save time and resources.
        torch_empty_cache_steps=1,
        save_strategy="steps",
        save_total_limit=3,  # Retain the latest 3 checkpoint
        save_steps=10,  # Save a checkpoint every 100 steps
    ),
)

logger.info(f"SFTT trainer initialized: \n{trainer}\n")

logger.info("Start training")
trainer_stats = trainer.train()
logger.info("Trainng completed successfully")