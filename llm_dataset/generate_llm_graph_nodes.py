import json
import time
# from typing import Any
from sentence_transformers import SentenceTransformer
import torch
from tqdm import tqdm
import sys
import logging
import os
from torch_geometric.data import Data
import random

import gc
import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from custom_logger.logger_config import get_logger

log: logging.Logger = get_logger(name=__name__)

TENSOR_GRAPH_OUTDIR = "/home/bruno/Documents/GitHub/social-media-nlp/llm_dataset/only_node_datasets"

ROOT_DOMAIN_TOPICS: list[str] = [
    "Sustainable Development Goals 1: End poverty in all its forms everywhere",
    "Sustainable Development Goals 2: End hunger, achieve food security and improved nutrition and promote sustainable agriculture"
]

# Generated with gemini 
# Prompt 'Generate a list of 10 topic regarding to medicine and biology'
ROOT_NON_DOMAIN_TOPICS: list[str] = [
    "Genetic Engineering and CRISPR Technology: Explore the revolutionary potential of gene editing tools like CRISPR-Cas9 in treating genetic diseases, developing new therapies, and even enhancing human capabilities",
    "Immunotherapy and Cancer Treatment: Investigate the latest advancements in harnessing the immune system to fight cancer, including checkpoint inhibitors, CAR T-cell therapy, and personalized cancer vaccines",
    # "Neurodegenerative Diseases: Delve into the complexities of Alzheimer's disease, Parkinson's disease, and other neurological disorders, examining the underlying causes, potential treatments, and the challenges of developing effective therapies",
    # "Microbiome and Human Health: Discover the intricate world of the human microbiome the trillions of bacteria, fungi, and other microorganisms living in our bodies and its profound impact on digestion, immunity, mental health, and overall well-being",
    # "Stem Cell Research and Regenerative Medicine: Explore the therapeutic potential of stem cells in repairing damaged tissues and organs, treating diseases like diabetes and spinal cord injuries, and even growing new organs for transplantation",
    # "Artificial Intelligence in Healthcare: Examine the growing role of AI in medicine, from diagnosing diseases and analyzing medical images to developing new drugs and personalizing treatment plans",
    # "Epidemiology and Global Health: Study the patterns, causes, and control of diseases in populations, including infectious diseases like COVID-19, as well as chronic conditions like heart disease and cancer, with a focus on improving global health outcomes",
    # "Mental Health and Neuroscience: Investigate the biological basis of mental health disorders, such as depression, anxiety, and schizophrenia, and explore new approaches to diagnosis, treatment, and prevention",
    # "Aging and Longevity: Explore the biological processes of aging and the factors that contribute to healthy aging and longevity, including genetics, lifestyle, and environmental influences",
    # "Biotechnology and Drug Development: Learn about the cutting-edge technologies used to develop new drugs and therapies, including monoclonal antibodies, gene therapy, and personalized medicine approaches",
]

HF_MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
HF_EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

SYSTEM_PROMPT = """You are a large language model, trained to be informative and concise.
Do not provide explanations or elaborations unless explicitly requested.
Focus on delivering the most direct and accurate answer to the user's query."""

NUM_GEN_NODES = 10

def __get_prompt(root_topic: str, num_gen_nodes: int = NUM_GEN_NODES) -> str:
    return f"Generate a JSON list of {num_gen_nodes} arguments related to {root_topic}." + \
        "\n\nExample of response structure: [\"apple\", \"banana\", \"cherry\"]." + \
        "\nAvoid creating nested JSON object, the generated JSON object list must have only one level."

def __get_gen_nodes(llm_out: str) -> list[str]:
    start_idx = llm_out.index("[")
    end_idx = llm_out.index("]")
    json_str = llm_out[start_idx:end_idx + 1]
    json_loaded = json.loads(json_str)
    
    nodes: list[str] = []
    for item in json_loaded:
        nodes.append(item)
    
    log.info(f"Nodes generated from LLM: {nodes}")
    return nodes
    
def __initialize_llm(model_name: str = HF_MODEL_NAME) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=False,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        quantization_config=bnb_config, 
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    log.info(f"Initialized LLM: {model_name}")
    return model, tokenizer

def __generate_llm(
    model: AutoModelForCausalLM, 
    tokenizer: AutoTokenizer, 
    prompt: str,
    system_prompt: str = SYSTEM_PROMPT
) -> str:
    pipe = transformers.pipeline(
        model=model,
        tokenizer=tokenizer,
        return_full_text=False,
        task="text-generation",
        max_new_tokens=1024,
        temperature=0.7,
        do_sample=True,
        top_p=0.95,
        top_k=100
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]
    gen_seqs = pipe(messages)
    llm_response = "".join([seq["generated_text"] for seq in gen_seqs])
    log.info(f"Generate LLM response of length {len(llm_response)}")
    
    del pipe
    gc.collect()
    torch.cuda.empty_cache()
    return llm_response

def __generate_nodes(root_topic_list: list[str] = ROOT_DOMAIN_TOPICS) -> list[str]:
    all_nodes:list[str] = []
    model, tokenizer = __initialize_llm()
    for root_topic in root_topic_list:
        prompt = __get_prompt(root_topic)
        llm_response = __generate_llm(model, tokenizer, prompt)
        nodes = __get_gen_nodes(llm_response)
        all_nodes.extend(nodes)
    
    # Destory LLM model and clean VRAM
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    
    return all_nodes

def __save_graph_tensor(
    node_list: list[str], 
    save_path: str = TENSOR_GRAPH_OUTDIR,
    hf_embedding_model: str = HF_EMBEDDING_MODEL_NAME
) -> None:
    embedding_model: SentenceTransformer = SentenceTransformer(hf_embedding_model)
    log.info(f"Loaded embedding model {hf_embedding_model}")
    
    nodes_idx_map: dict[int, str] = {}
    nodes_embed_ord: list[torch.Tensor] = []
    
    random.shuffle(node_list)
    for node_label in tqdm(node_list, desc="Parsing LMM generated nodes"):
        nodes_idx_map[len(nodes_idx_map)] = node_label
        node_embed = embedding_model.encode(node_label)
        node_embed_tensor = torch.tensor(node_embed, dtype=torch.float)
        nodes_embed_ord.append(node_embed_tensor)
    
    x = torch.stack(nodes_embed_ord)
    edge_index = torch.tensor([[], []], dtype=torch.int64)
    data = Data(x=x, edge_index=edge_index)
    tensor_to_save = {
        "data": data,
        "nodes_idx_map": nodes_idx_map
    }
    
    tesor_file_name = f"llm_gen_graph_{len(node_list)}_{time.strftime('%Y%m%d-%H%M%S')}.pt"
    tensor_file_path = os.path.join(save_path, tesor_file_name)
    torch.save(obj=tensor_to_save, f=tensor_file_path)
    log.info(f"Graph of {len(node_list)} nodes saved to {tensor_file_path}")

if __name__ == "__main__":
    list_nodes = __generate_nodes(root_topic_list=ROOT_NON_DOMAIN_TOPICS)
    __save_graph_tensor(list_nodes)