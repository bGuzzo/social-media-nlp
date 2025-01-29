import sys
import os
from wiki_dataset_builder import create_dataset
from wiki_graph_json_to_tensor import convert_jsons_to_tensors
import logging

# Import parent script
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from custom_logger.logger_config import get_logger

log: logging.Logger = get_logger(name=__name__)

MAX_NODES = 5000
JSON_FOLDER_PATH = "/home/bruno/Documents/GitHub/social-media-nlp/dataset_builder_wiki/final_dataset/json"
TESNOR_FOLDER_PATH = "/home/bruno/Documents/GitHub/social-media-nlp/dataset_builder_wiki/final_dataset/tensor"
HF_PHARESE_EMBEDDING_MODEL = "all-MiniLM-L6-v2"


# Genearted with Gemini 1.5 Pro
# Prompt: Generate a JSON list of 50 Wikipedia pages title about {topic}.
WIKI_ARTICLES = [
    
    # Sustinability
    "Sustainable development",
    "Environmental sustainability",
    "Social sustainability",
    "Economic sustainability",
    
    # "Sustainable agriculture",
    # "Sustainable energy",
    # "Sustainable transport",
    # "Sustainable cities",
    # "Sustainable consumption",
    # "Sustainable production",
    # "Circular economy",
    # "Green building",
    # "Renewable energy",
    # "Climate change mitigation",
    # "Biodiversity conservation",
    # "Water conservation",
    # "Waste management",
    # "Environmental ethics",
    # "Corporate social responsibility",
    # "Sustainable development goals",
]

def main():
    log.info("Start creating wiki dataset")
    create_dataset(
        wiki_articles=WIKI_ARTICLES,
        folder_path=JSON_FOLDER_PATH,
        max_nodes=MAX_NODES
    )
    log.info("Wiki dataset created successfully")
    
    log.info("Start converting wiki dataset to tensor")
    convert_jsons_to_tensors(
        json_folder=JSON_FOLDER_PATH,
        tensor_folder=TESNOR_FOLDER_PATH
    )
    log.info("Wiki dataset converted to tensor successfully")

if __name__ == "__main__":
    main()
