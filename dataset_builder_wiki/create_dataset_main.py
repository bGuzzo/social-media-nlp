import sys
import os
from wiki_dataset_builder import create_dataset
from wiki_graph_json_to_tensor import convert_jsons_to_tensors
import logging

# Import parent script
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from custom_logger.logger_config import get_logger

log: logging.Logger = get_logger(name=__name__)

MAX_NODES = 20000
JSON_FOLDER_PATH = "/home/bruno/Documents/GitHub/social-media-nlp/dataset_builder_wiki/final_dataset/json"
TESNOR_FOLDER_PATH = "/home/bruno/Documents/GitHub/social-media-nlp/dataset_builder_wiki/final_dataset/tensor"
WIKI_ARTICLES_FILE = "/home/bruno/Documents/GitHub/social-media-nlp/dataset_builder_wiki/wikipedia_articles.txt" 
HF_PHARESE_EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Main method to strat the creation of the dataset
def main():
    log.info("Start creating wiki dataset")
    create_dataset(
        wiki_file_path=WIKI_ARTICLES_FILE, # Use a file with articles title
        folder_path=JSON_FOLDER_PATH,
        max_nodes=MAX_NODES,
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
