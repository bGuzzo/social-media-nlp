"""
This script serves as the main entry point for the entire dataset creation pipeline.
It is designed to be a simple, high-level script that orchestrates the two major
stages of the data acquisition and preprocessing workflow:

1.  **JSON Graph Dataset Creation**: The first stage involves the creation of the graph
    dataset in a human-readable JSON format. This is achieved by invoking the
    `create_dataset` function from the `wiki_dataset_builder` module. This function
    manages the process of crawling Wikipedia and constructing the graphs.

2.  **Tensor Dataset Conversion**: Once the JSON dataset has been created, the second
    stage is to convert it into a tensor-based format that is optimized for use
    with PyTorch Geometric. This is handled by the `convert_jsons_to_tensors` function
    from the `wiki_graph_json_to_tensor` module. This function computes the node
    embeddings and structures the data into a format that can be efficiently loaded
    during model training.

By encapsulating the entire data creation process within this single script, it provides
a convenient and reproducible way to generate the dataset from scratch. The script is
configured with the necessary parameters, such as file paths and model names, ensuring
that the data pipeline is executed in a consistent and well-defined manner.
"""

import sys
import os
from wiki_dataset_builder import create_dataset
from wiki_graph_json_to_tensor import convert_jsons_to_tensors
import logging

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from custom_logger.logger_config import get_logger

log: logging.Logger = get_logger(name=__name__)

# Configuration parameters for the dataset creation process
MAX_NODES = 20000
JSON_FOLDER_PATH = "/home/bruno/Documents/GitHub/social-media-nlp/dataset_builder_wiki/final_dataset/json"
TESNOR_FOLDER_PATH = "/home/bruno/Documents/GitHub/social-media-nlp/dataset_builder_wiki/final_dataset/tensor"
WIKI_ARTICLES_FILE = "/home/bruno/Documents/GitHub/social-media-nlp/dataset_builder_wiki/wikipedia_articles.txt"
HF_PHARESE_EMBEDDING_MODEL = "all-MiniLM-L6-v2"

def main():
    """
    The main function that orchestrates the entire dataset creation process.
    """
    log.info("Start creating wiki dataset")
    # Stage 1: Create the graph dataset in JSON format.
    create_dataset(
        wiki_file_path=WIKI_ARTICLES_FILE,
        folder_path=JSON_FOLDER_PATH,
        max_nodes=MAX_NODES,
    )
    log.info("Wiki dataset created successfully")
    
    # Stage 2: Convert the JSON dataset to tensor format.
    log.info("Start converting wiki dataset to tensor")
    convert_jsons_to_tensors(
        json_folder=JSON_FOLDER_PATH,
        tensor_folder=TESNOR_FOLDER_PATH
    )
    log.info("Wiki dataset converted to tensor successfully")

if __name__ == "__main__":
    # This block ensures that the main function is called only when the script is executed directly.
    main()