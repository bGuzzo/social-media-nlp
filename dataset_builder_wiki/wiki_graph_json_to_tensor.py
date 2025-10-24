"""
This script is a critical component of the data preprocessing pipeline, responsible for
converting the graph dataset from a human-readable JSON format into a tensor-based
format optimized for PyTorch Geometric. This conversion is essential for efficient
data loading and processing during the training of the Graph Neural Network.

The script's main functionalities are:

1.  **Node Embedding**: The core of this script is the computation of node embeddings.
    It utilizes a pre-trained sentence-transformer model to convert the textual labels
    of the nodes into high-dimensional numerical vectors. This process transforms the
    semantic information of the nodes into a format that can be processed by a neural
    network.

2.  **Tensor Conversion**: The script takes the JSON representation of each graph, which
    includes a list of nodes and edges, and converts it into a PyTorch Geometric `Data`
    object. This object encapsulates:
    - `x`: A tensor containing the feature vectors (embeddings) of all nodes in the graph.
    - `edge_index`: A tensor that defines the connectivity of the graph by storing the
      source and target nodes of each edge.

3.  **Parallel Processing**: To handle the potentially large size of the dataset and the
    computationally intensive nature of node embedding, the script employs a multi-threaded
    approach. It uses a `ThreadPoolExecutor` to process multiple JSON files in parallel,
    significantly reducing the total time required for the conversion.

4.  **Data Persistence**: The resulting `Data` objects, along with a mapping from node indices
    to their original labels, are saved to disk as `.pt` files. This allows for fast and
    efficient loading of the pre-processed data during the model training phase, avoiding
    the need to re-compute the embeddings every time the model is trained.

This script represents a crucial step in bridging the gap between the raw, text-based
graph data and the numerical format required by deep learning frameworks.
"""

import multiprocessing
import concurrent.futures
from sentence_transformers import SentenceTransformer
import torch
from torch_geometric.data import Data
import sys
import logging
import os
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from custom_logger.logger_config import get_logger

log: logging.Logger = get_logger(name=__name__)

# Device configuration
device = torch.device('cpu')
log.info(f"Using torch device: {device}")

log.info(f"CPU cores available: {multiprocessing.cpu_count()}")

# Default folder paths
DEF_JSON_FOLDER = "/home/bruno/Documents/GitHub/social-media-nlp/dataset_builder_wiki/json_wiki_graph_dataset"
DEF_TENSOR_FOLDER = "/home/bruno/Documents/GitHub/social-media-nlp/dataset_builder_wiki/tensor_wiki_dataset"

# Default sentence-transformer model for node embedding
DEF_EMBED_MODEL_NAME = "all-MiniLM-L6-v2"

def __to_data_tensor(
    json_file_name: str, 
    json_graph: dict, 
    tensor_file_path: str, 
    embedding_model: SentenceTransformer
) -> None:
    """
    Converts a single JSON graph to a PyTorch Geometric `Data` object and saves it to a file.

    Args:
        json_file_name (str): The name of the JSON file being processed.
        json_graph (dict): The graph data in JSON format.
        tensor_file_path (str): The path to save the output tensor file.
        embedding_model (SentenceTransformer): The initialized sentence-transformer model.
    """
    
    if os.path.exists(tensor_file_path):
        log.warning(f"File {json_file_name} already present as tensor file {tensor_file_path}. Skipped")
        return
    
    nodes_idx_map: dict[int, str] = {}
    node_embed_ord: list[torch.Tensor] = []
    edge_src: list[int] = []
    edge_dst: list[int] = []
    
    for node in json_graph["nodes"]:
        nodes_idx_map[int(node["id"])] = str(node["label"])
    
    # Compute node embeddings
    for node_idx in sorted(nodes_idx_map.keys()):
        node_label = nodes_idx_map[node_idx]
        log.info(f"[{json_file_name}] - Parsing node, id: {node_idx}, label: {node_label}")
        node_embed = embedding_model.encode(node_label)
        node_embed_tensor = torch.tensor(node_embed, dtype=torch.float, device=device)
        node_embed_ord.append(node_embed_tensor)
    
    x = torch.stack(node_embed_ord).to(device)
    
    # Create an undirected graph by adding both forward and backward edges
    for edge in json_graph["edges"]:
        log.info(f"[{json_file_name}] - Parsing edge {edge}")
        edge_src.extend([int(edge["source"]), int(edge["target"])])
        edge_dst.extend([int(edge["target"]), int(edge["source"])])

    edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long).to(device)
    
    data = Data(x=x, edge_index=edge_index).to(device)
    log.info(f"[{json_file_name}] - Tensor graph loaded successfully with {data.num_nodes} nodes and {data.num_edges} edges")
    
    tensor_to_save = {"data": data, "nodes_idx_map": nodes_idx_map}
    torch.save(obj=tensor_to_save, f=tensor_file_path)
    log.info(f"Graph JSON file {json_file_name} parsed SUCCESSFULLY to tensor file {tensor_file_path}")

def convert_jsons_to_tensors(
    hf_embedding_model: str = DEF_EMBED_MODEL_NAME,  
    json_folder: str = DEF_JSON_FOLDER, 
    tensor_folder: str = DEF_TENSOR_FOLDER
) -> None:
    """
    Orchestrates the conversion of the entire JSON graph dataset to tensor format using multiple threads.

    Args:
        hf_embedding_model (str, optional): The name of the Hugging Face sentence-transformer model. Defaults to DEF_EMBED_MODEL_NAME.
        json_folder (str, optional): The folder containing the JSON graph files. Defaults to DEF_JSON_FOLDER.
        tensor_folder (str, optional): The folder to save the output tensor files. Defaults to DEF_TENSOR_FOLDER.
    """
    
    embedding_model: SentenceTransformer = SentenceTransformer(hf_embedding_model)
    log.info(f"Loading JSON data from {json_folder}")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        future_list = [executor.submit(__to_data_tensor, filename, json.load(open(os.path.join(json_folder, filename))), os.path.join(tensor_folder, filename.replace(".json", ".pt")), embedding_model) for filename in os.listdir(json_folder) if filename.endswith(".json")]
        
        log.info("Awaiting all workers to complete")
        for future in concurrent.futures.as_completed(future_list):
            future.result() # Block and wait for completion
            
    log.info("Execution completed")

if __name__ == "__main__":
    # This block initiates the conversion process for the entire dataset.
    convert_jsons_to_tensors()
