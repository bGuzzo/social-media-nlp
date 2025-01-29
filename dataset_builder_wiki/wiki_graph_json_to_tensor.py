import multiprocessing
import concurrent
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

# Use CPU to PyTorch Geometric incompatibility
device = torch.device('cpu')
log.info(f"Using torch device: {device}")

log.info(f"CPU cores available {multiprocessing.cpu_count()}")

DEF_JSON_FOLDER = "/home/bruno/Documents/GitHub/social-media-nlp/dataset_builder_wiki/json_wiki_graph_dataset"
DEF_TENSOR_FOLDER = "/home/bruno/Documents/GitHub/social-media-nlp/dataset_builder_wiki/tensor_wiki_dataset"

DEF_EMBED_MODEL_NAME = "all-MiniLM-L6-v2"


def __to_data_tensor(
    json_file_name:str, 
    json_graph: dict, 
    tensor_file_path: str, 
    embedding_model: SentenceTransformer
) -> None:
    
    # Key: node_id, Value: node_label
    nodes_idx_map: dict[int, str] = {}
    node_embed_ord: list[torch.Tensor] = []
    
    edge_src: list[int] = []
    edge_dst: list[int] = []
    
    for node in json_graph["nodes"]:
        nodes_idx_map[int(node["id"])] = str(node["label"])
    
    for node_idx in sorted(nodes_idx_map.keys()):
        node_label = nodes_idx_map[node_idx]
        log.debug(f"[{json_file_name}] - Parsing node, id: {node_idx}, label: {node_label}")
        
        if (node_label.find("\\") != -1):
            log.error(f"Found \\ in node label {node_label}")
            raise Exception(f"Found \\ in node label {node_label}")
        
        # Compute node embeddings using NLP
        node_embed = embedding_model.encode(node_label)
        node_embed_tensor = torch.tensor(node_embed, dtype=torch.float, device=device)
        node_embed_ord.append(node_embed_tensor)
    
    # Build node features tensor matrix
    x = torch.stack(node_embed_ord).to(device)
    
    # Load non-oriented graph
    for edge in json_graph["edges"]:
        log.debug(f"[{json_file_name}] - Parsing edge {edge}")
        
        # Forward edge
        edge_src.append(int(edge["source"]))
        edge_dst.append(int(edge["target"]))
        
        # Backward edge
        edge_src.append(int(edge["target"]))
        edge_dst.append(int(edge["source"]))

    # Build edge index tensor matrix
    edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long).to(device)
    
    data = Data(x=x, edge_index=edge_index).to(device)
    log.info(f"[{json_file_name}] - Tensor graph loaded successfully with {data.num_nodes} nodes and {data.num_edges} edges")
    
    tensor_to_save = {
        "data": data,
        "nodes_idx_map": nodes_idx_map
    }
    torch.save(obj=tensor_to_save, f=tensor_file_path)
    log.info(f"Graph JSON file {json_file_name} parsed SUCCESSFULLY to tensor file {tensor_file_path}")
    
    
def convert_jsons_to_tensors(
    hf_embedding_model: str = DEF_EMBED_MODEL_NAME,  
    json_folder: str = DEF_JSON_FOLDER, 
    tensor_folder: str = DEF_TENSOR_FOLDER
) -> None:
    
    # Thread list
    future_list: list[concurrent.futures.Future] = []
    
    # Initialize embedding model
    embedding_model: SentenceTransformer = SentenceTransformer(hf_embedding_model)
    
    log.info(f"Loading JSON data from {json_folder}")
    # Adjust: use the current numer of available CPU core 
    with concurrent.futures.ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        for filename in os.listdir(json_folder):
            if filename.endswith(".json"):
                    with open(os.path.join(json_folder, filename), "r") as json_file:
                        
                        log.info(f"Reading JSON file {filename}")
                        try:
                            json_dict = json.load(json_file)
                        except Exception as e:
                            log.error(f"Unable to load file {filename}. File skipped", e)
                            continue
                        
                        tensor_file_name = filename.replace(".json", ".pt")
                        tensor_file_path = os.path.join(tensor_folder, tensor_file_name)
                        if os.path.exists(tensor_file_path):
                            log.warning(f"File {filename} already present as tensor file {tensor_file_name}")
                            continue
                        
                        log.info(f"Start concurrent parsing JSON file {filename} to tensor")
                        future = executor.submit(__to_data_tensor, filename, json_dict, tensor_file_path, embedding_model)
                        future_list.append(future)
    
    log.info("Await all workers to complete")
    for future in concurrent.futures.as_completed(future_list):
        # Blocking wait
        future.result()
    log.info("Execution completed")


# Convert entire JSON Graph dataset to tensor
if __name__ == "__main__":
    convert_jsons_to_tensors()