from typing import Tuple
from sentence_transformers import SentenceTransformer
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
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

DATA_FOLDER = "/home/bruno/Documents/GitHub/social-media-nlp/dataset_builder_wiki/json_wiki_graph_dataset"

class WikiDataset(torch.utils.data.Dataset):
    
    def __load_data_json(self, index:int, json_graph: dict) -> Tuple[Data, dict[int, str]]:
        
        # Key: node_id, Value: node_label
        nodes_idx_map: dict[int, str] = {}
        node_embed_ord: list[torch.Tensor] = []
        
        edge_src: list[int] = []
        edge_dst: list[int] = []
        
        for node in json_graph["nodes"]:
            nodes_idx_map[int(node["id"])] = str(node["label"])
        
        for node_idx in sorted(nodes_idx_map.keys()):
            node_label = nodes_idx_map[node_idx]
            log.info(f"[{self.__json_file_list[index]}] - Parsing node, id: {node_idx}, label: {node_label}")
            
            # Compute node embeddings using NLP
            node_embed = self.__embedding_model.encode(node_label)
            node_embed_tensor = torch.tensor(node_embed, dtype=torch.float, device=device)
            node_embed_ord.append(node_embed_tensor)
        
        # Build node features tensor matrix
        x = torch.stack(node_embed_ord).to(device)
        
        # Load non-oriented graph
        for edge in json_graph["edges"]:
            log.info(f"[{self.__json_file_list[index]}] - Parsing edge {edge}")
            
            # Forward edge
            edge_src.append(int(edge["source"]))
            edge_dst.append(int(edge["target"]))
            
            # Backward edge
            edge_src.append(int(edge["target"]))
            edge_dst.append(int(edge["source"]))

        # Build edge index tensor matrix
        edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long).to(device)
        
        data = Data(x=x, edge_index=edge_index).to(device)
        log.info(f"[{self.__json_file_list[index]}] - Tensor graph loaded successfully with {data.num_nodes} nodes and {data.num_edges} edges")
        
        return (data, nodes_idx_map)
    
    
    def __init__(self, data_folder: str = DATA_FOLDER):
        
        # Initialize embedding model
        self.__embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        
        # JSON string graph data
        self.__json_list: list[dict] = [] 
        
        # JSON file names
        self.__json_file_list: list[str] = []
        
        # PyTorch Geometric Data object
        self.__data_list: list[Data] = []
        
        # Node labels 
        self.__node_labels: list[dict[int, str]] = []
        
        log.info(f"Loading data from {data_folder}")
        for filename in os.listdir(data_folder):
            if filename.endswith(".json"):
                with open(os.path.join(data_folder, filename), "r") as json_file:
                    log.info(f"Reading file {filename}")
                    try:
                        self.__json_list.append(json.load(json_file))
                        self.__json_file_list.append(filename)
                        log.info(f"File {filename} loaded with index {len(self.__json_list) -1}")
                    except Exception as e:
                        log.error(f"Error reading file {filename}. File skipped", e)
                        raise e
        log.info(f"Loaded {len(self.__json_list)} files")
        
        for i, json_dict in enumerate(self.__json_list):
            try:
                data_graph, node_labels = self.__load_data_json(index=i, json_graph=json_dict)
                self.__data_list.append(data_graph)
                self.__node_labels.append(node_labels)
                log.info(f"File {self.__json_file_list[i]} loaded, build graph with {data_graph.num_nodes} nodes and {data_graph.num_edges/2} edges")
            except Exception as e:
                log.error(f"Error loading graph from file {self.__json_file_list[i]} at internal index {i}. Graph skipped", e)
                raise e
        log.info(f"Loaded {len(self.__data_list)} data objects")
    
    def __len__(self) -> int:
        return len(self.__data_list)   
    
    def __getitem__(self, idx:int) -> Data:
        return self.__data_list[idx]
    
    def get_node_labels(self, idx:int) -> dict[int, str]:
        return self.__node_labels[idx]
    
    def get_current_file(self, idx:int) -> str:
        return self.__json_file_list[idx]
    


# Test only
if __name__ == "__main__":
    dataset = WikiDataset(data_folder="/home/bruno/Documents/GitHub/social-media-nlp/dataset_builder_wiki/json_wiki_graph_test")
    for i, data in enumerate(dataset):
        cur_file = dataset.get_current_file(i)
        log.info(f"{cur_file} - Edges {data.edge_index}")
        log.info(f"{cur_file} - Nodes features shape {data.x.shape}")
        log.info(f"{cur_file} - Nodes labels {dataset.get_node_labels(i)}")