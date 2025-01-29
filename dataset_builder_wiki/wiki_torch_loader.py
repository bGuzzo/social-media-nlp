import random
from typing import Tuple
import torch
from torch_geometric.data import Data
import sys
import logging
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from custom_logger.logger_config import get_logger

log: logging.Logger = get_logger(name=__name__)

# Use CPU to PyTorch Geometric incompatibility
device = torch.device('cpu')
log.info(f"Using torch device: {device}")

TENSOR_FOLDER = "/home/bruno/Documents/GitHub/social-media-nlp/dataset_builder_wiki/final_dataset/tensor"

class WikiDataset(torch.utils.data.Dataset):
    
    def __init__(self, tensor_folder: str = TENSOR_FOLDER, shuffle: bool = False):
        
        self.__obj_list: list[Tuple[Data, dict[int, str]]] = []
        
        log.info(f"Loading tensor data from {tensor_folder}")
        for filename in os.listdir(tensor_folder):
            if filename.endswith(".pt"):
                tensor_file_path = os.path.join(tensor_folder, filename)
                log.info(f"Loading tensor file {tensor_file_path}")
                
                try:
                    graph_tensor_dict = torch.load(tensor_file_path)
            
                    data: Data = graph_tensor_dict["data"]
                    data.to(device)
                    nodes_idx_map: dict[int, str] = graph_tensor_dict["nodes_idx_map"]

                    self.__obj_list.append((data, nodes_idx_map))
                except Exception as e:
                    log.error(f"Error loading tensor file {tensor_file_path}")
                    raise e
        
        if shuffle:
            log.info("Shuffling data")
            random.shuffle(self.__obj_list)
        
        log.info(f"Loaded {len(self.__obj_list)} data objects")
    
    def __len__(self) -> int:
        return len(self.__obj_list)   
    
    def __getitem__(self, idx:int) -> Data:
        return self.__obj_list[idx][0]
    
    def get_node_labels(self, idx:int) -> dict[int, str]:
        return self.__obj_list[idx][1]

# Test only
if __name__ == "__main__":
    dataset = WikiDataset(tensor_folder="/home/bruno/Documents/GitHub/social-media-nlp/dataset_builder_wiki/final_dataset/tensor")
    for i, data in enumerate(dataset):
        log.info(f"[{i}] Edges tensor shape {data.edge_index.shape}")
        log.info(f"[{i}] Nodes features tensor shape {data.x.shape}")