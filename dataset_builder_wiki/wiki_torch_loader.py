import random
from typing import Tuple
import torch
from torch_geometric.data import Data
import sys
import logging
import os
import numpy as np
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from custom_logger.logger_config import get_logger

log: logging.Logger = get_logger(name=__name__)

# Use CPU to PyTorch Geometric incompatibility
device = torch.device('cpu')
log.info(f"Using torch device: {device}")

TENSOR_FOLDER = "/home/bruno/Documents/GitHub/social-media-nlp/dataset_builder_wiki/final_dataset/tensor"

class WikiBaseDataset(torch.utils.data.Dataset):
    
    def __init__(self, obj_lis: list[Tuple[Data, dict[int, str]]], shuffle: bool = False) -> None:
        self.__obj_list = obj_lis
        if shuffle:
            random.shuffle(self.__obj_list)
            
    def __len__(self) -> int:
        return len(self.__obj_list)   
    
    def __getitem__(self, idx:int) -> Data:
        return self.__obj_list[idx][0]
    
    def get_node_labels(self, idx:int) -> dict[int, str]:
        return self.__obj_list[idx][1]

    def shuffle(self) -> None:
        random.shuffle(self.__obj_list)


class WikiGraphDataset(torch.utils.data.Dataset):
    
    def __init__(self, tensor_folder: str = TENSOR_FOLDER, shuffle: bool = False, size_limit: int = -1) -> None:
        
        self.__obj_list: list[Tuple[Data, dict[int, str]]] = []
        
        log.info(f"Loading tensor data from {tensor_folder}")
        
        tensor_files: list[str] = os.listdir(tensor_folder)
        if len(tensor_files) == 0:
            raise ValueError(f"No tensor files found in {tensor_folder}")
        
        if size_limit > 0 and size_limit > len(tensor_files):
            raise ValueError(f"Size limit {size_limit} is greater than the number of tensor files {len(tensor_files)}")
        
        if shuffle:
            log.info("Shuffling data")
            random.shuffle(tensor_files)
        
        if size_limit > 0:
            log.info(f"Dataset size limited to {size_limit}")
            tensor_files = tensor_files[:size_limit]
        
        for filename in tqdm(tensor_files, desc="Loading tensor files"):
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
        
        log.info(f"Loaded {len(self.__obj_list)} data objects")
    
    def __len__(self) -> int:
        return len(self.__obj_list)   
    
    def __getitem__(self, idx:int) -> Data:
        return self.__obj_list[idx][0]
    
    def get_node_labels(self, idx:int) -> dict[int, str]:
        return self.__obj_list[idx][1]
    
    def shuffle(self) -> None:
        random.shuffle(self.__obj_list)
    
    def split_dataset(
        self,
        train_percentage: float = 0.2, 
        shuffle_before_split: bool = False,
        shuffle_after_split: bool = False
    ) -> Tuple[WikiBaseDataset, WikiBaseDataset]:
        
        data_len = len(self.__obj_list)
        train_size = int(data_len * train_percentage)
        
        data_copy = self.__obj_list.copy()
        if shuffle_before_split:
            random.shuffle(data_copy)
        
        train_set = data_copy[:train_size]
        test_set = data_copy[train_size:]
        
        if shuffle_after_split:
            random.shuffle(train_set)
            random.shuffle(test_set)
        
        return WikiBaseDataset(train_set), WikiBaseDataset(test_set)
        
        

# Test only
if __name__ == "__main__":
    dataset = WikiGraphDataset(tensor_folder="/home/bruno/Documents/GitHub/social-media-nlp/dataset_builder_wiki/final_dataset/tensor")
    for i, data in enumerate(dataset):
        log.info(f"[{i}] Edges tensor shape {data.edge_index.shape}")
        log.info(f"[{i}] Nodes features tensor shape {data.x.shape}")