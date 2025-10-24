"""
This script is dedicated to the loading and management of the graph dataset for use with
PyTorch and PyTorch Geometric. It defines two main classes, `WikiBaseDataset` and
`WikiGraphDataset`, which provide a structured and efficient way to handle the dataset,
including loading from files, splitting into training and testing sets, and providing
the necessary interfaces for integration with PyTorch's data loading utilities.

The script’s primary contributions are:

1.  **`WikiGraphDataset`**: This is the main dataset class. It is responsible for:
    - Loading graph data from a directory of pre-processed tensor files (`.pt`).
    - Handling large datasets by allowing for size limiting and shuffling.
    - Providing a method (`split_dataset`) to partition the data into training and
      testing sets, which is a critical step for model evaluation.

2.  **`WikiBaseDataset`**: This is a more fundamental dataset class that serves as a container
    for the partitioned data (i.e., the training and testing sets). It inherits from
    `torch.utils.data.Dataset` and implements the essential methods (`__len__` and
    `__getitem__`) required by PyTorch's `DataLoader`.

By encapsulating the data loading and splitting logic within these classes, the script
promotes a clean and modular design, separating the data handling concerns from the
model training and evaluation code. This is a standard practice in machine learning
engineering that enhances code readability, reusability, and maintainability.
"""

import random
from typing import Tuple
import torch
from torch_geometric.data import Data
import sys
import logging
import os
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from custom_logger.logger_config import get_logger

log: logging.Logger = get_logger(name=__name__)

# Device configuration
device = torch.device('cpu')
log.info(f"Using torch device: {device}")

TENSOR_FOLDER = "/home/bruno/Documents/GitHub/social-media-nlp/dataset_builder_wiki/final_dataset/tensor"

class WikiBaseDataset(torch.utils.data.Dataset):
    """
A base dataset class for holding and accessing the graph data.

    Args:
        obj_lis (list[Tuple[Data, dict[int, str]]]): A list of tuples, where each tuple contains a PyTorch Geometric `Data` object and a node label map.
        shuffle (bool, optional): Whether to shuffle the dataset upon initialization. Defaults to False.
    """
    
    def __init__(self, obj_lis: list[Tuple[Data, dict[int, str]]], shuffle: bool = False) -> None:
        self.__obj_list = obj_lis
        if shuffle:
            random.shuffle(self.__obj_list)
            
    def __len__(self) -> int:
        """Returns the number of graphs in the dataset."""
        return len(self.__obj_list)   
    
    def __getitem__(self, idx:int) -> Data:
        """Returns the graph data object at the given index."""
        return self.__obj_list[idx][0]
    
    def get_node_labels(self, idx:int) -> dict[int, str]:
        """Returns the node label map for the graph at the given index."""
        return self.__obj_list[idx][1]

    def shuffle(self) -> None:
        """Shuffles the dataset in-place."""
        random.shuffle(self.__obj_list)

class WikiGraphDataset(torch.utils.data.Dataset):
    """
    The main dataset class for loading the Wikipedia graph dataset from tensor files.

    Args:
        tensor_folder (str, optional): The folder containing the tensor files. Defaults to TENSOR_FOLDER.
        shuffle (bool, optional): Whether to shuffle the files before loading. Defaults to False.
        size_limit (int, optional): The maximum number of graphs to load. Defaults to -1 (no limit).
    """
    
    def __init__(self, tensor_folder: str = TENSOR_FOLDER, shuffle: bool = False, size_limit: int = -1) -> None:
        
        self.__obj_list: list[Tuple[Data, dict[int, str]]] = []
        
        log.info(f"Loading tensor data from {tensor_folder}")
        
        tensor_files: list[str] = os.listdir(tensor_folder)
        if not tensor_files:
            raise ValueError(f"No tensor files found in {tensor_folder}")
        
        if 0 < size_limit > len(tensor_files):
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
                    data: Data = graph_tensor_dict["data"].to(device)
                    nodes_idx_map: dict[int, str] = graph_tensor_dict["nodes_idx_map"]
                    self.__obj_list.append((data, nodes_idx_map))
                except Exception as e:
                    log.error(f"Error loading tensor file {tensor_file_path}")
                    raise e
        
        log.info(f"Loaded {len(self.__obj_list)} data objects")
    
    def __len__(self) -> int:
        """Returns the number of graphs in the dataset."""
        return len(self.__obj_list)   
    
    def __getitem__(self, idx:int) -> Data:
        """Returns the graph data object at the given index."""
        return self.__obj_list[idx][0]
    
    def get_node_labels(self, idx:int) -> dict[int, str]:
        """Returns the node label map for the graph at the given index."""
        return self.__obj_list[idx][1]
    
    def shuffle(self) -> None:
        """Shuffles the dataset in-place."""
        random.shuffle(self.__obj_list)
    
    def split_dataset(
        self,
        train_percentage: float = 0.2, 
        shuffle_before_split: bool = False,
        shuffle_after_split: bool = False
    ) -> Tuple[WikiBaseDataset, WikiBaseDataset]:
        """
        Splits the dataset into training and testing sets.

        Args:
            train_percentage (float, optional): The percentage of the dataset to be used for training. Defaults to 0.2.
            shuffle_before_split (bool, optional): Whether to shuffle the dataset before splitting. Defaults to False.
            shuffle_after_split (bool, optional): Whether to shuffle the training and testing sets after splitting. Defaults to False.

        Returns:
            Tuple[WikiBaseDataset, WikiBaseDataset]: A tuple containing the training and testing datasets.
        """
        
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
        
if __name__ == "__main__":
    # This block demonstrates how to load the dataset and inspect its properties.
    dataset = WikiGraphDataset(tensor_folder=TENSOR_FOLDER)
    for i, data in enumerate(dataset):
        log.info(f"[{i}] Edges tensor shape {data.edge_index.shape}")
        log.info(f"[{i}] Nodes features tensor shape {data.x.shape}")
