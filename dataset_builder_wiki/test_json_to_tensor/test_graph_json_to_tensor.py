import unittest
import torch
from torch_geometric.data import Data
import sys
import logging
import os

# Parent packages import
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from custom_logger.logger_config import get_logger
from dataset_builder_wiki.wiki_graph_json_to_tensor import convert_jsons_to_tensors

log: logging.Logger = get_logger(name=__name__)

# Use CPU to PyTorch Geometric incompatibility
device = torch.device('cpu')


class TestGraphJsonToTensor(unittest.TestCase):
    
    def test_json_to_tensor(self):
        test_folder = "/home/bruno/Documents/GitHub/social-media-nlp/dataset_builder_wiki/test_json_to_tensor"
        tensor_file_name = "Sustainable_development.pt"
        tensor_file_path = os.path.join(test_folder, tensor_file_name)
        
        if os.path.exists(tensor_file_path):
            log.warning(f"Found file {tensor_file_path}, deleted")
            os.remove(tensor_file_path)
        
        convert_jsons_to_tensors(json_folder=test_folder, tensor_folder=test_folder)
        graph_tensor_dict = torch.load(tensor_file_path)
        
        data: Data = graph_tensor_dict["data"]
        nodes_idx_map: dict[int, str] = graph_tensor_dict["nodes_idx_map"]
        
        log.info(f"Loaded data tensor file {data}, with {data.num_nodes} nodes, {data.num_edges} edges, directed graph {data.is_directed()}")
        log.info(f"Loaded nodes index map {nodes_idx_map}")
        
        if os.path.exists(tensor_file_path):
            log.info(f"Removing tensor file {tensor_file_path}")
            os.remove(path=tensor_file_path)

if __name__ == "__main__":
    unittest.main()