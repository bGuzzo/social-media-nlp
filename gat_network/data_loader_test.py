from sentence_transformers import SentenceTransformer
from torch import Tensor
import torch
from torch_geometric.data import Data
import sys
import logging
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from custom_logger.logger_config import get_logger

log: logging.Logger = get_logger(name=__name__)


def get_data() -> Data:
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Use CPU to PyTorch Geometric incompatibility
    device = torch.device('cpu')
    log.info(f"Using torch device: {device}")
    model.to(device)

    def encode_phrase(phrase: str) -> list:
        return model.encode(phrase)

    # Define node features (you can use embeddings from an LLM or simpler representations)
    # For simplicity, we'll use one-hot encoding here
    no_poverty = torch.tensor(encode_phrase("No Poverty"), dtype=torch.float, device=device)
    zero_hunger = torch.tensor(encode_phrase("Zero Hunger"), dtype=torch.float, device=device)
    quality_education = torch.tensor(encode_phrase("Quality Education"), dtype=torch.float, device=device)
    climate_action = torch.tensor(encode_phrase("Climate Action"), dtype=torch.float, device=device)
    life_below_water = torch.tensor(encode_phrase("Life Below Water"), dtype=torch.float, device=device)
    life_on_land = torch.tensor(encode_phrase("Life On Land"), dtype=torch.float, device=device)

    x = torch.stack(
        [
            no_poverty,
            zero_hunger,
            quality_education,
            climate_action,
            life_below_water,
            life_on_land,
        ]
    ).to(device)


    # Define the edges (connections between SDGs)
    edge_index = torch.tensor(
        [[0, 1, 0, 2, 3, 4, 3, 5], 
        [1, 0, 2, 0, 4, 3, 5, 3]], dtype=torch.long
    ).to(device)


    # Create a PyTorch Geometric Data object
    data = Data(x=x, edge_index=edge_index).to(device)
    log.info(f"Data loaded {data}")
    return data


if __name__ == "__main__":
    folder_path = "/home/bruno/Documents/GitHub/social-media-nlp/gat_network/tensor_dumps"
    tensor_file = os.path.join(folder_path, "data.pt")
    
    data: Data = get_data()
    torch.save(data, tensor_file)
    
    del data
    data: Data = torch.load(tensor_file, weights_only=False)
    # log.info(f"Data loaded {data}")
    
