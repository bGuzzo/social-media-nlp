from sentence_transformers import SentenceTransformer
from torch import Tensor
import torch
from torch_geometric.data import Data

model = SentenceTransformer("all-MiniLM-L6-v2")


def encode_phrase(phrase: str) -> Tensor:
    return model.encode(phrase)

# Define node features (you can use embeddings from an LLM or simpler representations)
# For simplicity, we'll use one-hot encoding here
no_poverty = torch.tensor(encode_phrase("No Poverty"), dtype=torch.float)
zero_hunger = torch.tensor(encode_phrase("Zero Hunger"), dtype=torch.float)
quality_education = torch.tensor(encode_phrase("Quality Education"), dtype=torch.float)
climate_action = torch.tensor(encode_phrase("Climate Action"), dtype=torch.float)
life_below_water = torch.tensor(encode_phrase("Life Below Water"), dtype=torch.float)
life_on_land = torch.tensor(encode_phrase("Life On Land"), dtype=torch.float)

x = torch.stack(
    [
        no_poverty,
        zero_hunger,
        quality_education,
        climate_action,
        life_below_water,
        life_on_land,
    ]
)

# Define the edges (connections between SDGs)
edge_index = torch.tensor(
    [[0, 1, 0, 2, 3, 4, 3, 5], 
     [1, 0, 2, 0, 4, 3, 5, 3]], dtype=torch.long
)

# Create a PyTorch Geometric Data object
data = Data(x=x, edge_index=edge_index)

print(data)
