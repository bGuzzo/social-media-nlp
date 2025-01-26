import torch
from torch_geometric.data import Data

# Define node features (you can use embeddings from an LLM or simpler representations)
# For simplicity, we'll use one-hot encoding here
no_poverty = torch.tensor([1, 0, 0, 0, 0], dtype=torch.float)
zero_hunger = torch.tensor([0, 1, 0, 0, 0], dtype=torch.float)
quality_education = torch.tensor([0, 0, 1, 0, 0], dtype=torch.float)
climate_action = torch.tensor([0, 0, 0, 1, 0], dtype=torch.float)
life_below_water = torch.tensor([0, 0, 0, 0, 1], dtype=torch.float)
life_on_land = torch.tensor(
    [0, 0, 0, 0, 1], dtype=torch.float
)  # Example with duplicate feature vector

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
