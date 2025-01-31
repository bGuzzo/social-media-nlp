from sentence_transformers import SentenceTransformer
import torch
from torch_geometric.data import Data
import sys
import logging
import os
from tqdm import tqdm
from gnn_model_v1.gat_net_module import GatModule
import matplotlib.pyplot as plt
import networkx as nx

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from custom_logger.logger_config import get_logger

log: logging.Logger = get_logger(name=__name__)

# Use CPU to PyTorch Geometric incompatibility
device = torch.device('cpu')
log.info(f"Using torch device: {device}")

DEF_EMBED_MODEL_NAME = "all-MiniLM-L6-v2"

MODEL_PATH = "/home/bruno/Documents/GitHub/social-media-nlp/gat_network/model_dumps/gnn_model_v1_attn_1x4_sizes_384_512_256_final_20250131-070307.pth"


# Model parameters
INPUT_SIZE = 384 # Embedding model output size
HIDDEN_SIZE = 512
OUTPUT_SIZE = 256
NUM_ATTENTION_LAYER = 1
NUM_ATTENTION_HEAD = 4

NODE_SAMPLE = [
    # SDG1
    "Eradicating extreme poverty",
    "Ending hunger",
    "Improving food security",
    "Reducing child mortality",
    "Eradicating hunger",
    "Ensuring food security",
    "Achieving full employment",
    "Providing decent work",
    "Combating poverty",
    "Reducing inequality",
    "Protecting human rights",
    "Improving access to services",
    "Ensuring healthcare access",
    "Supporting small-scale farmers",
    "Providing social services",
    "Promoting gender equality",
    "Ensuring access to education",
    "Eliminating child labor",
    "End poverty",
    "Ending social isolation",
    "Improving living standards",
    "Enhancing disaster risk reduction",
    "Supporting economic growth",
    "Improving nutrition",
    "Providing clean water",
    "Eradicating hunger and malnutrition",
    "Ensuring food security",
    "Reducing inequality and injustice",
    
    # SDG 2:
    "Improved agricultural productivity",
    "Reduce post-harvest losses",
    "Increase access to nutrition information",
    "Use of drought-tolerant crops",
    "Support small-scale farmers",
    "Food waste reduction",
    "Promote agroecology",
    "Water conservation",
    "Soil conservation",
    "Sustainable irrigation systems",
    "Integrated pest management",
    "Organic farming",
    "Farm-to-school programs",
    "School gardens",
    "Urban agriculture",
    "Food storage and preservation",
    "Sustainable livestock production",
    "Animal husbandry",
    "Irrigation management",
    "Crop rotation",
    "Polycultures",
    "Agroforestry",
    "Integrated Pest and Disease Management",
    "Crop insurance",
    "Food assistance programs",
    "Nutrition education",
    "Food labeling",
    "Carbon sequestration",
]

def convert_to_tensor(node_list:list[str] = NODE_SAMPLE, embedding_model_name:str = DEF_EMBED_MODEL_NAME) -> Data:
    
    # Initialize embedding model
    embedding_model: SentenceTransformer = SentenceTransformer(embedding_model_name)
    
    node_embed_ord: list[torch.Tensor] = []
    
    for node_text in tqdm(node_list, desc="Parsing LMM generated nodes"):
        node_embed = embedding_model.encode(node_text)
        node_embed_tensor = torch.tensor(node_embed, dtype=torch.float, device=device)
        node_embed_ord.append(node_embed_tensor)

    # Build node features tensor matrix
    x = torch.stack(node_embed_ord).to(device)
    
    # Build empty edge index tensor matrix 
    edge_index = torch.tensor([[], []], dtype=torch.long).to(device)
    
    return Data(x=x, edge_index=edge_index).to(device)

def load_model(model_path:str = MODEL_PATH) -> GatModule:
    return torch.load(model_path)

def extract_edges(adj_matrix, threshold):
    """
    Extracts edges from a probability adjacency matrix above a given threshold.

    Args:
        adj_matrix: A PyTorch tensor representing the adjacency matrix.
        threshold: The probability threshold above which to consider an edge.

    Returns:
        A tuple containing two tensors:
            - row_indices: The row indices of the extracted edges.
            - col_indices: The column indices of the extracted edges.
            If no edges are found above the threshold, empty tensors are returned.
    """

    # Efficiently find indices above the threshold
    row_indices, col_indices = torch.where(adj_matrix > threshold)

    return row_indices, col_indices

@torch.no_grad()
def main(node_label_list: list[str] = NODE_SAMPLE):
    data = convert_to_tensor()
    model = load_model()
    model.eval()
    
    z = model.encode(data.x, data.edge_index)
    score_matrix = model.decode_all(z)
    adj_prob_matrix = torch.sigmoid(score_matrix)
    
    # Build graph to visualize
    graph = nx.Graph()
    
    for label in node_label_list:
        graph.add_node(label)
    
    
    # predicted_edges = (adj_prob_matrix > 2).nonzero(as_tuple=False).t()
    # log.info(f"Predicted edges shape {predicted_edges.shape}")
    # log.info(f"Predicted edges: {predicted_edges}")
    
    row_indices, col_indices = extract_edges(adj_prob_matrix, 0.9)
    if row_indices.numel() > 0:  # Check if any edges were found
        # print("Edges above threshold:")
        
        for i in range(row_indices.size(0)):
            row = row_indices[i].item()
            col = col_indices[i].item()
            if row != col:
                prob = adj_prob_matrix[row, col].item()
                # print(f"Edge: {row} -> {col}, Probability: {prob}")
                graph.add_edge(node_label_list[row], node_label_list[col])
    else:
        print("No edges found above the threshold.")
    
    
    # Draw the graph (adjust layout as needed)
    plt.figure(figsize=(12, 12))
    nx.draw(graph, with_labels=True, node_size=300, font_size=8, node_color="skyblue")
    plt.title("Environmental sustainability")
    plt.show()
    
    
    # Plot the tensor as an image with a colorbar
    plt.imshow(adj_prob_matrix, cmap='viridis', interpolation='nearest') # 'viridis' is a common colormap
    plt.colorbar(label='Value')  # Add a colorbar with a label

    # Customize the plot (optional)
    plt.title('Tensor Visualization')
    plt.xlabel('Column Index')
    plt.ylabel('Row Index')
    
    plt.show()
    
    # print(adj_prob_matrix)
    # print(f""predicted_edges)

if __name__ == "__main__":
    main()