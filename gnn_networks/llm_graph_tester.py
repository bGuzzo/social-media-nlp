from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score
import torch
import sys
import logging
import os
from torch_geometric.data import Data
import networkx as nx
from gnn_model_v1.gat_net_module_v1 import GatModelV1 

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from custom_logger.logger_config import get_logger

log: logging.Logger = get_logger(name=__name__)

# Path to graphs and model
MODEL_PATH = "/home/bruno/Documents/GitHub/social-media-nlp/gnn_networks/model_dumps/gat_model_v1_ReLU_LayerNorm_128_64_4x2_d_0.75_315_final_2_20250202-134024.pth"
LLM_GRAPH_DOMAIN = "/home/bruno/Documents/GitHub/social-media-nlp/llm_dataset/only_node_datasets/llm_gen_graph_domain-related_100_20250202-125314.pt"
LLM_GRAPH_NON_DOMAIN = "/home/bruno/Documents/GitHub/social-media-nlp/llm_dataset/only_node_datasets/llm_gen_graph_no-domain-related_100_20250202-124046.pt"

# Prediction threshold
COSIN_SIM_THRESHOLD = 0.7
EDGE_PROB_THRESHOLD = 0.95

# Graph evaluation params
EDGE_PR_TRESH_LIST: list[float] = [
    0.5,
    0.6,
    0.7,
    0.8,
    0.9
]

def __get_avg_degree(graph: nx.Graph) -> float:
    return sum(dict(graph.degree()).values()) / graph.number_of_nodes()


def __load_datset(torch_grap_path: str) -> tuple[Data, dict[int, str]]:
    grap_dict = torch.load(torch_grap_path)
    data: Data = grap_dict["data"]
    nodes_idx_map: dict[int, str] = grap_dict["nodes_idx_map"]
    
    log.info(f"Loaded tensor graph from {torch_grap_path} with {data.num_nodes} nodes")
    return (data, nodes_idx_map)

@torch.no_grad()
def __get_a_auc(
    model: torch.nn.Module, 
    data: Data, 
    cos_sim_thresh: float = COSIN_SIM_THRESHOLD
) -> float:
    
    model.eval() 
    
    # Compute cosine similarity 
    normalized_features = torch.nn.functional.normalize(data.x, p=2, dim=1)  # L2 normalization
    cos_similarity_matrix = torch.matmul(normalized_features, normalized_features.transpose(0, 1))
    
    # Fake true adj matrix
    treshold_matrix = torch.zeros_like(cos_similarity_matrix)
    treshold_matrix[cos_similarity_matrix > cos_sim_thresh] = 1
    
    # Probabilistic ajd predicted matrix
    z = model.encode(data.x, data.edge_index)
    pred_adj_matrix: torch.Tensor = model.get_adj_prob_matrix(z)

    # Flatten matrices to get labels 
    pred_labels = pred_adj_matrix.flatten()
    fake_true_labels = treshold_matrix.flatten()

    auc = roc_auc_score(fake_true_labels.cpu().numpy(), pred_labels.cpu().numpy())

    return auc

def __build_graph(
    model: torch.nn.Module, 
    data: Data,
    nodes_idx: dict[int, str],
    edge_prob_tresh: float = EDGE_PROB_THRESHOLD
) -> nx.Graph:
    
    graph = nx.Graph()
    for node_label in nodes_idx.values():
        graph.add_node(node_label)

    model.eval() 
    z = model.encode(data.x, data.edge_index)
    prob_adj_matrix: torch.Tensor = model.get_adj_prob_matrix(z)
    
    row_index, col_index = torch.where(prob_adj_matrix > edge_prob_tresh)
    
    for i in range(row_index.size(0)):
        row = row_index[i].item()
        col = col_index[i].item()
        
        if graph.has_edge(nodes_idx[row], nodes_idx[col]):
            continue
        if graph.has_edge(nodes_idx[col], nodes_idx[row]):
            continue
        
        if row != col:
            graph.add_edge(nodes_idx[row], nodes_idx[col])
    
    log.info(f"Built graph with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges usign prob threshold {edge_prob_tresh}")
    return graph

def __plot_graph(graph: nx.Graph, title: str) -> None:
    plt.figure(figsize=(15, 15))
    nx.draw(graph, with_labels=True, node_size=30, font_size=10, font_color="red", node_color="skyblue", edge_color="gray")
    plt.title(title)
    plt.show()

def compute_llm_gramph(
    llm_graph_path: str, 
    domain_related_info: str, 
    cos_sim_thresh: float = COSIN_SIM_THRESHOLD, 
    prob_thresh: float = EDGE_PROB_THRESHOLD,
    show_graph: bool = False
) -> None:
    node_grap: tuple[Data, dict[int, str]] = __load_datset(llm_graph_path)
    data: Data = node_grap[0]
    nodes_idx: dict[int, str] = node_grap[1]
    
    # Compute A-AUC by comapring predicte labels to cosine similarity threshold reconstrcuted graph 
    model = torch.load(MODEL_PATH)
    a_auc = __get_a_auc(model=model, data=data)
    log.info(f"LLM {domain_related_info} graph, A-AUC({cos_sim_thresh})={a_auc}")
    
    # Evaute edge probability threshold graph
    pred_graph: nx.Graph = __build_graph(model=model, data=data, nodes_idx=nodes_idx, edge_prob_tresh=prob_thresh)
    avg_degree = __get_avg_degree(pred_graph)
    log.info(f"LLM Reconstrcuted {domain_related_info} graph (edge_prob_thres={prob_thresh:.4f}) of {pred_graph.number_of_nodes()} nodes, {pred_graph.number_of_edges()} edges, avg-degree {avg_degree:.4f} and connected {nx.is_connected(pred_graph)}")
    
    if show_graph:
        title = f"Predicted graph with edge prob {prob_thresh}" + \
                f"\nModel: {model.get_name()}" + \
                f"\n{domain_related_info}"
        __plot_graph(pred_graph, title=title)

def evaluate_domain_graph(cos_sim_thres: float = COSIN_SIM_THRESHOLD, prob_to_test: list[float] = EDGE_PR_TRESH_LIST):
    for edge_prob in prob_to_test:
        log.info(f"Evaluating LMM domain graph with edge prob {edge_prob:.4f}")
        compute_llm_gramph(
            llm_graph_path=LLM_GRAPH_DOMAIN, 
            domain_related_info="domain-related", 
            cos_sim_thresh=cos_sim_thres, 
            prob_thresh=edge_prob,
            show_graph=False
        )

def evaluate_non_domain_graph(cos_sim_thres: float = COSIN_SIM_THRESHOLD, prob_to_test: list[float] = EDGE_PR_TRESH_LIST):
    for edge_prob in prob_to_test:
        log.info(f"Evaluating LMM non-domain graph with edge prob {edge_prob:.4f}")
        compute_llm_gramph(
            llm_graph_path=LLM_GRAPH_NON_DOMAIN, 
            domain_related_info="non-domain-related", 
            cos_sim_thresh=cos_sim_thres, 
            prob_thresh=edge_prob,
            show_graph=False
        )

if __name__ == "__main__":
    # evaluate_domain_graph()
    # evaluate_non_domain_graph()
    compute_llm_gramph(
        llm_graph_path=LLM_GRAPH_DOMAIN, 
        domain_related_info="domain-related", 
        cos_sim_thresh=COSIN_SIM_THRESHOLD, 
        prob_thresh=EDGE_PROB_THRESHOLD,
        show_graph=True
    )
    compute_llm_gramph(
        llm_graph_path=LLM_GRAPH_NON_DOMAIN, 
        domain_related_info="non-domain-related", 
        cos_sim_thresh=COSIN_SIM_THRESHOLD, 
        prob_thresh=EDGE_PROB_THRESHOLD,
        show_graph=True
    )