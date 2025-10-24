"""
This script orchestrates the training and evaluation of Graph Neural Network (GNN) models for link prediction.
It encompasses the entire pipeline, from data loading and preprocessing to model training, evaluation, and metric visualization.

The script is structured to perform the following key operations:

1.  **Configuration Management**:
    - Defines critical parameters for the training process, including the number of epochs, learning rate, and weight decay.
    - Specifies the architectural parameters of the GNN models, such as the number of hidden channels, attention layers, and heads.
    - Sets parameters for dataset splitting and for the computation of the custom Agnostic AUC (A-AUC) metric.

2.  **Data Handling**:
    - Loads a graph dataset, which is expected to be in a PyTorch Geometric format.
    - Splits the dataset into training and testing sets to ensure a robust evaluation of the model's generalization capabilities.

3.  **Core Training and Evaluation Logic**:
    - **`__train_and_dump`**: This function implements the main training loop. It iterates over the training data for a specified
      number of epochs, performs forward and backward passes, computes the loss, and updates the model's weights.
      It also periodically saves the model and logs various performance metrics.
    - **`__test_model`**: This function evaluates the performance of the trained model on the test set. It computes
      the standard link prediction AUC and the custom A-AUC metric to assess the model's performance from different perspectives.

4.  **Metric Computation and Visualization**:
    - **`__get_auc`**: Calculates the Area Under the ROC Curve (AUC) for the link prediction task, which measures the model's
      ability to discriminate between existing and non-existing edges.
    - **`__get_agnostic_auc`**: Computes a custom metric, termed Agnostic AUC (A-AUC), which evaluates the model's capacity
      to capture the semantic similarity between nodes, independent of the explicit graph structure.
    - The script generates and saves plots of the training loss, AUC, and A-AUC over time, providing visual insights into
      the training dynamics.

5.  **Model Execution**:
    - The main execution block initializes one or more GNN models, initiates the training process, and subsequently
      evaluates the trained models, thereby providing a comprehensive assessment of their performance.

This script is central to the project, providing the means to empirically validate the effectiveness of the proposed GNN
architectures on the task of link prediction in semantic graphs.
"""

import time
import torch
import torch_geometric
import torch.version
from torch_geometric.utils import negative_sampling
from torch_geometric.data import Data
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from plot_utils import plot_train_multiple_auc, plot_train_loss, plot_test_res
from metric_exporter import dump_test_metric, dump_train_metric
import sys
import logging
import os
from gnn_model_v1.gat_net_module_v1 import GatModelV1
from gnn_model_v2.gat_net_module_v2 import GatModelV2

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from custom_logger.logger_config import get_logger
from dataset_builder_wiki.wiki_torch_loader import WikiGraphDataset

log: logging.Logger = get_logger(name=__name__)

# Device configuration: Use CPU due to potential PyTorch Geometric incompatibility issues.
device = torch.device('cpu')
log.info(f"Using torch device: {device}")

# Configuration for input/output folders
MODEL_DUMP_PATH = "/home/bruno/Documents/GitHub/social-media-nlp/gnn_networks/model_dumps"
TENSOR_FOLDER_PATH = "/home/bruno/Documents/GitHub/social-media-nlp/dataset_builder_wiki/final_dataset/tensor"

# Training parameters
NUM_EPOCH = 2
INITIAL_LR = 0.001
WEIGHT_DECAY = 0.01
SAVE_MODEL_EPOCH_INTERVAL = 2
SAVE_MODEL = True

# Model architecture parameters
INPUT_SIZE = 384  # Corresponds to the output size of the sentence embedding model
HIDDEN_SIZE = 128
OUTPUT_SIZE = 64
NUM_ATTENTION_LAYER = 4
NUM_ATTENTION_HEAD = 2
DROPOUT_PROB = 0.75

# Dataset splitting parameters
LEN_LIMIT = 200  # Maximum size of the dataset to be used
TRAIN_PERC = 0.8  # Percentage of the dataset to be used for training

# Agnostic-AUC (A-AUC) parameters
COSIN_SIM_THRESHOLD = 0.7  # T_cos: Threshold for cosine similarity to define ground-truth in A-AUC
NODE_SAMPLING_RATE = 0.1  # R_cos: Sampling rate for nodes during A-AUC computation in training

# Optimization parameters
LOSS_NEG_EDGE_SAMLING_RATE = 0.3  # R_neg: Rate of negative edge sampling for loss computation

@torch.no_grad()
def __get_agnostic_auc(
    model: torch.nn.Module, 
    data: Data, 
    thresh: float = COSIN_SIM_THRESHOLD,
    sampling_rate: float = NODE_SAMPLING_RATE,
    training: bool = False
) -> float:
    """
    Computes the Agnostic Area Under the Curve (A-AUC), a custom metric that evaluates the model's ability
    to capture semantic similarity between nodes, independent of the graph's structure.

    Args:
        model (torch.nn.Module): The GNN model being evaluated.
        data (Data): The graph data.
        thresh (float, optional): The cosine similarity threshold for creating the ground-truth matrix. Defaults to COSIN_SIM_THRESHOLD.
        sampling_rate (float, optional): The rate at which to sample nodes for the computation. Defaults to NODE_SAMPLING_RATE.
        training (bool, optional): Flag to indicate if the model is in training mode. Defaults to False.

    Returns:
        float: The computed A-AUC score.
    """
    model.eval()
    
    # Undersample input node features
    num_nodes = data.x.shape[0]
    num_sampled_nodes = int(num_nodes * sampling_rate)
    sampled_x_idx = torch.randperm(num_nodes)[:num_sampled_nodes]
    sampled_x = data.x[sampled_x_idx]
    
    # Extract node features from the model and undersample in the same fashion
    z = model.encode(data.x, data.edge_index)
    sampled_z = z[sampled_x_idx]
    sampled_prob_adj_matrix = model.get_adj_prob_matrix(sampled_z)
    
    # Compute cosine similarity
    normalized_features = torch.nn.functional.normalize(sampled_x, p=2, dim=1)  # L2 normalization
    cos_similarity_matrix = torch.matmul(normalized_features, normalized_features.transpose(0, 1))
    
    # Compute ground-truth matrix based on the cosine similarity threshold
    treshold_matrix = torch.zeros_like(cos_similarity_matrix)
    treshold_matrix[cos_similarity_matrix > thresh] = 1
    
    # Flatten predicted and ground-truth adjacency matrices
    fake_true_labels = treshold_matrix.flatten()
    pred_labels = sampled_prob_adj_matrix.flatten()

    # Compute AUC
    auc = roc_auc_score(fake_true_labels.cpu().numpy(), pred_labels.cpu().numpy())
    
    if training:
        model.train()
    
    return auc

@torch.no_grad()
def __get_auc(model: torch.nn.Module, data: Data, training: bool = False) -> float:
    """
    Computes the standard Area Under the ROC Curve (AUC) for link prediction.

    Args:
        model (torch.nn.Module): The GNN model.
        data (Data): The graph data.
        training (bool, optional): Flag to indicate if the model is in training mode. Defaults to False.

    Returns:
        float: The computed AUC score.
    """
    model.eval()
    z = model.encode(data.x, data.edge_index)
    
    neg_edge_index = negative_sampling(
        edge_index=data.edge_index,
        num_nodes=data.num_nodes,
        num_neg_samples=data.num_edges,
    )
    
    # Get predicted probabilities for positive and negative edges
    y_pred = model.decode(z, data.edge_index, neg_edge_index)
    y_pred = torch.sigmoid(y_pred)
    
    # Create true labels
    y_true = torch.cat([torch.ones(data.num_edges), torch.zeros(data.num_edges)])
    
    auc = roc_auc_score(y_true.cpu().numpy(), y_pred.cpu().numpy())
    
    if training:
        model.train()
    
    return auc

def __train_and_dump(
    torch_model: torch.nn.Module,
    torch_train_set: torch_geometric.data.Dataset,
    num_epoch: int = NUM_EPOCH,
    initial_lr: float = INITIAL_LR,
    loss_neg_edge_sampl_rate: float = LOSS_NEG_EDGE_SAMLING_RATE,
    weight_decay: float = WEIGHT_DECAY,
    a_auc_cos_thresh: float = COSIN_SIM_THRESHOLD,
    dump_model: bool = SAVE_MODEL,
    save_model_epoch_interval: int = SAVE_MODEL_EPOCH_INTERVAL,
    model_dump_path: str = MODEL_DUMP_PATH
) -> torch.nn.Module:
    """
    Manages the training of the GNN model and saves the trained model to disk.

    Args:
        torch_model (torch.nn.Module): The GNN model to be trained.
        torch_train_set (torch_geometric.data.Dataset): The training dataset.
        num_epoch (int, optional): The number of training epochs. Defaults to NUM_EPOCH.
        initial_lr (float, optional): The initial learning rate. Defaults to INITIAL_LR.
        loss_neg_edge_sampl_rate (float, optional): The rate of negative edge sampling for loss computation. Defaults to LOSS_NEG_EDGE_SAMLING_RATE.
        weight_decay (float, optional): The weight decay for the optimizer. Defaults to WEIGHT_DECAY.
        a_auc_cos_thresh (float, optional): The cosine similarity threshold for A-AUC. Defaults to COSIN_SIM_THRESHOLD.
        dump_model (bool, optional): Whether to save the model. Defaults to SAVE_MODEL.
        save_model_epoch_interval (int, optional): The interval at which to save the model. Defaults to SAVE_MODEL_EPOCH_INTERVAL.
        model_dump_path (str, optional): The path to save the model. Defaults to MODEL_DUMP_PATH.

    Returns:
        torch.nn.Module: The trained GNN model.
    """
    
    if dump_model and save_model_epoch_interval <= 0:
        raise ValueError(f"Save model epoch interval must be greater than 0, got {save_model_epoch_interval}")
    
    if dump_model and save_model_epoch_interval > num_epoch:
        raise ValueError(f"Save model epoch interval must be less than or equal to number of epochs, got {save_model_epoch_interval}")
    
    model = torch_model.to(device)
    log.info(f"Training GNN model {model.get_name()}, on device {device}")
    
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=initial_lr, weight_decay=weight_decay)
    
    log.info(f"Start training for {num_epoch} epochs, using {initial_lr} learning rate, saving model every {save_model_epoch_interval} epochs")

    # Lists to store metrics across all data points and epochs
    auc_train_seps, a_auc_train_step, loss_train_step = [], [], []
    auc_avg_epochs_list, a_auc_avg_epochs_list, loss_avg_epochs_list = [], [], []
    
    model.train()
    start_time = time.time()
    for epoch in tqdm(range(num_epoch), desc="Epoch loop"):
        
        loss_list, auc_list, agnostic_auc_list = [], [], []

        for idx, obj in tqdm(enumerate(torch_train_set), total=len(torch_train_set), desc="Data loop"):
            data: Data = obj.to(device)
            
            optimizer.zero_grad()
            z = model.encode(data.x, data.edge_index)
            
            # Use an under-sampled set of negative edges for training efficiency
            num_neg_edges = int(data.num_edges * loss_neg_edge_sampl_rate)
            neg_edge_index = negative_sampling(
                edge_index=data.edge_index,
                num_nodes=data.num_nodes,
                num_neg_samples=num_neg_edges
            )
            
            logits = model.decode(z, data.edge_index, neg_edge_index)
            labels = torch.cat([torch.ones(data.num_edges), torch.zeros(num_neg_edges)]).to(device)
            
            loss = criterion(logits, labels)
            loss_list.append(loss.item())
            loss_train_step.append(loss.item())
            
            auc = __get_auc(model, data, training=True)
            auc_list.append(auc)
            auc_train_seps.append(auc)
            
            agn_auc = __get_agnostic_auc(model, data, thresh=a_auc_cos_thresh, training=True)
            agnostic_auc_list.append(agn_auc)
            a_auc_train_step.append(agn_auc)
            
            loss.backward()
            optimizer.step()
            
            log.info(f"Epoch: {epoch + 1}/{num_epoch}, Data index: {idx}, Loss: {loss.item():.4f}, AUC: {auc:.4f}, A-AUC: {agn_auc:.4f}, Time: {time.time() - start_time:.2f}s")
        
        epoch_loss = sum(loss_list) / len(loss_list)
        epoch_auc = sum(auc_list) / len(auc_list)
        epoch_agn_auc = sum(agnostic_auc_list) / len(agnostic_auc_list)
        
        auc_avg_epochs_list.append(epoch_auc)
        a_auc_avg_epochs_list.append(epoch_agn_auc)
        loss_avg_epochs_list.append(epoch_loss)
        
        log.info(f"Epoch: {epoch + 1}/{num_epoch}, Loss (epoch): {epoch_loss:.4f}, AUC (epoch): {epoch_auc:.4f}, A-AUC (epoch): {epoch_agn_auc:.4f}, Time: {time.time() - start_time:.2f}s")
        
        if dump_model and (epoch + 1) % save_model_epoch_interval == 0:
            torch.save(model, os.path.join(model_dump_path, f"{model.get_name()}_epoch_{epoch + 1}_{time.strftime('%Y%m%d-%H%M%S')}.pth"))
            log.info(f"Model saved on epoch {epoch + 1}")
    
    log.info(f"Training of model {model.get_name()} completed in {time.time() - start_time:.2f}s")
    
    dump_train_metric(model.get_name(), num_epoch, len(torch_train_set), len(torch_train_set)*num_epoch, auc_avg_epochs_list, a_auc_avg_epochs_list, loss_avg_epochs_list, a_auc_cos_thresh)
    plot_train_multiple_auc(model.get_name(), num_epoch, len(torch_train_set), auc_train_seps, a_auc_train_step, COSIN_SIM_THRESHOLD)
    plot_train_loss(model.get_name(), num_epoch, len(torch_train_set), loss_train_step)
    
    if dump_model:
        torch.save(model, os.path.join(model_dump_path, f"{model.get_name()}_{len(torch_train_set)}_final_{num_epoch}_{time.strftime('%Y%m%d-%H%M%S')}.pth"))
        log.info("Final model saved successfully")
    
    model.eval()
    return model

def __test_model(
    model: torch.nn.Module, 
    torch_test_set: torch_geometric.data.Dataset,
    train_epochs: int,
    train_dataset_size: int,
    train_num_steps: int,
    a_auc_cos_thresh: float = COSIN_SIM_THRESHOLD
) -> float:
    """
    Evaluates the trained GNN model on the test dataset.

    Args:
        model (torch.nn.Module): The trained GNN model.
        torch_test_set (torch_geometric.data.Dataset): The test dataset.
        train_epochs (int): The number of epochs used for training.
        train_dataset_size (int): The size of the training dataset.
        train_num_steps (int): The total number of training steps.
        a_auc_cos_thresh (float, optional): The cosine similarity threshold for A-AUC. Defaults to COSIN_SIM_THRESHOLD.

    Returns:
        float: The average AUC score over the test dataset.
    """
    
    model.eval()
    log.info(f"Evaluating model {model.get_name()} with {len(torch_test_set)} samples")
    
    auc_list, agnostic_auc_list = [], []
    for idx, obj in tqdm(enumerate(torch_test_set), total=len(torch_test_set), desc="Eval"):
        data: Data = obj.to(device)
        
        agn_auc = __get_agnostic_auc(model, data, thresh=a_auc_cos_thresh, training=False)
        agnostic_auc_list.append(agn_auc)
        
        auc = __get_auc(model, data, training=False)
        auc_list.append(auc)
        log.info(f"Data index: {idx}, AUC: {auc:.4f}, A-AUC: {agn_auc:.4f}")
    
    avg_test_auc = sum(auc_list) / len(auc_list)
    avg_test_agnostic_auc = sum(agnostic_auc_list) / len(agnostic_auc_list)
    
    dump_test_metric(model.get_name(), train_epochs, train_dataset_size, train_num_steps, len(torch_test_set), auc_list, agnostic_auc_list, a_auc_cos_thresh)
    plot_test_res(model.get_name(), train_num_steps, train_epochs, train_dataset_size, auc_list, agnostic_auc_list, COSIN_SIM_THRESHOLD)
    
    log.info(f"Evaluation of model {model.get_name()} completed. Average AUC: {avg_test_auc:.4f}, Average A-AUC: {avg_test_agnostic_auc:.4f}")
    return avg_test_auc

def __get_splitted_dataset(
    torch_tensor_path: str = TENSOR_FOLDER_PATH,
    len_limit: int = LEN_LIMIT,
    train_perc: float = TRAIN_PERC
) -> tuple[torch_geometric.data.Dataset, torch_geometric.data.Dataset]:
    """
    Loads the dataset from tensor files and splits it into training and testing sets.

    Args:
        torch_tensor_path (str, optional): The path to the folder containing the tensor files. Defaults to TENSOR_FOLDER_PATH.
        len_limit (int, optional): The maximum number of graphs to load. Defaults to LEN_LIMIT.
        train_perc (float, optional): The percentage of the dataset to use for training. Defaults to TRAIN_PERC.

    Returns:
        tuple[torch_geometric.data.Dataset, torch_geometric.data.Dataset]: A tuple containing the training and testing datasets.
    """
    
    log.info(f"Loading tensor data from {torch_tensor_path}")
    dataset = WikiGraphDataset(tensor_folder=torch_tensor_path, shuffle=True, size_limit=len_limit)
    train_set, test_set = dataset.split_dataset(train_percentage=train_perc, shuffle_before_split=False, shuffle_after_split=True)
    
    log.info(f"Split dataset of {len(dataset)} into train set of {len(train_set)} and test set of {len(test_set)}")
    return train_set, test_set

def __train_and_test(torch_model: torch.nn.Module) -> float:
    """
    A wrapper function that orchestrates the loading of the dataset, training of the model, and its final evaluation.

    Args:
        torch_model (torch.nn.Module): The GNN model to be trained and tested.

    Returns:
        float: The final test AUC score.
    """
    
    train_set, test_set = __get_splitted_dataset()
    model = __train_and_dump(torch_model=torch_model, torch_train_set=train_set)
    test_auc = __test_model(model, test_set, len(train_set), NUM_EPOCH, len(train_set)*NUM_EPOCH)
    return test_auc

def __test_and_train_v1():
    """Initializes and runs the training and testing for the GatModelV1 model."""
    model = GatModelV1(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, NUM_ATTENTION_LAYER, NUM_ATTENTION_HEAD, DROPOUT_PROB)
    __train_and_test(torch_model=model)

def __test_and_train_v2():
    """Initializes and runs the training and testing for the GatModelV2 model."""
    model = GatModelV2(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, NUM_ATTENTION_LAYER, NUM_ATTENTION_HEAD, DROPOUT_PROB)
    __train_and_test(torch_model=model)

if __name__ == "__main__":
    # This block serves as the main entry point of the script.
    # It sequentially trains and tests two versions of the GAT model.
    __test_and_train_v1()
    __test_and_train_v2()