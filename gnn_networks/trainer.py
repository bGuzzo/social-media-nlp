import time
import torch
import torch_geometric
import torch.version
from torch_geometric.utils import negative_sampling
from torch_geometric.data import Data
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
import sys
import logging
import os
import numpy as np
from gnn_model_v1.gat_net_module_v1 import GatModelV1
from gnn_model_v2.gat_net_module_v2 import GatModelV2


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from custom_logger.logger_config import get_logger
from dataset_builder_wiki.wiki_torch_loader import WikiGraphDataset

log: logging.Logger = get_logger(name=__name__)

# Use CPU to PyTorch Geometric incompatibility
device = torch.device('cpu')
log.info(f"Using torch device: {device}")

# In/out folder configuration
MODEL_DUMP_PATH = "/home/bruno/Documents/GitHub/social-media-nlp/gnn_networks/model_dumps"
TENSOR_FOLDER_PATH = "/home/bruno/Documents/GitHub/social-media-nlp/dataset_builder_wiki/final_dataset/tensor"

# Traininer parameters
NUM_EPOCH = 20
INITIAL_LR = 0.001
SAVE_MODEL_EPOCH_INTERVAL = 5

# Model parameters
INPUT_SIZE = 384 # Embedding model output size
HIDDEN_SIZE = 128
OUTPUT_SIZE = 64
NUM_ATTENTION_LAYER = 3
NUM_ATTENTION_HEAD = 2
DROPOUT_PROB = 0.75

# Dataset split params
LEN_LIMIT = 100
TRAIN_PERC = 0.9

# Agnostic AUC params
COSIN_SIM_THRESHOLD = 0.7
NODE_SAMPLING_RATE = 0.1

# Optimizations
LOSS_NEG_EDGE_SAMLING_RATE = 0.5

# Edge agnostic measure
@torch.no_grad()
def __get_agnostic_auc(
    model: torch.nn.Module, 
    data: Data, 
    thresh: float = COSIN_SIM_THRESHOLD,
    training: bool = False
) -> float:
    model.eval() 
    
    # Udersample of input nodes features
    num_nodes = data.x.shape[0]
    num_sampled_nodes = int(num_nodes * NODE_SAMPLING_RATE)
    sampled_x_idx = torch.randperm(num_nodes)[:num_sampled_nodes]
    sampled_x = data.x[sampled_x_idx]
    
    # Extract node features form the model and udersample in same fashon 
    z = model.encode(data.x, data.edge_index)
    sampled_z = z[sampled_x_idx]
    sampled_prob_adj_matrix = model.get_adj_prob_matrix(sampled_z)
    
    # Compute cosine similarity 
    normalized_features = torch.nn.functional.normalize(sampled_x, p=2, dim=1)  # L2 normalization
    cos_similarity_matrix = torch.matmul(normalized_features, normalized_features.transpose(0, 1))
    
    treshold_matrix = torch.zeros_like(cos_similarity_matrix)
    treshold_matrix[cos_similarity_matrix > thresh] = 1
    
    fake_true_labels = treshold_matrix.flatten()
    pred_labels = sampled_prob_adj_matrix.flatten()

    auc = roc_auc_score(fake_true_labels.cpu().numpy(), pred_labels.cpu().numpy())
    
    if training:
        model.train()
    
    return auc


@torch.no_grad()
def __get_auc(model: torch.nn.Module, data: Data, training: bool = False) -> float:
    model.eval()    
    z = model.encode(data.x, data.edge_index)
    
    neg_edge_index = negative_sampling(
        edge_index=data.edge_index,
        num_nodes=data.num_nodes,
        num_neg_samples=data.num_edges,
    )
    
    # Edges predicted prob
    y_pred = model.decode(z, data.edge_index, neg_edge_index)
    y_pred = torch.sigmoid(y_pred)
    
    # Edges true prob
    y_true = torch.cat([torch.ones(data.num_edges), torch.zeros(data.num_edges)])
    
    auc = roc_auc_score(y_true.cpu().numpy(), y_pred.cpu().numpy())
    
    if training:
        model.train()
    
    return auc

def __train_and_dump(
    # Model params
    torch_model: torch.nn.Module,
    torch_train_set: torch_geometric.data.Dataset,
    num_epoch: int = NUM_EPOCH,
    # input_size: int = INPUT_SIZE,
    # hidden_size: int = HIDDEN_SIZE,
    # output_size: int = OUTPUT_SIZE,
    # num_attention_layer: int = NUM_ATTENTION_LAYER,
    # num_attention_head: int = NUM_ATTENTION_HEAD,
    # Trainer params
    
    initial_lr: float = INITIAL_LR,
    dump_model: bool = True,
    save_model_epoch_interval: int = SAVE_MODEL_EPOCH_INTERVAL,
    model_dump_path: str = MODEL_DUMP_PATH
) -> torch.nn.Module:
    
    if dump_model and save_model_epoch_interval <= 0:
        raise ValueError(f"Save model epoch interval must be greater than 0, got {save_model_epoch_interval}")
    
    if dump_model and save_model_epoch_interval > num_epoch:
        raise ValueError(f"Save model epoch interval must be less than or equal to number of epochs, got {save_model_epoch_interval}")
    
    model = torch_model.to(device)
    log.info(f"Training GNN model {model.get_name()}, on device {device}")
    
    # BCEWithLogitsLoss(x, y) = - [y * log(sigmoid(x)) + (1 - y) * log(1 - sigmoid(x))]
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=initial_lr)
    
    log.info(f"Start trainign for {num_epoch} epochs, using {initial_lr} learning rate, saving model every {save_model_epoch_interval} epochs")

    # Training loop
    model.train()
    start_time = time.time()
    for epoch in tqdm(range(num_epoch), desc="Epoch loop"):
        loss_list = []
        auc_list = []
        agnostic_auc_list = []

        # Shuffle dataset every epoch
        # torch_train_set.shuffle()
        for idx, obj in tqdm(enumerate(torch_train_set), total=len(torch_train_set), desc="Data loop"):
            data: Data = obj
            data = data.to(device)
            
            optimizer.zero_grad()
            z = model.encode(data.x, data.edge_index)
            
            # Use a under-sampled negative edges set for training efficiency
            num_neg_edges = int(data.num_edges * LOSS_NEG_EDGE_SAMLING_RATE)
            neg_edge_index = negative_sampling(
                edge_index=data.edge_index,
                num_nodes=data.num_nodes,
                num_neg_samples=data.num_edges,
            )
            
            logits = model.decode(z, data.edge_index, neg_edge_index)
            labels = torch.cat([torch.ones(data.num_edges), torch.zeros(data.num_edges)]).to(device)
            
            loss = criterion(logits, labels)
            loss_list.append(loss.item())
            
            auc = __get_auc(model, data, training=True)
            auc_list.append(auc)
            agn_auc = __get_agnostic_auc(model, data, training=True)
            agnostic_auc_list.append(agn_auc)
            
            loss.backward()
            optimizer.step()
            
            log.info(f"Epoch: {epoch + 1}/{num_epoch}, Data index: {idx}, Loss (single item): {loss.item():.4f}, AUC (single item): {auc:.4f},  Agnostic AUC (single item): {agn_auc:.4f}, Time: {time.time() - start_time:.2f}s")
        
        epoch_loss = sum(loss_list) / len(loss_list)
        epoch_auc = sum(auc_list) / len(auc_list)
        epoch_agn_auc = sum(agnostic_auc_list) / len(agnostic_auc_list)
        log.info(f"Epoch: {epoch + 1}/{num_epoch}, Loss (epoch): {epoch_loss:.4f}, AUC (epoch): {epoch_auc:.4f}, Agnostic AUC (epoch): {epoch_agn_auc:.4f}, Time: {time.time() - start_time:.2f}s")
        
        if dump_model and (epoch + 1) % save_model_epoch_interval == 0:
            torch.save(model, os.path.join(model_dump_path, f"{model.get_name()}_epoch_{epoch + 1}_{time.strftime('%Y%m%d-%H%M%S')}.pth"))
            log.info(f"Model salved on epoch {epoch + 1}")
    
    log.info(f"Training of model {model.get_name()} completed")
    log.info(f"Training finished after {num_epoch} epochs and {time.time() - start_time:.2f}s")
    
    if dump_model:
        torch.save(model, os.path.join(model_dump_path, f"{model.get_name()}_final_{time.strftime('%Y%m%d-%H%M%S')}.pth"))
    
    log.info("Final model saved successfully")
    
    # Prepare model for evaluation
    model.eval()
    return model


def __test_model(model: torch.nn.Module, torch_est_set: torch_geometric.data.Dataset) -> float:
    
    model.eval()
    log.info(f"Evaluating model {model.get_name()} with {len(torch_est_set)} samples")
    
    auc_list = []
    agnostic_auc_list = []
    for idx, obj in tqdm(enumerate(torch_est_set), total=len(torch_est_set), desc="Eval"):
        data: Data = obj
        data = data.to(device)
        
        agn_auc = __get_agnostic_auc(model, data, training=False)
        agnostic_auc_list.append(agn_auc)
        
        auc = __get_auc(model, data, training=False)
        auc_list.append(auc)
        log.info(f"Data index: {idx}, AUC (single item): {auc:.4f}, Agnostic AUC (single item): {agn_auc:.4f}")
    
    avg_test_auc = sum(auc_list) / len(auc_list)
    avg_test_agnostic_auc = sum(agnostic_auc_list) / len(agnostic_auc_list)
    
    log.info(f"Evaluation of model {model.get_name()} completed")
    log.info(f"Evaluation completed successfully on {len(torch_est_set)} samples, average AUC: {avg_test_auc:.4f}, average agnostic AUC: {avg_test_agnostic_auc:.4f}")
    return avg_test_auc

def __get_splitted_dataset(
    torch_tensor_path: str = TENSOR_FOLDER_PATH,
    len_limit: int = LEN_LIMIT,
    train_perc: float = TRAIN_PERC
) -> tuple[torch_geometric.data.Dataset, torch_geometric.data.Dataset]:
    
    log.info(f"Loading tensor data from {torch_tensor_path}")
    dataset = WikiGraphDataset(tensor_folder=torch_tensor_path, shuffle=True, size_limit=len_limit)
    train_set, test_set = dataset.split_dataset(train_percentage=train_perc, shuffle_before_split=False, shuffle_after_split=True)
    
    log.info(f"Splitted orignal dataset of {len(dataset)} into train set of {len(train_set)} and test set of {len(test_set)}")
    return train_set, test_set

def __train_and_test(
    torch_model: torch.nn.Module,
) -> float:
    
    train_set, test_set = __get_splitted_dataset()
    model = __train_and_dump(torch_model=torch_model, torch_train_set=train_set)
    test_auc = __test_model(model=model, torch_est_set=test_set)
    return test_auc

def __test_and_train_v1():
    model = GatModelV1(
        in_channels=INPUT_SIZE,
        hidden_channels=HIDDEN_SIZE,
        out_channels=OUTPUT_SIZE,
        num_attention_layer=NUM_ATTENTION_LAYER,
        num_attention_head=NUM_ATTENTION_HEAD,
        dropout_prob=DROPOUT_PROB,
    )
    __train_and_test(torch_model=model)

def __test_and_train_v2():
    model = GatModelV2(
        in_channels=INPUT_SIZE,
        hidden_channels=HIDDEN_SIZE,
        out_channels=OUTPUT_SIZE,
        num_attention_layer=NUM_ATTENTION_LAYER,
        num_attention_head=NUM_ATTENTION_HEAD,
        dropout_prob=DROPOUT_PROB,
    )
    __train_and_test(torch_model=model)

if __name__ == "__main__":
    # __test_and_train_v1()
    __test_and_train_v2()
