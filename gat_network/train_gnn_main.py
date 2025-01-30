import time
import torch
import torch.version
from torch.utils.data import DataLoader
from torch_geometric.utils import negative_sampling
from torch_geometric.data import Data
from tqdm import tqdm
from gnn_model.gat_net_module import GatModule
from sklearn.metrics import roc_auc_score
import sys
import logging
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from custom_logger.logger_config import get_logger
from dataset_builder_wiki.wiki_torch_loader import WikiDataset

log: logging.Logger = get_logger(name=__name__)

# Use CPU to PyTorch Geometric incompatibility
device = torch.device('cpu')
log.info(f"Using torch device: {device}")

# In/out folder configuration
MODEL_DUMP_PATH = "/home/bruno/Documents/GitHub/social-media-nlp/gat_network/model_dumps"
TENSOR_FOLDER_PATH = "/home/bruno/Documents/GitHub/social-media-nlp/dataset_builder_wiki/final_dataset/tensor"

# Traininer parameters
NUM_EPOCH = 20
INITIAL_LR = 0.001
SAVE_MODEL_EPOCH_INTERVAL = 5

# Model parameters
INPUT_SIZE = 384 # Embedding model output size
HIDDEN_SIZE = 1024
OUTPUT_SIZE = 256
NUM_ATTENTION_LAYER = 1

@torch.no_grad()
def __get_auc(model: GatModule, data: Data, training: bool = False):
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
    # log.info(f"Predicted edges prob {y_pred}")
    
    # Edges true prob
    y_true = torch.cat([torch.ones(data.num_edges), torch.zeros(data.num_edges)])
    # log.info(f"True edges prob {y_true}")
    
    auc = roc_auc_score(y_true.cpu().numpy(), y_pred.cpu().numpy())
    
    if training:
        model.train()
    
    return auc

def __get_num_params(model: torch.nn.Module):
    return sum(p.numel() for p in model.parameters())

def train_and_dump(
    model_dump_path: str = MODEL_DUMP_PATH,
    tensor_folder_path: str = TENSOR_FOLDER_PATH,
    num_epoch: int = NUM_EPOCH,
    initial_lr: float = INITIAL_LR,
    input_size: int = INPUT_SIZE,
    hidden_size: int = HIDDEN_SIZE,
    output_size: int = OUTPUT_SIZE,
    num_attention_layer: int = NUM_ATTENTION_LAYER,
    save_model_epoch_interval: int = SAVE_MODEL_EPOCH_INTERVAL,
) -> None:
    
    if save_model_epoch_interval <= 0:
        raise ValueError(f"Save model epoch interval must be greater than 0, got {save_model_epoch_interval}")
    
    if save_model_epoch_interval > num_epoch:
        raise ValueError(f"Save model epoch interval must be less than or equal to number of epochs, got {save_model_epoch_interval}")
    
    dataset: WikiDataset = WikiDataset(tensor_folder=tensor_folder_path, shuffle=True)
    
    model: GatModule = GatModule(
        in_channels=input_size,
        hidden_channels=hidden_size,
        out_channels=output_size,
        num_attention_layer=num_attention_layer,
    ).to(device)
    
    num_model_params = __get_num_params(model)
    log.info(f"Built GNN model of {num_model_params} parameters with " + 
            f"{num_attention_layer} attention layers, input size {input_size}, " + 
            f"hidden size {hidden_size}, output size {output_size}, on device {device}")
    
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
    
        for idx, obj in tqdm(enumerate(dataset), desc=f"Epoch {epoch + 1}, data loop"):
            data: Data = obj
            data = data.to(device)
            
            optimizer.zero_grad()
            z = model.encode(data.x, data.edge_index)
            
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
            
            loss.backward()
            optimizer.step()
            
            log.info(f"Epoch: {epoch + 1}/{num_epoch}, Data index: {idx}, Loss (single item): {loss.item():.4f}, AUC (single item): {auc:.4f}, Time: {time.time() - start_time:.2f}s")
        
        epoch_loss = sum(loss_list) / len(loss_list)
        epoch_auc = sum(auc_list) / len(auc_list)
        log.info(f"Epoch: {epoch + 1}/{num_epoch}, Loss (epoch): {epoch_loss:.4f}, AUC (epoch): {epoch_auc:.4f}, Time: {time.time() - start_time:.2f}s")
        
        if (epoch + 1) % save_model_epoch_interval == 0:
            torch.save(model, os.path.join(model_dump_path, f"gnn_model_{num_model_params}_params_{input_size}_{hidden_size}_{output_size}_epoch_{epoch + 1}_{time.strftime('%Y%m%d-%H%M%S')}.pth"))
            log.info(f"Model salved on epoch {epoch + 1}")
    
    log.info(f"Training finished after {num_epoch} epochs and {time.time() - start_time:.2f}s, saving final model")
    torch.save(model, os.path.join(model_dump_path, f"gnn_model_{num_model_params}_params_{input_size}_{hidden_size}_{output_size}_final_{time.strftime('%Y%m%d-%H%M%S')}.pth"))
    log.info("Final model saved successfully")


# model = GatModule(
#     in_channels=INPUT_SIZE,
#     hidden_channels=HIDDEN_SIZE,
#     out_channels=OUTPUT_SIZE,
#     num_attention_layer=NUM_ATTENTION_LAYER,
# ).to(device)

# # BCEWithLogitsLoss(x, y) = - [y * log(sigmoid(x)) + (1 - y) * log(1 - sigmoid(x))]
# criterion = torch.nn.BCEWithLogitsLoss()
# optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01)


# def train(data: Data):
#     model.train()
#     optimizer.zero_grad()
#     z = model.encode(data.x, data.edge_index)

#     # Sample negative edges
#     neg_edge_index = negative_sampling(
#         edge_index=data.edge_index,
#         num_nodes=data.num_nodes,
#         num_neg_samples=data.num_edges,
#     )

#     # Decode and compute loss
#     logits = model.decode(z, data.edge_index, neg_edge_index)
#     labels = torch.cat([torch.ones(data.num_edges), torch.zeros(data.num_edges)])    
#     loss = criterion(logits, labels)
    
#     loss.backward()
#     optimizer.step()
#     return loss


# @torch.no_grad()
# def test(data: Data):
#     model.eval()    
#     z = model.encode(data.x, data.edge_index)
#     return model.decode_all(z)

# @torch.no_grad()
# def get_auc(data: Data):
#     model.eval()    
#     z = model.encode(data.x, data.edge_index)
    
#     neg_edge_index = negative_sampling(
#         edge_index=data.edge_index,
#         num_nodes=data.num_nodes,
#         num_neg_samples=data.num_edges,
#     )
    
#     # Edges predicted prob
#     y_pred = model.decode(z, data.edge_index, neg_edge_index).sigmoid()
    



# dataset: WikiDataset = WikiDataset(tensor_folder=TENSOR_FOLDER_PATH)



# # model = GatModule(
# #     in_channels=data.num_node_features,
# #     hidden_channels=64,
# #     out_channels=32,
# #     num_attention_layer=1,
# # ).to(device)



if __name__ == "__main__":
    train_and_dump()