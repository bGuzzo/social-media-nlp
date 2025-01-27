

import time
import torch
import torch.version
from torch_geometric.utils import negative_sampling
from gat_network.test.data_loader_test import get_data
from gat_net_module import GatModule
from sklearn.metrics import roc_auc_score
import sys
import logging
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from custom_logger.logger_config import get_logger

log: logging.Logger = get_logger(name=__name__)

# Use CPU to PyTorch Geometric incompatibility
device = torch.device('cpu')
log.info(f"Using torch device: {device}")

data = get_data()

SAVE_PATH = "/home/bruno/Documents/GitHub/social-media-nlp/gat_network/model_dumps"

NUM_EPOCH = 100

model = GatModule(
    in_channels=data.num_node_features,
    hidden_channels=64,
    out_channels=32,
    num_attention_layer=1,
).to(device)


# BCEWithLogitsLoss(x, y) = - [y * log(sigmoid(x)) + (1 - y) * log(1 - sigmoid(x))]
criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01)


def train():
    model.train()
    optimizer.zero_grad()
    z = model.encode(data.x, data.edge_index)

    # Sample negative edges
    neg_edge_index = negative_sampling(
        edge_index=data.edge_index,
        num_nodes=data.num_nodes,
        num_neg_samples=data.num_edges,
    )

    # Decode and compute loss
    logits = model.decode(z, data.edge_index, neg_edge_index)
    labels = torch.cat([torch.ones(data.num_edges), torch.zeros(data.num_edges)])    
    loss = criterion(logits, labels)
    
    loss.backward()
    optimizer.step()
    return loss


@torch.no_grad()
def test():
    model.eval()    
    z = model.encode(data.x, data.edge_index)
    return model.decode_all(z)

@torch.no_grad()
def get_auc():
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
    return auc


for epoch in range(1, NUM_EPOCH + 1):
    loss = train()
    auc = get_auc()
    log.info(f"Epoch: {epoch:03d}, Loss: {loss:.4f}, AUC: {auc:.4f}")

torch.save(model, os.path.join(SAVE_PATH, f"gat_model_{time.strftime("%Y%m%d-%H%M%S")}.pt"))

# Get predicted links
predicted_links = test()
print(predicted_links)
