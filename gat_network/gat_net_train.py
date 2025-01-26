

import torch
from torch_geometric.utils import negative_sampling
from data_loader_test import data
from gat_net_module import GatModule


NUM_EPOCH = 200


model = GatModule(
    in_channels=data.num_node_features
)  



optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01)
criterion = torch.nn.BCEWithLogitsLoss()


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
    return model.decode_prob(z)


for epoch in range(1, NUM_EPOCH + 1):
    loss = train()
    print(f"Epoch: {epoch:03d}, Loss: {loss:.4f}")

# Get predicted links
predicted_links = test()
print(predicted_links)
