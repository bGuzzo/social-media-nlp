import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.utils import negative_sampling
from data_loader_w_embed import data


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels)
        self.conv2 = GATConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, out_channels)

    def encode(self, x, edge_index):
        # First message passing layer
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        # Second message passing layer
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        # Output layer
        x = self.lin(x)
        return x
    
        # x = self.conv1(x, edge_index)
        # x = x.relu()
        # return self.conv2(x, edge_index)

    def decode(self, z, pos_edge_index, neg_edge_index):
        edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=-1)
        logits = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=-1)
        return logits

    def decode_all(self, z):
        prob_adj = z @ z.t()
        return (prob_adj > 0).nonzero(as_tuple=False).t()

    def decode_prob(self, z):
        prob_adj = z @ z.t()
        # prob_adj = torch.sigmoid(prob_adj)
        print(torch.sigmoid(prob_adj))
        return (prob_adj > 0).nonzero(as_tuple=False).t()


model = GCN(
    data.num_node_features, 64, 32
)  # Example with hidden_channels=64 and out_channels=32
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


for epoch in range(1, 101):
    loss = train()
    print(f"Epoch: {epoch:03d}, Loss: {loss:.4f}")

# Get predicted links
predicted_links = test()
print(predicted_links)
