import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import negative_sampling
from sklearn.metrics import roc_auc_score
from torch_geometric.data import Data


class GCN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        torch.manual_seed(1234567)
        self.conv1 = GCNConv(dataset.num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, dataset.num_classes)

    def forward(self, x, edge_index):
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


# Load your SDG graph data (replace with your actual data loading)
# Assuming you have `edge_index` and `x` defined
data = Data(x=x, edge_index=edge_index)

# Split edges into training and test sets
train_data, test_data = train_test_split_edges(data)

# Initialize the GNN model
model = GCN(hidden_channels=64)

# Define optimizer and loss function
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.BCEWithLogitsLoss()


# Train the GNN
def train():
    model.train()
    optimizer.zero_grad()
    z = model(train_data.x, train_data.edge_index)
    # Generate negative samples for link prediction
    neg_edge_index = negative_sampling(
        edge_index=train_data.edge_index,
        num_nodes=train_data.num_nodes,
        num_neg_samples=train_data.edge_label_index.size(1),
    )
    # Calculate loss based on positive and negative edges
    edge_label_index = torch.cat(
        [train_data.edge_label_index, neg_edge_index],
        dim=-1,
    )
    edge_label = torch.cat(
        [
            train_data.edge_label,
            train_data.edge_label.new_zeros(neg_edge_index.size(1)),
        ],
        dim=0,
    )
    loss = criterion(z[edge_label_index].view(-1), edge_label)
    loss.backward()
    optimizer.step()
    return loss


# Evaluate the GNN
def test(data):
    model.eval()
    z = model(data.x, data.edge_index)
    # Calculate AUC for link prediction
    auc = roc_auc_score(
        data.edge_label.cpu().numpy(),
        z[data.edge_label_index].view(-1).detach().cpu().numpy(),
    )
    return auc


# Training loop
for epoch in range(1, 101):
    loss = train()
    auc = test(test_data)
    print(f"Epoch: {epoch:03d}, Loss: {loss:.4f}, AUC: {auc:.4f}")
