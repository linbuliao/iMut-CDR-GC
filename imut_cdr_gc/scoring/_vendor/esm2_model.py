"""Model-only exact GCNNet extraction; no dataset or device side effects."""
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_max_pool as gmp
MAX_SEQ_LEN = 95

class GCNNet(nn.Module):
    def __init__(self, n_output: int = 1, n_filters: int = 64, num_features_xd: int = 20, output_dim: int = 1280, dropout: float = 0.2):
        super().__init__()
        self.conv1 = GCNConv(num_features_xd, num_features_xd)
        self.conv2 = GCNConv(num_features_xd, num_features_xd * 2)
        self.conv3 = GCNConv(num_features_xd * 2, num_features_xd * 4)
        self.conv4 = GCNConv(num_features_xd * 4, num_features_xd * 4)
        self.fc_g1 = nn.Linear(num_features_xd * 4, 1024)
        self.fc_g2 = nn.Linear(1024, output_dim)
        self.conv_xt_1 = nn.Conv1d(in_channels=20, out_channels=n_filters, kernel_size=8)
        self.fc1_xt = nn.Linear(64 * 88, output_dim)
        self.fc1 = nn.Linear(2 * output_dim, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.out = nn.Linear(512, n_output)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, data: Data) -> torch.Tensor:
        x, edge_index, batch = data.x, data.edge_index, data.batch
        target = data.target.reshape(-1, MAX_SEQ_LEN, 20).permute(0, 2, 1)

        x = self.relu(self.conv1(x, edge_index))
        x = self.relu(self.conv2(x, edge_index))
        x = self.relu(self.conv3(x, edge_index))
        x = self.relu(self.conv4(x, edge_index))
        x = gmp(x, batch)
        x = self.relu(self.fc_g1(x))
        x = self.dropout(self.fc_g2(x))

        xt = self.conv_xt_1(target)
        xt = self.fc1_xt(xt.view(-1, 64 * 88))

        xc = torch.cat((x, xt), dim=1)
        xc = self.relu(self.fc1(xc))
        xc = self.dropout(self.relu(self.fc2(xc)))
        return torch.sigmoid(self.out(xc))
