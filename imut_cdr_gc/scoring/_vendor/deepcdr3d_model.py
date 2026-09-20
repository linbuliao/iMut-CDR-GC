"""Portable network with explicit antigen-first, antibody-second branch roles.

Parameter names and operations follow the retained project GCNNet source.
This is a CDR-focused dual-GCN adaptation, not a claim of a new GCN architecture.
Unlike the historical wrapper, graph roles cannot be inferred from positional
arguments. Forward requires named antigen and antibody graph batches.
"""
import torch
from torch import nn
from torch_geometric.nn import GCNConv, global_max_pool


class DeepCDR3DNet(nn.Module):
    def __init__(self, feature_dim=30, output_dim=1280, dropout=0.2):
        super().__init__()
        self.conv1 = GCNConv(feature_dim, feature_dim)
        self.conv2 = GCNConv(feature_dim, feature_dim * 2)
        self.conv3 = GCNConv(feature_dim * 2, feature_dim * 4)
        self.conv4 = GCNConv(feature_dim * 4, feature_dim * 4)
        self.fc_g1 = nn.Linear(feature_dim * 4, 1024)
        self.fc_g2 = nn.Linear(1024, output_dim)
        self.conv1_xt = GCNConv(feature_dim, feature_dim)
        self.conv2_xt = GCNConv(feature_dim, feature_dim * 2)
        self.conv3_xt = GCNConv(feature_dim * 2, feature_dim * 4)
        self.conv4_xt = GCNConv(feature_dim * 4, feature_dim * 4)
        self.fc_g1_xt = nn.Linear(feature_dim * 4, 1024)
        self.fc_g2_xt = nn.Linear(1024, output_dim)
        self.fc1 = nn.Linear(2 * output_dim, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.out = nn.Linear(512, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def _encode(self, graph, suffix):
        x = graph.x
        for n in range(1, 5):
            x = self.relu(getattr(self, f'conv{n}{suffix}')(x, graph.edge_index))
        x = global_max_pool(x, graph.batch)
        x = self.relu(getattr(self, f'fc_g1{suffix}')(x))
        return self.dropout(getattr(self, f'fc_g2{suffix}')(x))

    def forward(self, *, antigen, antibody):
        if antigen.num_graphs != antibody.num_graphs:
            raise ValueError('Antigen and antibody batch lengths differ')
        ag = self._encode(antigen, '')
        ab = self._encode(antibody, '_xt')
        x = self.relu(self.fc1(torch.cat((ag, ab), dim=1)))
        x = self.dropout(self.relu(self.fc2(x)))
        return torch.sigmoid(self.out(x))
