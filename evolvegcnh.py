import torch
import torch.nn.functional as F
from torch_geometric_temporal.nn.recurrent import EvolveGCNH

class RecurrentEvolveGCNH(torch.nn.Module):
    def __init__(self, node_count, node_features):
        super(RecurrentEvolveGCNH, self).__init__()
        self.recurrent = EvolveGCNH(node_count, node_features)
        self.linear = torch.nn.Linear(node_features, 1)

    def forward(self, x, edge_index, edge_weight):
        h = self.recurrent(x, edge_index, edge_weight)
        h = F.relu(h)
        h = self.linear(h)
        return h