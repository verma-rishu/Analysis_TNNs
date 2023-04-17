import torch
import torch.nn.functional as F
from torch_geometric_temporal.nn.recurrent import EvolveGCNO


class RecurrentEvolveGCNO(torch.nn.Module):
    def __init__(self, node_features):
        super(RecurrentEvolveGCNO, self).__init__()
        self.recurrent = EvolveGCNO(node_features)
        self.linear = torch.nn.Linear(node_features, 1)

    def forward(self, x, edge_index, edge_weight):
        h = self.recurrent(x, edge_index, edge_weight)
        h = F.tanh(h)
        h = self.linear(h)
        return h
        