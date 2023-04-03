import torch
import torch.nn.functional as F
from torch_geometric_temporal.nn.recurrent import GConvLSTM

class RecurrentGConvLSTM(torch.nn.Module):
    def __init__(self, node_features):
        super(RecurrentGConvLSTM, self).__init__()
        self.recurrent = GConvLSTM(node_features, 32, 1)
        self.linear = torch.nn.Linear(32, 1)

    def forward(self, x, edge_index, edge_weight, h, c):
        h_0, c_0 = self.recurrent(x, edge_index, edge_weight, h, c)
        h = F.relu(h_0)
        h = self.linear(h)
        return h, h_0, c_0