try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable):
        return iterable

import torch
import numpy as np
import matplotlib.pyplot as plt

import torch.nn.functional as F
from torch_geometric_temporal.nn.recurrent import EvolveGCNO, EvolveGCNH, GConvGRU, GConvLSTM

from torch_geometric_temporal.dataset import EnglandCovidDatasetLoader
from torch_geometric_temporal.signal import temporal_signal_split
from evolvegcnh import RecurrentEvolveGCNH
from evolvegcno import RecurrentEvolveGCNO
from gconvgru import RecurrentGConvGRU
from gconvlstm import RecurrentGConvLSTM
from utils import *

loader = EnglandCovidDatasetLoader()

mean_error_evolvegcno = []
mean_error_evolvegcnh = []
mean_error_gconvgru = []
mean_error_gconvlstm = []

dataset = loader.get_dataset(8)
train_dataset, test_dataset = temporal_signal_split(dataset, train_ratio=0.8)

lr = [0.1, 0.01, 0.001, 0.0001]

for lr in lr:
    print(f"Running for LR = {lr}")
    ################# EvolveGCNO
    print ("##################### Running EvolveGCNO #####################")
    model_evolvegcno = RecurrentEvolveGCNO(node_features = 8)
    for param in model_evolvegcno.parameters():
        param.retain_grad()

    optimizer = torch.optim.Adam(model_evolvegcno.parameters(), lr=lr)

    model_evolvegcno.train()

    for epoch in tqdm(range(200)):
        cost = 0
        for time, snapshot in enumerate(train_dataset):
            y_hat = model_evolvegcno(snapshot.x, snapshot.edge_index, snapshot.edge_attr)
            cost = cost + torch.mean((y_hat-snapshot.y)**2)
        cost = cost / (time+1)
        cost.backward(retain_graph=True)
        optimizer.step()
        optimizer.zero_grad()
        
    model_evolvegcno.eval()
    local_mean_error_evolvegcno = []
    for time, snapshot in enumerate(test_dataset):
        if time == 0:
            model_evolvegcno.recurrent.weight = None
        y_hat = model_evolvegcno(snapshot.x, snapshot.edge_index, snapshot.edge_attr)
        y_hat_flaten = torch.flatten(y_hat)
        #print(f'y Predicted {y_hat_flaten} Size {y_hat_flaten.size()}')
        #print(f'y {snapshot.y} Size {snapshot.y.size()}')
        local_mean_error_evolvegcno.append(torch.mean((y_hat_flaten-snapshot.y)**2).item())

    mean_error_evolvegcno.append(local_mean_error_evolvegcno)
    print(f'local_mean_error_evolvegcno {local_mean_error_evolvegcno}')


    ############################################# EvolveGCNH
    print ("##################### Running EvolveGCNH #####################")

            
    model_evolvegcnh = RecurrentEvolveGCNH(node_features = 8, node_count = 129)

    optimizer = torch.optim.Adam(model_evolvegcnh.parameters(), lr=lr)

    model_evolvegcnh.train()

    for epoch in tqdm(range(200)):
        cost = 0
        for time, snapshot in enumerate(train_dataset):
            y_hat = model_evolvegcnh(snapshot.x, snapshot.edge_index, snapshot.edge_attr)
            cost = cost + torch.mean((y_hat-snapshot.y)**2)
        cost = cost / (time+1)
        cost.backward()
        optimizer.step()
        optimizer.zero_grad()
        
    model_evolvegcnh.eval()
    local_mean_error_evolvegcnh = []
    for time, snapshot in enumerate(test_dataset):
        y_hat = model_evolvegcnh(snapshot.x, snapshot.edge_index, snapshot.edge_attr)
        y_hat_flaten = torch.flatten(y_hat)
        #print(f'y_hat_flattern {y_hat_flaten} Size {y_hat_flaten.size()}')
        local_mean_error_evolvegcnh.append(torch.mean((y_hat_flaten-snapshot.y)**2).item())

    mean_error_evolvegcnh.append(local_mean_error_evolvegcnh)
    print(f'local_mean_error_evolvegcnh {local_mean_error_evolvegcnh}')

    ##################### GConvGRU
    print ("##################### Running GConvGRU #####################")

            
    model_gconvgru = RecurrentGConvGRU(node_features = 8)

    optimizer = torch.optim.Adam(model_gconvgru.parameters(), lr=lr)

    model_gconvgru.train()

    for epoch in tqdm(range(200)):
        cost = 0
        for time, snapshot in enumerate(train_dataset):
            y_hat = model_gconvgru(snapshot.x, snapshot.edge_index, snapshot.edge_attr)
            cost = cost + torch.mean((y_hat-snapshot.y)**2)
        cost = cost / (time+1)
        cost.backward()
        optimizer.step()
        optimizer.zero_grad()
        
    model_gconvgru.eval()
    local_mean_error_gconvgru = []
    for time, snapshot in enumerate(test_dataset):
        y_hat = model_gconvgru(snapshot.x, snapshot.edge_index, snapshot.edge_attr)
        y_hat_flaten = torch.flatten(y_hat)
        local_mean_error_gconvgru.append(torch.mean((y_hat_flaten-snapshot.y)**2).item())

    mean_error_gconvgru.append(local_mean_error_gconvgru)
    print(f'local_mean_error_gconvgru {local_mean_error_gconvgru}')

    ##################### GConvLSTM
    print ("##################### Running GConvLSTM #####################")

            
    model_gconvlstm = RecurrentGConvLSTM(node_features=8)

    optimizer = torch.optim.Adam(model_gconvlstm.parameters(), lr=lr)

    model_gconvlstm.train()

    for epoch in tqdm(range(200)):
        cost = 0
        h, c = None, None
        for time, snapshot in enumerate(train_dataset):
            y_hat, h, c = model_gconvlstm(snapshot.x, snapshot.edge_index, snapshot.edge_attr, h, c)
            cost = cost + torch.mean((y_hat-snapshot.y)**2)
        cost = cost / (time+1)
        cost.backward()
        optimizer.step()
        optimizer.zero_grad()
        
    model_gconvlstm.eval()
    local_mean_error_gconvlstm = []
    cost = 0
    for time, snapshot in enumerate(test_dataset):
        y_hat, h, c = model_gconvlstm(snapshot.x, snapshot.edge_index, snapshot.edge_attr, h, c)
        y_hat_flaten = torch.flatten(y_hat)
        local_mean_error_gconvlstm.append(torch.mean((y_hat_flaten-snapshot.y)**2).item())

    mean_error_gconvlstm.append(local_mean_error_gconvlstm)
    print(f'local_mean_error_gconvlstm {mean_error_evolvegcno}')


print(f'mean_error_evolvegcno {mean_error_evolvegcno}')
print(f'mean_error_evolvegcnh {mean_error_evolvegcnh}')
print(f'mean_error_gconvgru {mean_error_gconvgru}')
print(f'mean_error_gconvlstm {mean_error_gconvlstm}')


plot_models_vs_lr(mean_error_evolvegcno,mean_error_evolvegcnh,mean_error_gconvgru,mean_error_gconvlstm)
