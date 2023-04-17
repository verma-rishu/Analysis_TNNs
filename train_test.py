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
from mpnnlstm import RecurrentMPNNLSTM
from utils import *
np.random.seed(42)

loader = EnglandCovidDatasetLoader()

mean_error_evolvegcno = []
mean_error_evolvegcnh = []
mean_error_gconvgru = []
mean_error_gconvlstm = []
mean_error_mpnnlstm = []

MSE_evolvegcno = []
MSE_evolvegcnh = []
MSE_gconvgru = []
MSE_gconvlstm = []
MSE_mpnnlstm = []

lr = [0.1, 0.01, 0.001, 0.0001]
nf = 8
node_count = 129
dataset = loader.get_dataset(nf)
train_dataset, test_dataset = temporal_signal_split(dataset, train_ratio=0.8)

# for lr in lr:
#     print(f"Running for LR = {lr}")

################# EvolveGCNO
print ("##################### Running EvolveGCNO #####################")
model_evolvegcno = RecurrentEvolveGCNO(node_features = nf)
for param in model_evolvegcno.parameters():
    param.retain_grad()

optimizer = torch.optim.RMSprop(model_evolvegcno.parameters(), lr=0.0001)#, momentum=0.9)

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
cost = 0
local_mean_error_evolvegcno = []
for time, snapshot in enumerate(test_dataset):
    if time == 0:
        model_evolvegcno.recurrent.weight = None
    y_hat = model_evolvegcno(snapshot.x, snapshot.edge_index, snapshot.edge_attr)
    y_hat_flaten = torch.flatten(y_hat)
    local_mean_error_evolvegcno.append(torch.mean((y_hat_flaten-snapshot.y)**2).item())
    cost = cost + torch.mean((y_hat-snapshot.y)**2)


mean_error_evolvegcno.append(local_mean_error_evolvegcno)
cost = cost / (time+1)
cost = cost.item()
MSE_evolvegcno.append(cost)
print("MSE_evolvegcno: {:.4f}".format(cost))

############################################# EvolveGCNH
print ("##################### Running EvolveGCNH #####################")

        
model_evolvegcnh = RecurrentEvolveGCNH(node_features = nf, node_count = 129)

optimizer = torch.optim.RMSprop(model_evolvegcnh.parameters(), lr=0.01)#, momentum=0.9)

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
    local_mean_error_evolvegcnh.append(torch.mean((y_hat_flaten-snapshot.y)**2).item())
    cost = cost + torch.mean((y_hat-snapshot.y)**2)

mean_error_evolvegcnh.append(local_mean_error_evolvegcnh)
cost = cost / (time+1)
cost = cost.item()
MSE_evolvegcnh.append(cost)
print("MSE_evolvegcnh: {:.4f}".format(cost))

##################### GConvGRU
print ("##################### Running GConvGRU #####################")
model_gconvgru = RecurrentGConvGRU(node_features = nf)

optimizer = torch.optim.RMSprop(model_gconvgru.parameters(), lr=0.0001)#, momentum=0.9)

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
cost = 0
local_mean_error_gconvgru = []
for time, snapshot in enumerate(test_dataset):
    y_hat = model_gconvgru(snapshot.x, snapshot.edge_index, snapshot.edge_attr)
    y_hat_flaten = torch.flatten(y_hat)
    cost = cost + torch.mean((y_hat-snapshot.y)**2)
    local_mean_error_gconvgru.append(torch.mean((y_hat_flaten-snapshot.y)**2).item())

mean_error_gconvgru.append(local_mean_error_gconvgru)
cost = cost / (time+1)
cost = cost.item()
MSE_gconvgru.append(cost)
print("MSE_gconvgru: {:.4f}".format(cost))

##################### GConvLSTM
print ("##################### Running GConvLSTM #####################")

        
model_gconvlstm = RecurrentGConvLSTM(node_features=nf)

optimizer = torch.optim.RMSprop(model_gconvlstm.parameters(), lr=0.1)#, momentum=0.9)

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
    cost = cost + torch.mean((y_hat-snapshot.y)**2)

mean_error_gconvlstm.append(local_mean_error_gconvlstm)
cost = cost / (time+1)
cost = cost.item()
MSE_gconvlstm.append(cost)
print("MSE_gconvlstm: {:.4f}".format(cost))

'''
##################### MPNNLSTM
print ("##################### Running MPNNLSTM #####################")

model = RecurrentMPNNLSTM(8)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

model.train()

for epoch in tqdm(range(50)):
    cost = 0
    for time, snapshot in enumerate(train_dataset):
        y_hat = model(snapshot.x, snapshot.edge_index, snapshot.edge_attr)
        cost = cost + torch.mean((y_hat-snapshot.y)**2)
    cost = cost / (time+1)
    cost.backward()
    optimizer.step()
    optimizer.zero_grad()
    
model.eval()
local_mean_error_mpnnlstm = []
cost = 0
for time, snapshot in enumerate(test_dataset):
    y_hat = model(snapshot.x, snapshot.edge_index, snapshot.edge_attr)
    cost = cost + torch.mean((y_hat-snapshot.y)**2)
    local_mean_error_mpnnlstm.append(torch.mean((y_hat-snapshot.y)**2).item())

mean_error_mpnnlstm.append(local_mean_error_mpnnlstm)
cost = cost / (time+1)
cost = cost.item()
MSE_mpnnlstm.append(cost)
print("MSE_mpnnlstm: {:.4f}".format(cost))
    '''
print(f'mean_error_evolvegcno {mean_error_evolvegcno} ;  MSE_evolvegcno {MSE_evolvegcno}')
print(f'mean_error_evolvegcnh {mean_error_evolvegcnh} ;  MSE_evolvegcnh {MSE_evolvegcnh}')
print(f'mean_error_gconvgru {mean_error_gconvgru} ;  MSE_gconvgru  {MSE_gconvgru}')
print(f'mean_error_gconvlstm {mean_error_gconvlstm} ; MSE_gconvlstm {MSE_gconvlstm}')
#print(f'mean_error_mpnnlstm {mean_error_mpnnlstm} ; MSE_mpnnlstm {MSE_mpnnlstm}')
