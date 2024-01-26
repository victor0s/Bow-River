# -*- coding: utf-8 -*-
"""
Created on Wed Sep 14 08:44:18 2022

@author: Paulo
"""

####################################################################################################################
# Libraries importing
####################################################################################################################

import torch
print('\nTorch version: ', torch.__version__)
print('CUDA available:', torch.cuda.is_available(), '\n') 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(2023)

import pandas as pd

import numpy as np

#import matplotlib.pyplot as plt

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils.convert import to_networkx

#import networkx as nx


####################################################################################################################
# Overall settings
####################################################################################################################

list_df_GNN_train = list_df_GNN_flow_train + list_df_GNN_precip_train + list_df_GNN_snowpillow_train + list_df_GNN_temperature_train
list_df_GNN_valid = list_df_GNN_flow_valid + list_df_GNN_precip_valid + list_df_GNN_snowpillow_valid + list_df_GNN_temperature_valid

NUMBER_OF_NEARBY_STATIONS = len(list_df_GNN_train) - 1

TIMESTEPS_TRAIN = len(list_df_GNN_train[0]) # one graph per timestep
TIMESTEPS_VALID = len(list_df_GNN_valid[0]) # one graph per timestep

TRAIN_BATCH_SIZE = 2048
VALID_BATCH_SIZE = 2048


####################################################################################################################
# Dataset and dataloader creation
####################################################################################################################

df_list_train_tmp = []
df_list_valid_tmp = []

for station in range(NUMBER_OF_NEARBY_STATIONS+1):          # iterate over nodes (stations)
    target_mean_X = np.mean(list_df_GNN_train[station].iloc[:, -1], axis=0)
    target_stdev_X = np.std(list_df_GNN_train[station].iloc[:, -1], axis=0)
    
    df_list_train_tmp.append(
        (list_df_GNN_train[station].iloc[:, 1:].copy(deep="all") - target_mean_X) / target_stdev_X
        )
    #print(((list_df_GNN_valid[station].iloc[:, 1:] - target_mean_X) / target_stdev_X).mean())
    #print(((list_df_GNN_valid[station].iloc[:, 1:] - target_mean_X) / target_stdev_X).std())
    df_list_valid_tmp.append(
        (list_df_GNN_valid[station].iloc[:, 1:].copy(deep="all") - target_mean_X) / target_stdev_X
        )
    #print(np.mean(df_list_valid_tmp[station].iloc[:, 1], axis=0), np.std(df_list_valid_tmp[station].iloc[:, 1], axis=0))

for station in range(NUMBER_OF_NEARBY_STATIONS+1):          # iterate over nodes (stations)
    if use_DoY_HoD:
        df_list_train_tmp[station].insert(0,'sinDoY',
                                          (np.sin(2*np.pi*pd.to_datetime(list_df_GNN_train[station].iloc[:, 0]).dt.dayofyear/365))) # * target_stdev_X + target_mean_X) # Normalization warning: de-normalizing to restore original data later
        df_list_train_tmp[station].insert(1, 'cosDoY',
                                          (np.cos(2*np.pi*pd.to_datetime(list_df_GNN_train[station].iloc[:, 0]).dt.dayofyear/365))) # * target_stdev_X + target_mean_X)
        df_list_train_tmp[station].insert(2,'sinHoD',
                                          (np.sin(2*np.pi*pd.to_datetime(list_df_GNN_train[station].iloc[:, 0]).dt.hour/24))) # * target_stdev_X + target_mean_X) # Normalization warning: de-normalizing to restore original data later
        df_list_train_tmp[station].insert(3, 'cosHoD',
                                          (np.cos(2*np.pi*pd.to_datetime(list_df_GNN_train[station].iloc[:, 0]).dt.hour/24))) # * target_stdev_X + target_mean_X)

        df_list_valid_tmp[station].insert(0,'sinDoY',
                                          (np.sin(2*np.pi*pd.to_datetime(list_df_GNN_valid[station].iloc[:, 0]).dt.dayofyear/365))) # * target_stdev_X + target_mean_X)
        df_list_valid_tmp[station].insert(1, 'cosDoY',
                                          (np.cos(2*np.pi*pd.to_datetime(list_df_GNN_valid[station].iloc[:, 0]).dt.dayofyear/365))) # * target_stdev_X + target_mean_X)
        df_list_valid_tmp[station].insert(2,'sinHoD',
                                          (np.sin(2*np.pi*pd.to_datetime(list_df_GNN_valid[station].iloc[:, 0]).dt.hour/24))) # * target_stdev_X + target_mean_X)
        df_list_valid_tmp[station].insert(3, 'cosHoD',
                                          (np.cos(2*np.pi*pd.to_datetime(list_df_GNN_valid[station].iloc[:, 0]).dt.hour/24))) # * target_stdev_X + target_mean_X)


# train dataloader

print('Creating train dataloader...')

data_list = [] # to skip the use of Dataset class

for timestep in range(TIMESTEPS_TRAIN): # iterate over the timesteps - one graph per timestep
    
    # Create edge connections
    
    #edge_index = torch.tensor([[0, 1, 1, 2],
    #                           [1, 0, 2, 1]], dtype=torch.long)
    #print('\nEdge indexes:')
    edge_index = torch.vstack((torch.arange(0,NUMBER_OF_NEARBY_STATIONS+1, dtype=torch.long),
                               torch.zeros(NUMBER_OF_NEARBY_STATIONS+1, dtype=torch.long)))
    #print(edge_index)
    
    # Give nodes (stations) information
    
    x = torch.zeros((NUMBER_OF_NEARBY_STATIONS+1, TIMELAGS*1 + 4*use_DoY_HoD), dtype=torch.float)
    for station in range(NUMBER_OF_NEARBY_STATIONS+1):          # iterate over nodes (stations)
        x[station] = torch.tensor(df_list_train_tmp[station].iloc[timestep, :-1], dtype=torch.float)
    #print('\nNodes information:')
    #print(x)
    
    # Buid output
    
    ys = torch.zeros(1, dtype=torch.float)
    ys = torch.tensor(df_list_train_tmp[0].iloc[timestep, -1], dtype=torch.float)
    #print('\nOutput:')
    #print(ys)
    #print(ys.shape)
    ys = ys.reshape(1,1)
    #print(ys)
    #print(ys.shape)
    
    # Build graph
    
    graph = Data(x=x, edge_index=edge_index, y=ys)
    data_list.append(graph)
    
    #print()
    #print(graph)
    #print()
    
    #for prop in graph:
    #    print(prop)
        
# Plot the graph

#vis = to_networkx(graph)

#node_labels = torch.rand((NUMBER_OF_NEARBY_STATIONS + 1))#.round().long()
#node_labels = graph.y.numpy()

#plt.figure(1,figsize=(40,30)) 
#nx.draw(vis, cmap=plt.get_cmap('Set3'), node_color = node_labels, node_size=7000, width=7, linewidths=7, arrowsize=70)
#plt.show()
#del vis, node_labels

train_loader = DataLoader(data_list, batch_size=TRAIN_BATCH_SIZE, shuffle=True)

print('\nTrain dataloader created!')

# validation dataloader

print('\nCreating validation dataloader...')

data_list = [] # to skip the use of Dataset class

for timestep in range(TIMESTEPS_VALID): # iterate over the timesteps - one graph per timestep
    
    # Create edge connections
    
    #edge_index = torch.tensor([[0, 1, 1, 2],
    #                           [1, 0, 2, 1]], dtype=torch.long)
    #print('\nEdge indexes:')
    edge_index = torch.vstack((torch.arange(0,NUMBER_OF_NEARBY_STATIONS+1, dtype=torch.long),
                               torch.zeros(NUMBER_OF_NEARBY_STATIONS+1, dtype=torch.long)))
    #print(edge_index)
    
    # Give nodes (stations) information
    
    x = torch.zeros((NUMBER_OF_NEARBY_STATIONS+1, TIMELAGS*1 + 4*use_DoY_HoD), dtype=torch.float)
    for station in range(NUMBER_OF_NEARBY_STATIONS+1):          # iterate over nodes (stations)
        x[station] = torch.tensor(df_list_valid_tmp[station].iloc[timestep, :-1], dtype=torch.float)
    
    #print('\nNodes information:')
    #print(x)
    
    # Buid output
    
    ys = torch.zeros(1, dtype=torch.float)
    ys = torch.tensor(df_list_valid_tmp[0].iloc[timestep, -1], dtype=torch.float)
    #print('\nOutput:')
    #print(ys)
    #print(ys.shape)
    ys = ys.reshape(1,1)
    #print(ys)
    #print(ys.shape)
 
    # Build graph
    
    graph = Data(x=x, edge_index=edge_index, y=ys)
    data_list.append(graph)
    
    #print()
    #print(graph)
    #print()
    
    #for prop in graph:
    #    print(prop)
        
valid_loader = DataLoader(data_list, batch_size=VALID_BATCH_SIZE, shuffle=False)

print('\nValidation dataloader created!')

del edge_index, x, ys, timestep, station, df_list_train_tmp, df_list_valid_tmp
