import torch
import pandas as pd
import json
import os
from torch_geometric import seed_everything
from torch_geometric.data import Data
import torch_geometric.transforms as T

# initial seed for reproducibility
seed_everything(1000)

# Experiment concept:
# we take already reconstructed tracks and hits to create graph for training network
# to predict existance of the edge
# therefore we create "real" edges between elements belonging to the same track
# and RandomLinkSplit will label them as 1s and create same number of "false" edges
# and label them as 0s. The task is to check how well different architectures
# will learn to distinguish between those labels 

# function preparing data from single pair of hits/tracks_ambi files
def prepare_data_from_event(hits_file_path, tracks_file_path):
    # Read all hits from the file
    df = pd.read_csv(hits_file_path, engine='python', delimiter=',')
    
    # Prepare features of graph nodes, hit will be treated as a node
    x_s = df['tx'].values.tolist()
    y_s = df['ty'].values.tolist()
    z_s = df['tz'].values.tolist()

    # Read all tracks and use them to construct edge_index
    df = pd.read_csv(tracks_file_path, engine='python', delimiter=',')
    filter = df['good/duplicate/fake'] == 'good'
    
    tracks_strings = df.where(filter)['Hits_ID'].dropna().tolist()

    tracks_lists = [json.loads(str(track_string).replace(',]', ']')) for track_string in tracks_strings]

    edge_list = list()

    for track_list in tracks_lists:
        edge_list.extend([[track_list[i], track_list[i + 1]] for i in range(len(track_list) - 1)])

    data = dict(x_s=x_s, y_s=y_s, z_s=z_s, edge_list=edge_list)

    return data

def prepare_node_features_from_dict(data):
    node_features = torch.tensor([data['x_s'], data['y_s'], data['z_s']])
    return node_features

def prepare_edge_list_from_dict(data):
    edge_list = data['edge_list']
    return edge_list

def split_data(data):
    transform = T.Compose([T.NormalizeFeatures(), T.RandomLinkSplit(add_negative_train_samples=True, disjoint_train_ratio=0.3)])
    train_data, val_data, test_data = transform(data)

    return train_data, val_data, test_data

def prepare_graph_from_event(hits_file_path, tracks_file_path):
    data_dict = prepare_data_from_event(hits_file_path, tracks_file_path)
    node_features = prepare_node_features_from_dict(data_dict)
    edge_list = prepare_edge_list_from_dict(data_dict)

    data = Data(node_features, edge_list)

    return data

def prepare_training_data_from_event(hits_file_path, tracks_file_path):
    data = prepare_graph_from_event(hits_file_path, tracks_file_path)
    return split_data(data)

# CONTROVERSIAL!!!
# Function preparing graph as connection of graphs from different events
def prepare_graph_from_multiple_files(path, number_of_files):
    file_names = os.listdir(path)

    x_all = list()
    y_all = list()
    z_all = list()
    edge_all = list()

    hits_file = tracks_file = None

    iter = 0

    for file_name in file_names:
        if 'hits' in file_name:
            hits_file = path + '/' + file_name

        if 'tracks_ambi' in file_name:
            tracks_file = path + '/' + file_name

        if hits_file is not None and tracks_file is not None:
            data_dict = prepare_data_from_event(hits_file, tracks_file)

            x_all.extend(data_dict['x_s'])
            y_all.extend(data_dict['y_s'])
            z_all.extend(data_dict['z_s'])
            edge_all.extend(data_dict['edge_list'])

            iter += 1

            print(f'{iter}. files prepared')

            hits_file = None
            tracks_file = None

        if iter >= number_of_files:
            break
    
    
    node_features = torch.tensor([x_all, y_all, z_all], dtype=torch.float).t()
    edge_index = torch.tensor(edge_all, dtype=torch.long).t()

    data = Data(node_features, edge_index)
    print(data)
    
    train_data, test_data, val_data = split_data(data)
    print(train_data)
    print(test_data)
    print(val_data)

    return train_data, test_data, val_data
