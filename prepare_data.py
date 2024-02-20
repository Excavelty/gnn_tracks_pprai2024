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

# function preparing input graph by taking one event (singe hits file and tracks_ambi file)
def prepare_graph_from_file(hits_file_path, tracks_file_path):
    # Read all hits from the file
    df = pd.read_csv(hits_file_path, engine='python', delimiter=',')
    
    # Prepare features of graph nodes, hit will be treated as a node
    x_s = df['tx'].values.tolist()
    y_s = df['ty'].values.tolist()
    z_s = df['tz'].values.tolist()

    # squash into features tensor and transpose
    node_features = torch.tensor([x_s, y_s, z_s], dtype=torch.float).t()

    # Read all tracks and use them to construct edge_index
    df = pd.read_csv(tracks_file_path, engine='python', delimiter=',')
    filter = df['good/duplicate/fake'] == 'good'
    
    tracks_strings = df.where(filter)['Hits_ID'].tolist()
    tracks_lists = [json.loads(str(track_string).replace(',]', ']')) for track_string in tracks_strings]

    edge_list = list()

    for track_list in tracks_lists:
        # create edges between every two consecutive elements of the particle track
        edge_list.extend([[track_list[i], track_list[i + 1]] for i in range(len(track_list) - 1)])

    edge_index = torch.tensor(edge_list, dtype=torch.long).t()

    print(edge_index.size())

    data = Data(node_features, edge_index)
    print(data)
    transform = T.Compose([T.NormalizeFeatures(), T.RandomLinkSplit(add_negative_train_samples=True, disjoint_train_ratio=0.3)])
    train_data, val_data, test_data = transform(data)

    print(train_data)
    print(test_data)
    print(val_data)

    return train_data, test_data, val_data

# function preparing data from single pair of hits/tracks_ambi files
def prepare_data_from_file(hits_file_path, tracks_file_path):
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
    print(tracks_strings[0:3])

    tracks_lists = [json.loads(str(track_string).replace(',]', ']')) for track_string in tracks_strings]

    edge_list = list()

    for track_list in tracks_lists:
        edge_list.extend([[track_list[i], track_list[i + 1]] for i in range(len(track_list) - 1)])

    return x_s, y_s, z_s, edge_list

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
            x_s, y_s, z_s, edge_list = prepare_data_from_file(hits_file, tracks_file)

            x_all.extend(x_s)
            y_all.extend(y_s)
            z_all.extend(z_s)
            edge_all.extend(edge_list)

            iter += 1

            print(f'{iter}. files prepared')

            hits_file = None
            tracks_file = None

        if iter >= number_of_files:
            break
    
    
    node_features = torch.tensor([x_all, y_all, z_all], dtype=torch.float).t()
    edge_index = torch.tensor(edge_all, dtype=torch.long).t()

    print(edge_index.size())

    data = Data(node_features, edge_index)
    print(data)
    transform = T.Compose([T.NormalizeFeatures(), T.RandomLinkSplit(add_negative_train_samples=True, disjoint_train_ratio=0.3)])
    train_data, val_data, test_data = transform(data)

    print(train_data)
    print(test_data)
    print(val_data)

    return train_data, test_data, val_data
