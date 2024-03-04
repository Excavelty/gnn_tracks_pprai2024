# Template of the model based on https://medium.com/@pytorch_geometric/link-prediction-on-heterogeneous-graphs-with-pyg-6d5c29677c70

import torch
from torch_geometric.data import Data
import torch.nn.functional as F
import torch_geometric.transforms as T
import numpy as np
import matplotlib.pyplot as plt
import argparse
from tqdm.auto import tqdm
from torch_geometric.nn import SAGEConv, GCNConv, GatedGraphConv
from prepare_data import prepare_graph_from_multiple_files, prepare_graph_from_event, split_data

NUM_OF_EPOCHS = 100

def calculate_metrics_for_model(model, data, title):
    tp = 0
    tn = 0
    fp = 0
    fn = 0

    out = model(data)

    for i, el in enumerate(out):
        # assume that 0.5 is enough to treat as real connection
        if el < 0.5:
            el = 0
        else:
            el = 1

        if data.edge_label[i] == 1:
            if el == 1:
                tp += 1
            else:
                fp += 1

        if data.edge_label[i] == 0:
            if el == 0:
                tn += 1
            else:
                fn += 1

    print("####")
    print(title)
    print(f'    TP={tp}, TN={tn}, FP={fp}, FN={fn}')
    print(f'    Sensitivity = TP / (TP + FN) = {tp / (tp + fn)}')
    print(f'    Specificity = TN / (TN + FP) = {tn / (tn + fp)}')
    print(f'    Precision = TP / (TP + FP) = {tp / (tp + fp)}')
    print(f'    Negative predictive value = TN / (TN + FN) = {tn / (tn + fn)}')
    print(f'    F! score = {2*tp / (2*tp + fp + fn)}')
    print(f'    Accuracy = (TP + TN) / (TP + TN + FP + FN) = {(tp + tn) / (tp + tn + fp + fn)}')
    print("####")

def plot_loss(all_loss, epochs_num):
    epochs = np.linspace(1, epochs_num, epochs_num)

    plt.title('Loss function')
    plt.plot(epochs, all_loss)
    plt.xlabel('No. of epoch')
    plt.ylabel('Loss')
    plt.show()

def train_model(model, optimizer, data_train, data_val):
    print(f'Training model for {NUM_OF_EPOCHS} epochs')

    all_train_loss = list()
    all_val_loss = list()

    for epoch in tqdm(range(NUM_OF_EPOCHS)):
        # Training step
        model.train()    
        
        optimizer.zero_grad()
        out = model(data_train)

        loss = F.binary_cross_entropy_with_logits(out, data_train.edge_label)
        loss.backward()

        all_train_loss.append(loss.item())

        optimizer.step()

        # Validation step
        model.eval()
        
        with torch.no_grad():
            out = model(data_val)

            val_loss = F.binary_cross_entropy_with_logits(out, data_val.edge_label)
            all_val_loss.append(val_loss)

            # early stopping, if val_loss is not better than any of the latest losses (up to 15 latest losses)
            if all(val_loss >= previous_loss for previous_loss in all_val_loss[-6:-1]) and len(all_val_loss) > 5:
                print(f'Early stopping in epoch {epoch}')
                break

    calculate_metrics_for_model(model, data_train, f'Training results after {NUM_OF_EPOCHS} epochs')
    plot_loss(all_train_loss, epoch + 1)

    # save latest model
    torch.save(model.state_dict(), 'latest_model.pth')

def test_model(model, data_test):
    with torch.no_grad():
        calculate_metrics_for_model(model, data_test, f'Testing results')


class SAGEConvModel(torch.nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = SAGEConv(3, channels)
        self.conv2 = SAGEConv(channels, channels)
        self.conv3 = SAGEConv(channels, channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.conv3(x, edge_index)
        x = F.relu(x)

        return x

class GCNConvModel(torch.nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = GCNConv(3, channels)
        self.conv2 = GCNConv(channels, channels)
        self.conv3 = GCNConv(channels, channels) 

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.conv3(x, edge_index)
        x = F.relu(x)

        return x

class GRUModel(torch.nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.gru1 = GatedGraphConv(channels, 10)
        self.gru2 = GatedGraphConv(channels, 10)
        self.gru3 = GatedGraphConv(channels, 10)

    def forward(self, x, edge_index):
        x = self.gru1(x, edge_index)
        x = F.relu(x)
        x = self.gru2(x, edge_index)
        x = F.relu(x)
        x = self.gru3(x, edge_index)

        return x 

class Classifier(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, edge_label_index):
        from_edges = x[edge_label_index[0]] 
        to_edges = x[edge_label_index[1]]
        
        return (from_edges * to_edges).sum(dim=-1)

class Model(torch.nn.Module):
    def __init__(self, gnn_model):
        super().__init__()

        self.GNN = gnn_model
        self.Classifier = Classifier()

    def forward(self, data):
        x = self.GNN(data.x, data.edge_index)
        result = self.Classifier(x, data.edge_label_index)

        return result

if __name__ == '__main__':
    # training based on single event
    # data = prepare_graph_from_event(hits_file_path='data/event000000000-hits.csv',
      #                                                        tracks_file_path='data/event000000000-tracks_ambi.csv')


    # data_train, data_test, data_val = split_data(data)

    # connecting events
    data_train, data_test, data_val = prepare_graph_from_multiple_files(path='data', number_of_files=1)

    parser = argparse.ArgumentParser()

    parser.add_argument('--model_type', type=str, default='gcn', help='Type of model to run, available: gcn [default], gru, sage')

    args = parser.parse_args()
    model_type = args.model_type

    if model_type == 'gru':
        model = Model(GRUModel(channels=128))
    elif model_type == 'sage':
        model = Model(SAGEConvModel(channels=256))
    else:
        model = Model(GCNConvModel(channels=256))

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    train_model(model, optimizer, data_train, data_val)
    test_model(model, data_test)
