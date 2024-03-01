# Template of the model based on https://medium.com/@pytorch_geometric/link-prediction-on-heterogeneous-graphs-with-pyg-6d5c29677c70

import torch
from torch_geometric.data import Data
import torch.nn.functional as F
import torch_geometric.transforms as T
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from torch_geometric.nn import SAGEConv, GCNConv, GatedGraphConv
from prepare_data import prepare_data_from_file, prepare_graph_from_file, prepare_graph_from_multiple_files

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
    print(f'    TP={tp}, FP={fp}, TN={tn}, FN={fn}')
    print(f'    Sensitivity = TP / (TP + FN) = {tp / (tp + fn)}')
    print(f'    Specificity = TN / (TN + FP) = {tn / (tn + fp)}')
    print(f'    Precision = TP / (TP + FP) = {tp / (tp + fp)}')
    print(f'    Negative predictive value = TN / (TN + FN) = {tn / (tn + fn)}')
    print(f'    Accuracy = (TP + TN) / (TP + TN + FP + FN) = {(tp + tn) / (tp + tn + fp + fn)}')
    print(f'    F1 score = 2TP / (2TP + FP + FN) = {2*tp / (2*tp + fp + fn)}')
    print("####")

def plot_loss(loss_function):
    epochs = np.linspace(1, NUM_OF_EPOCHS, NUM_OF_EPOCHS)

    plt.title('Loss function')
    plt.plot(epochs, loss_function)
    plt.xlabel('No. of epoch')
    plt.ylabel('Loss')
    plt.show()

def train_model(model, optimizer, data_train):
    model.train()
    
    print(f'Training model for {NUM_OF_EPOCHS} epochs')

    loss_function = list()
    pos_weight = torch.tensor([34])
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # total_loss = total_examples = 0
    for epoch in tqdm(range(NUM_OF_EPOCHS)):
        # print(epoch + 1)
        optimizer.zero_grad()
        out = model(data_train)

        loss = criterion(out, data_train.edge_label)
        loss.backward()

        loss_function.append(float(loss))

        optimizer.step()        
        # total_loss += float(loss) * out.numel(exit
        # total_examples += out.numel()
    
        # print(f"Epoch: {epoch:03d}, Loss: {total_loss / total_examples:.4f}")

        # calculate metrics after final epoch
    
    calculate_metrics_for_model(model, data_train, f'Training results after {NUM_OF_EPOCHS} epochs')
    plot_loss(loss_function)

def test_model(model, data_test):
    with torch.no_grad():
        calculate_metrics_for_model(model, data_test, f'Testing results')


class SAGEConvModel(torch.nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = SAGEConv(4, channels)
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
        self.conv1 = GCNConv(4, channels)
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

    def forward(self, x, edge_index):
        x = self.gru1(x, edge_index)
        x = F.relu(x)
        # x = self.gru2(x, edge_index)
        # x = F.relu(x)
        # x = self.gru3(x, edge_index)

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
    # data_train, data_test, data_val = prepare_graph_from_file(hits_file_path='..\odd_output\event000000000-hits.csv',
      #                                                        tracks_file_path='..\odd_output\event000000000-tracks_ambi.csv')

    # connecting events
    data_train, data_test, data_val = prepare_graph_from_multiple_files(path='data', number_of_files=1)

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # print(device)

    # model = Model(GRUModel(channels=128))#.to(device)
    model = Model(SAGEConvModel(channels=256))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    train_model(model, optimizer, data_train)
    test_model(model, data_test)

    # print(F.binary_cross_entropy_with_logits(pred, data_test.edge_label))
