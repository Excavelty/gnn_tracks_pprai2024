import torch
from run_model import GCNConvModel, Model, test_model
from prepare_data import prepare_graph_from_event, split_data

if __name__ == '__main__':
    model = Model(GCNConvModel(channels=256))
    model.load_state_dict(torch.load('latest_model.pth'))
    
    data = prepare_graph_from_event(hits_file_path='./data/event000000001-hits.csv', tracks_file_path='./data/event000000001-tracks_ambi.csv')
    train_data, val_data, test_data = split_data(data)

    test_model(model, train_data)

    print(model)