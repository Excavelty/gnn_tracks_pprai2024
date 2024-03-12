import torch
import seaborn as sns
from matplotlib import pyplot as plt
from run_model import GCNConvModel, GRUModel, SAGEConvModel, Model, calculate_metrics_for_model
from prepare_data import prepare_graph_from_event, split_data
from collections import namedtuple

def plot_confusion_matrix(confusion_matrix, name, output_file):
    # normalize confusion matrix
    confusion_matrix = confusion_matrix / confusion_matrix.sum(axis=1, keepdims=True)

    # Create heatmap using seaborn
    sns.heatmap(confusion_matrix, annot=True, fmt='.2f', cmap='Reds', xticklabels=['Exists', 'Does not exist'], yticklabels=['Exists', 'Does not exist'])

    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion matrix for edge prediction ({name})')
    
    plt.savefig(f'output/{output_file}')

    plt.close()

if __name__ == '__main__':
    
    model_gcn = Model(GCNConvModel(channels=256))
    model_gcn.load_state_dict(torch.load('output/gcn_model.pth'))

    model_sage = Model(SAGEConvModel(channels=256))
    model_sage.load_state_dict(torch.load('output/sage_model.pth'))

    model_gru = Model(GRUModel(channels=64))
    model_gru.load_state_dict(torch.load('output/gru_model.pth'))

    ModelConfig = namedtuple('ModelConfig', ['model', 'name', 'output'])
    
    model_configs = [
    ModelConfig(model=model_gcn, name='GCN model', output='gcn_confusion_matrix.png'),
    ModelConfig(model=model_sage, name='SAGE model', output='sage_confusion_matrix.png'),
    ModelConfig(model=model_gru, name='GatedGraphConv model', output='gru_confusion_matrix.png')]

    data = prepare_graph_from_event(hits_file_path='./data/event000000003-hits.csv', tracks_file_path='./data/event000000003-tracks_ambi.csv')
    train_data, val_data, test_data = split_data(data)

    for config in model_configs:
        results = calculate_metrics_for_model(config.model, train_data, 'Testing on different event')
        plot_confusion_matrix(results, config.name, config.output)
