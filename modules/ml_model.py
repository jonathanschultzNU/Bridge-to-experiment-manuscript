"""
Module containing the neural network definition and machine learning operations.
"""
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np

# . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 

def f1_score_custom(y_true, y_pred, average_methods_list=['micro', 'macro', 'weighted', None]):
    """
    Calculate various F1 scores for classification evaluation.

    Parameters:
        y_true (array-like): Ground truth labels.
        y_pred (array-like): Predicted labels.
        average_methods_list (list): List of averaging methods for F1-score calculation.

    Returns:
        dict: Dictionary containing calculated F1 scores.
    """
    from sklearn.metrics import f1_score
    f1_scores = {}
    for method in average_methods_list:
        f1_scores[method] = 100 * f1_score(y_true, y_pred, average=method) if method else (list(f1_score(y_true, y_pred, average=None)), None)
    return f1_scores

# . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 

class PyTorchDataset(Dataset):
    """
    Custom dataset class for handling spectral data in PyTorch.
    """
    def __init__(self, data, p, isTrain=True):
        system_ID_numbers = data['system ID numbers']
        sys_data = data['spectra']
        classes = data['classes']
        
        # Shuffle data
        rng = np.random.default_rng(seed=p['split_seed'])
        shuffle_indices = rng.permutation(len(system_ID_numbers))
        system_ID_numbers = system_ID_numbers[shuffle_indices]
        classes = classes[shuffle_indices]
        sys_data = sys_data[shuffle_indices]
        
        split_point = int(p['train-test split'] * len(system_ID_numbers))
        if isTrain:
            sys_data = sys_data[:split_point]
        else:
            sys_data = sys_data[split_point:]
        
        self.imgs = torch.from_numpy(sys_data).float()
        self.img_labels = torch.from_numpy(classes).long()
    
    def __getitem__(self, index):
        return self.imgs[index], self.img_labels[index]

    def __len__(self):
        return len(self.imgs)

# . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 

class NeuralNet(nn.Module):
    """
    Simple feed-forward neural network for classification.
    """
    def __init__(self, inputSize, hiddenSize, numClasses, dropout_val):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(inputSize, hiddenSize)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hiddenSize, numClasses)
        self.dropout = nn.Dropout(dropout_val)
    
    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.l2(out)
        return out

# . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 