"""
Module containing the neural network definition and machine learning operations.
"""
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from copy import deepcopy
import numpy as np

# . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 

def f1_score_custom(y_true, y_pred, 
                    average_methods_list=['micro', 'macro', 'weighted', None]):
    """
    Calculate one or more types of F1 scores for multiclass/multilabel problems.

    Parameters:
    - y_true: array-like of shape (n_samples,)
        Ground truth (correct) target values.
    - y_pred: array-like of shape (n_samples,)
        Estimated targets as returned by a classifier.
    - average_methods_list: list of str, default=['micro', 'macro', 'weighted', None]
        A list of averaging methods to calculate F1 scores. 
        Supported values: ['micro', 'macro', 'weighted', None].

    Returns:
    - f1_scores: dict
        A dictionary where keys are averaging methods and values are the calculated F1 scores.
        For `average=None`, the value is a tuple of (F1 score list, label list, label counts).

    """
    # Validate average_methods_list
    valid_methods = ['micro', 'macro', 'weighted', None]
    if not all(method in valid_methods for method in average_methods_list):
        raise ValueError(f"Invalid averaging method(s) in {average_methods_list}. "
                         f"Supported methods are: {valid_methods}")
        
    from sklearn.utils.multiclass import unique_labels
    from sklearn.metrics import f1_score

    # Calculate F1 scores
    f1_scores = {}
    for method in average_methods_list:
        if method is None:
            labels = list(unique_labels(y_true, y_pred))     # Get unique labels
            scores = f1_score(y_true, y_pred, average=None, labels=labels)
            counts = [y_true.count(label) for label in labels]
            f1_scores[method] = (list(100 * scores), labels, counts)
            
        else:
            f1_scores[method] = 100 * f1_score(y_true, y_pred, average=method)

    return f1_scores

# . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 

class PyTorchDataset(Dataset):
    """
    Custom dataset class for handling spectral data in PyTorch.
    
    Instance Attributes:
    - imgs (torch.Tensor): 3D tensor of all spectra, shape: [# systems x t2 points, w1 points, w3 points].
    - img_labels (torch.Tensor): 1D tensor of classes corresponding to spectra, shape: [# systems x t2 points].
    - img_IDnums (torch.Tensor): 1D tensor of run ID numbers corresponding to spectra, shape: [# systems x t2 points].
    - img_t2s (torch.Tensor): 1D tensor of t2 timepoints corresponding to spectra, shape: [# systems x t2 points].
    - num_imgs (int): Total number of images in the dataset.
    - num_sys (int): Total number of systems reflected in the dataset.
    
    """
    def __init__(self, data, p, isTrain=True):
        system_ID_numbers = data['system ID numbers']
        sys_data = data['spectra']
        classes = data['classes']
        t2 = data['t2']
        
        # Shuffle data
        rng = np.random.default_rng(seed=p['split seed'])
        shuffle_indices = rng.permutation(len(system_ID_numbers)).astype(int)
        system_ID_numbers = system_ID_numbers[shuffle_indices]
        classes = classes[shuffle_indices]
        sys_data = sys_data[shuffle_indices]
        t2 = t2[shuffle_indices]
        
        split_point = int(p['train-test split'] * len(system_ID_numbers))
            
        # select appropriate subset of data for train/test
        if isTrain:
            system_ID_numbers = system_ID_numbers[0:split_point]
            classes = classes[0:split_point]
            t2 = t2[0:split_point,:]
            sys_data = sys_data[0:split_point,:,:,:]
        else:
            system_ID_numbers = system_ID_numbers[split_point:]
            classes = classes[split_point:]
            t2 = t2[split_point:,:]
            sys_data = sys_data[split_point:,:,:,:]
        
        # Calculate the shape of the final data array
        sys_data = sys_data.transpose(0, 3, 1, 2)
        num_sys, nt2s, nw, _ = sys_data.shape
        final_data_shape = (num_sys * nt2s, nw, nw)
        data_array = sys_data.reshape(final_data_shape)
        
        final_t2_shape = (num_sys * nt2s)
        t2_array = t2.reshape(final_t2_shape)
        
        if p['task'] == 'noise':
        
            # ~~~~~~~~ determine if any spectra need to be dropped (low SNR) ~~~~~~~~~~~
            full_mask = list()
            sub_mask = np.ones(len(system_ID_numbers), dtype=bool)
    
            for index, ID in enumerate(system_ID_numbers):
                if ID in data['dropped images']:
                    temp = data['dropped images'][ID]
                    full_mask.extend(temp)
                else:
                    full_mask.extend(deepcopy(sub_mask))
                
            if len(full_mask) != len(system_ID_numbers)*nt2s:
                raise Exception(f'ERROR: full mask has length {len(full_mask)} (expected {len(system_ID_numbers)*nt2s}). Now exiting.')

            else:
                full_mask = np.array(full_mask)
        
        # labels and ID nums scaled to match the number of t2points
        labels_array = np.kron(np.array(classes, dtype=int), np.ones(nt2s))
        IDnums_array = np.kron(np.array(system_ID_numbers, dtype=int), np.ones(nt2s))
        
        if p['task'] == 'noise_addition':
            # drop the spectra
            data_array = data_array[full_mask,:,:]
            t2_array = t2_array[full_mask]
            IDnums_array = IDnums_array[full_mask]
            labels_array = labels_array[full_mask]
        
        # takes into account any spectra that may have been dropped
        self.num_imgs = data_array.shape[0]
        self.num_sys = len(np.unique(IDnums_array))
        
        # float32
        self.imgs = torch.from_numpy(data_array).float()
        self.img_t2s = torch.from_numpy(t2_array).float()
        self.img_IDnums = torch.from_numpy(IDnums_array).long()
        self.img_labels = torch.from_numpy(labels_array).long()
                
    def __getitem__(self, index):
        return self.imgs[index], self.img_labels[index], self.img_IDnums[index], self.img_t2s[index]

    def __len__(self):
        return self.num_imgs

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