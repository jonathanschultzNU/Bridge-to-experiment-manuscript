"""
Module to handle dataset loading, transformations, and classification.
"""

import numpy as np
import pandas as pd
import pickle

# . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 

def load_central_dataset(p: dict, dataset_path: str) -> dict:
    """
    Load and preprocess the central dataset for machine learning.

    Parameters:
        p (dict): Configuration parameters.

    Returns:
        dict: Central dataset with system indicies, spectra, and frequency axes.
    """
    
    with open(dataset_path, "rb") as pklfile: 
        dataset = pickle.load(pklfile)
        
    central_data = {'spectra': np.array(dataset['system spectra']), 
                    'w1': np.array(dataset['system w1 axis']), 
                    'w3': np.array(dataset['system w3 axis']),
                    't2': np.array(dataset['system t2 value']),
                    'number of systems': len(dataset['system w1 axis']),
                    'nt2': len(dataset['system t2 value'][0]),
                    'nw1': len(dataset['system w1 axis'][0]),
                    'nw3': len(dataset['system w3 axis'][0]),
                    'system ID numbers': np.array(dataset['system index'])
                    }
    
    return central_data

# . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 

def classify_central_dataset(p: dict, labels_path: str, central_data: dict) -> tuple:
    """
    Classify datasets into predefined electronic coupling classes.

    Parameters:
        p (dict): Configuration parameters containing class boundaries.
        central_data (dict): Loaded central dataset.

    Returns:
        tuple: (Updated central dataset, classification metadata)
    """
    
    numClasses = len(p['class bounds']) - 1
    class_information = {'N': numClasses, 
                         'bounds': [(p['class bounds'][i], p['class bounds'][i+1]) for i in range(numClasses)],
                         'numbers': np.linspace(0, numClasses-1, numClasses)
                         }
    
    labelsDf = pd.read_csv(labels_path, index_col=0).loc[central_data['system ID numbers'], :]
    all_classes = []
    for system in labelsDf.index:
        J = labelsDf.loc[system, "J_Coul"]
        for i, (lower, upper) in enumerate(class_information['bounds']):
            if lower <= J < upper:
                all_classes.append(i)
                break
    
    central_data['classes'] = np.array(all_classes)
    
    return central_data, class_information

# . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 