"""
Module to handle dataset loading, transformations, and classification.
"""
import os
import numpy as np
import pandas as pd
import pickle
from typing import Optional
from modules.utils import extract_pattern

# . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 

def load_system_data(database_path: str, sys_ID: str):
    """
    Load a system's dataset from a pickled file.

    Parameters:
        database_path (str): Path to the directory containing dataset files.
        sys_ID (str): Unique identifier for the system dataset.

    Returns:
        tuple: (2D spectra array, w1 values, w3 values, t2 values)
    """
    filepath = os.path.join(database_path, f'{sys_ID}.pkl')
    with open(filepath, "rb") as pklfile:
        fullEntry = pickle.load(pklfile)
    return fullEntry["Rw1w3Absorptive"], fullEntry['w1'], fullEntry['w3'], fullEntry['t2']

# . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 

def load_central_dataset(runlist_file: Optional[str], p: dict) -> dict:
    """
    Load and preprocess the central dataset for machine learning.

    Parameters:
        runlist_file (str): File containing a list of system IDs to load.
        p (dict): Configuration parameters.

    Returns:
        dict: Central dataset with system IDs, spectra, and metadata.
    """

    if runlist_file is None:
        filenames = [f for f in os.listdir(p['data_path']) if f.endswith(".pkl")]
        
        # Extract 7-digit numbers from filenames and convert to integers
        central_ID_numbers = [
            int(name) for file in filenames if (name := extract_pattern(pattern=r'\d{7}', arg=file))
        ]
        
    else:
        central_ID_numbers = np.loadtxt(runlist_file, dtype=int)
    
    central_data = {'system ID numbers': central_ID_numbers, 'system IDs': [f"Hamiltonian{str(i).zfill(7)}" for i in central_ID_numbers]}
    
    data_full, w1, w3, t2 = [], [], [], []
    for system_ID in central_data['system IDs']:
        data_temp, w1_temp, w3_temp, t2_temp = load_system_data(p['data_path'], system_ID)
        data_full.append(data_temp)
        w1.append(w1_temp)
        w3.append(w3_temp)
        t2.append(t2_temp)
    
    central_data.update({'spectra': np.array(data_full), 'w1': np.array(w1), 'w3': np.array(w3), 't2': np.array(t2)})
    return central_data

# . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 

def classify_central_dataset(p: dict, central_data: dict) -> tuple:
    """
    Classify datasets into predefined electronic coupling classes.

    Parameters:
        p (dict): Configuration parameters containing class boundaries.
        central_data (dict): Loaded central dataset.

    Returns:
        tuple: (Updated central dataset, classification metadata)
    """
    numClasses = len(p['class_bounds']) - 1
    class_information = {'N': numClasses, 'bounds': [(p['class_bounds'][i], p['class_bounds'][i+1]) for i in range(numClasses)]}
    
    labelsDf = pd.read_csv(p['data_labels'], index_col=0).loc[central_data['system IDs'], :]
    all_classes = []
    for system in labelsDf.index:
        J = labelsDf.loc[system, "J"]
        for i, (lower, upper) in enumerate(class_information['bounds']):
            if lower <= J < upper:
                all_classes.append(i)
                break
    
    central_data['classes'] = np.array(all_classes)
    return central_data, class_information

# . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 