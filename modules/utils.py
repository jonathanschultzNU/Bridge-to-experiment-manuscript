"""
Module containing helper functions for logging, debugging, and other utilities.
"""

import pandas as pd
from git import Repo, exc
import re
import numpy as np

# . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 

def extract_pattern(pattern, arg=None):
    """Extracts the pattern from the input string arg."""
    if arg is None:
        return None
    match = re.search(pattern, arg)
    return match.group(0) if match else None

# . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 

def get_git_info():
    try:
        repo = Repo(search_parent_directories=True)
        branch = repo.active_branch.name
        commit_hash = repo.head.object.hexsha
        tag = next((t.name for t in repo.tags if t.commit == repo.head.object), "No tag available")
        return f"Branch: {branch}\nCommit: {commit_hash}\nTag: {tag}"
    except exc.InvalidGitRepositoryError:
        return "Not a Git repository."

# . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 

def read_input_file(f):
    from pathlib import Path
    p = {}
    data = f.readlines()
    keys = ['class bounds', 'noise fraction', 'bandwidth', 'center frequency']
    for line in data:
        key, value = line.split("=")
        if key.strip() == "data_path" or key.strip() == "data_labels":
           p[key.strip()] = Path(r"{value.strip()}")
        if key.strip() in keys:
           entries = value.count(",") + 1 
           entry_values = value.split(",")
           
           temp = []
           for i in range(entries):
               temp.append(entry_values[i].strip())
           p[key.strip()] = np.array(temp)
        else:
           p[key.strip()] = value.strip()
           
    p['f1 scores'] = ['micro', 'macro', 'weighted', None]
           
    return p

# . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 

def check_inputs(p, dirs):
    """
    Validates input parameters and ensures required keys are present.
    
    Parameters:
        p (dict): Dictionary containing user-defined parameters.
        dirs (dict): Dictionary containing directory paths.
    """
    
    # Define valid task options
    valid_tasks = ['noise', 'bandwidth', 'center frequency', 'bandwidth and center frequency']
    if p['task'] not in valid_tasks:
        print_to_log_file(dirs['log file'], "ERROR: Unsupported task requested. Exiting.")
        raise ValueError(f"Unsupported task requested: {p['task']}")

    # Required keys
    required_keys = ['class bounds', 'input size', 'hidden layer size',
        'number of epochs', 'batch size', 'learning rate', 'dropout probability'
    ]
    
    # Keys with defaults
    optional_keys = [
        'save ML output', 'save ML report images', 'save 2D plots', 'spec save interval',
        'check-system ID', 't2 truncate', 'train-test split', 'torch seed', 'numpy seed', 'split seed'
    ]

    if p['task'] == 'noise':
        required_keys.extend(['SNR filter', 'SNR threshold', 'noise method', 'noise fraction'])

    elif p['task'] in ['bandwidth', 'center frequency', 'bandwidth and center frequency']:
        required_keys.extend(['bandwidth', 'center frequency'])

    # Check for missing keys
    print_to_log_file(dirs['log file'], "\nChecking for missing parameters:")
    for key in required_keys + optional_keys:
        if key not in p:
            default_val = get_default_parameters(key)
            if default_val is None:
                print_to_log_file(dirs['log file'], f"ERROR: {key} unspecified. Exiting.")
                raise KeyError(f"Missing required parameter: {key}")
            else:
                p[key] = default_val
                print_to_log_file(dirs['log file'], f"{key} set to default value: {p[key]}")
    
    print_to_log_file(dirs['log file'], "All required parameters verified.\n")

# . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 
                
def convert_parameter_datatypes(p):
    """
    Converts input parameters to appropriate data types.

    Parameters:
        p (dict): Dictionary containing user-defined parameters.
    """
    
    p['class bounds'] = [float(item) for item in p['class bounds']]
    
    if p['task'] == 'noise':
        p['noise fraction'] = [float(item) for item in p['noise fraction']]
        p['SNR threshold'] = float(p['SNR threshold'])
        
    elif p['task'] == 'bandwidth':    
        p['bandwidth'] = [float(item) for item in p['bandwidth']]
        p['center frequency'] = float(p['center frequency'][0])
        
    elif p['task'] == 'center frequency':    
        p['center frequency'] = [float(item) for item in p['center frequency']]
        p['bandwidth'] = float(p['bandwidth'][0])
        
    elif p['task'] == 'bandwidth and center frequency':    
        p['center frequency'] = [float(item) for item in p['center frequency']]
        p['bandwidth'] = [float(item) for item in p['bandwidth']]
        
        
    float_keys = ['learning rate', 'dropout probability', 'train-test split']
    int_keys = ['input size', 'hidden layer size', 'number of epochs', 'batch size', 'spec save interval']

    for key in float_keys:
        p[key] = float(p[key])

    for key in int_keys:
        p[key] = int(p[key])

# . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 

def get_default_parameters(arg):
    """
    Returns the default parameter value if available.

    Parameters:
        arg (str): Name of the parameter.

    Returns:
        The default value if found, otherwise None.
    """
    
    default_dict = {'save ML output': 'True', 
                    'save ML report images': 'False', 
                    'save 2D plots': 'False',
                    'spec save interval': 5,
                    'check-system ID': 11,
                    't2 truncate': 'False',
                    'train-test split': 0.8,
                    'torch seed': 2942,
                    'numpy seed': 72067,
                    'split seed': 72067,
                    }
    
    return default_dict.get(arg, None)

# . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 
    
def initialize_dataframes(p):
    """
    Initializes dataframes for logging ML performance and other metrics.

    Parameters:
        p (dict): Dictionary containing user-defined parameters.

    Returns:
        tuple: Updated parameter dictionary, accuracy dataframe, F1-score dataframe.
    """
    
    task_to_scan_key = {
        'noise': 'noise fraction',
        'bandwidth': 'bandwidth',
        'center frequency': 'center frequency',
        'bandwidth and center frequency': {'1': 'bandwidth', '2': 'center frequency'}
        }
    
    scan_key = task_to_scan_key.get(p['task'], None)
    p['scan key'] = scan_key

        
    if p['task'] == 'bandwidth and center frequency':
    
        accuracy_columns = ['bandwidth', 'center frequency', 'Train accuracy', 'Train-test accuracy', 'Test accuracy', 'Test top2 accuracy']
        f1_columns = [
            'bandwidth', 'center frequency', 'Train micro f1', 'Train macro f1', 'Train weighted f1',
            'Train-test micro f1', 'Train-test macro f1', 'Train-test weighted f1',
            'Test micro f1', 'Test macro f1', 'Test weighted f1'
        ]
    
    else:
        accuracy_columns = [scan_key, 'Train accuracy', 'Train-test accuracy', 'Test accuracy', 'Test top2 accuracy']
        f1_columns = [
            scan_key, 'Train micro f1', 'Train macro f1', 'Train weighted f1',
            'Train-test micro f1', 'Train-test macro f1', 'Train-test weighted f1',
            'Test micro f1', 'Test macro f1', 'Test weighted f1'
        ]
        
    accuracy_df = pd.DataFrame(index=range(1, p['total passes'] + 1), columns=accuracy_columns)
    f1_df = pd.DataFrame(index=range(1, p['total passes'] + 1), columns=f1_columns)
    
    return p, accuracy_df, f1_df
    
# . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 

def print_to_log_file(filename: str, message: str):
    """
    Append a log message to the specified file.

    Parameters:
        filename (str): Path to log file.
        message (str): Log message.
    """
    with open(filename, "a") as logfile:
        logfile.write(f'{message}\n')

# . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 

def remove_workspace_variables(variables_to_delete):
    """
    Clears variables from workspace
    """
    for var_name in variables_to_delete:
        try:
            del var_name
        except KeyError:
            pass
        
# . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 

def get_list_dimensions(lst):
    """
    Recursively determines the depth of a nested list.

    Parameters:
        lst (list): The list whose dimensions are to be determined.

    Returns:
        int: The depth of the list.
    """
    if not isinstance(lst, list):
        return 0
    return 1 + max(get_list_dimensions(item) for item in lst)

# . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 