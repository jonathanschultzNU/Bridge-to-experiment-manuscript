"""
Module containing helper functions for logging, debugging, and other utilities.
"""

import logging
import sys
import pandas as pd

def check_inputs(p):
    
    options = ['noise_addition', 'pump_bandwidth', 'pump_center', 'dual_pump']
    if p['task'] not in options:
        print_to_log_file(p['log filename'],"ERROR: unsupported task requested. Exiting.")
        raise Exception("ERROR: Unsupported task requested.")
        sys.exit()
    
    # Ensure all necessary keys are specified
    necessary_keys = ['data_path', 'data_labels', 'class_bounds','inSize', 'hiddenSize',
                      'num_epochs', 'batchSize', 'lr', 'p_dropout'] 
    
    # Extend to include keys with available defaults
    necessary_keys.extend(['save_ML_output', 'save_ML_report_images', 'save_2D_plots', 'spec_save_interval',
                           'check_system_ID', 't2_truncate', 'train_test_split', 'torch_seed',
                           'numpy_seed', 'split_seed'])
    
    if p['task'] == 'noise addition':
        necessary_keys.extend(['SNR_filter', 'noise_threshold', 'noise_method', 'noise_fraction'])
        
    elif p['task'] == 'pump_bandwidth' or p['task'] == 'pump_center' or p['task'] == 'dual_pump':
        necessary_keys.extend(['pump_bandwidth', 'pump_center'])
    
    print_to_log_file(p['log filename'], '\nChecking for defaults:')
    for key in necessary_keys:
        if key not in p:
            default_val = get_default_parameters(key)
            if default_val == None:
                print_to_log_file(p['log filename'], f"ERROR: {key} unspecified. Exiting.")
                raise Exception(f"ERROR: {key} unspecified.")
                sys.exit()
            else:
                p[key] = default_val
                print_to_log_file(p['log filename'], f"{key} set to default value of: {p[key]}")
    print_to_log_file(p['log filename'], 'Done\n')
                

def convert_parameter_datatypes(p):
    """
    Convert data types of user-inputs

    Parameters:
    p: A dictionary containing required parameters.

    """
    
    p['class_bounds'] = [float(item) for item in p['class_bounds']]
    
    if p['task'] == 'noise_addition':
        p['noise_fraction'] = [float(item) for item in p['noise_fraction']]
        p['noise_threshold'] = float(p['noise_threshold'])
        
    elif p['task'] == 'pump_bandwidth':    
        p['pump_bandwidth'] = [float(item) for item in p['pump_bandwidth']]
        p['pump_center'] = float(p['pump_center'][0])
        
    elif p['task'] == 'pump_center':    
        p['pump_center'] = [float(item) for item in p['pump_center']]
        p['pump_bandwidth'] = float(p['pump_bandwidth'][0])
        
    elif p['task'] == 'dual_pump':    
        p['pump_center'] = [float(item) for item in p['pump_center']]
        p['pump_bandwidth'] = [float(item) for item in p['pump_bandwidth']]
            
    keys = ['lr', 'p_dropout', 'train_test_split']
    for key in keys:
        p[key] = float(p[key])
        
    keys = ['inSize', 'hiddenSize', 'num_epochs', 'batchSize', 'spec_save_interval']
    for key in keys:
        p[key] = int(p[key])


def get_default_parameters(arg):
    """
    Define and return default parameter value

    Parameters:
    arg (string): key describing the requested parameter

    Returns:
    default_dict[arg]: default value if available, otherwise None
    """
    
    default_dict = {'save_ML_output': 'True', 
                    'save_ML_report_images': 'True', 
                    'save_2D_plots': 'True',
                    'spec_save_interval': 5,
                    'check_system_ID': 19785,
                    't2_truncate': 'False',
                    'train_test_split': 0.8,
                    'torch_seed': 2942,
                    'numpy_seed': 72067,
                    'split_seed': 72067,
                    }
    
    if arg in default_dict:
        return default_dict[arg]
    else:
        return None
    
def initalize_dataframes(p):
    """
    Initialize empty dataframes for logging ML performance and other metrics

    Parameters:

    Returns:
    """
    
    if p['task'] == 'noise_addition':
        scan_key = 'noise_fraction'
        scan_item_df = 'Noise fraction'
        
    elif p['task'] == 'pump_bandwidth':
        scan_key = 'pump_bandwidth'
        scan_item_df = 'Pump bandwidth'
        
    elif p['task'] == 'pump_center':
        scan_key = 'pump_center'
        scan_item_df = 'Pump center'
        
    elif p['task'] == 'ablation':
        scan_key = 'window_centers'
        scan_item_df = 'Window center inds'
        
    elif p['task'] == 'dual_pump':
        scan_key = {}
        scan_item_df = {}
        scan_key[1] = 'pump_bandwidth'
        scan_item_df[1] = 'Pump bandwidth'
        scan_key[2] = 'pump_center'
        scan_item_df[2] = 'Pump center'

    p['scan_key'] = scan_key
    p['scan_item_df'] = scan_item_df
        
    if p['task'] == 'dual_pump':
    
        accuracy_df = pd.DataFrame(index = range(1, p['total_passes'] + 1), 
                                   columns = [scan_item_df[1], scan_item_df[2], 'Train accuracy', 'Train-test accuracy', 'Test accuracy', 'Test top2 accuracy'])
        
        f1_df = pd.DataFrame(index = range(1, p['total_passes'] + 1), 
                             columns = [scan_item_df[1], scan_item_df[2], 'Train micro f1', 'Train macro f1', 'Train weighted f1', 
                                        'Train-test micro f1', 'Train-test macro f1', 'Train-test weighted f1', 
                                        'Test micro f1', 'Test macro f1', 'Test weighted f1'] )
    
    else:
        accuracy_df = pd.DataFrame(index = range(1, p['total_passes'] + 1), 
                                   columns = [scan_item_df, 'Train accuracy', 'Train-test accuracy', 'Test accuracy', 'Test top2 accuracy'])
        
        f1_df = pd.DataFrame(index = range(1, p['total_passes'] + 1), 
                             columns = [scan_item_df, 'Train micro f1', 'Train macro f1', 'Train weighted f1', 
                                        'Train-test micro f1', 'Train-test macro f1', 'Train-test weighted f1', 
                                        'Test micro f1', 'Test macro f1', 'Test weighted f1'] )
    
    return p, accuracy_df, f1_df


def setup_logging(logfile: str):
    """
    Set up logging configuration.

    Parameters:
        logfile (str): Path to log file.
    """
    logging.basicConfig(filename=logfile, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    

def print_to_log_file(filename: str, message: str):
    """
    Append a log message to the specified file.

    Parameters:
        filename (str): Path to log file.
        message (str): Log message.
    """
    with open(filename, "a") as logfile:
        logfile.write(f'{message}\n')
        

def remove_workspace_variables():
    """
    Clears large unused variables to free memory before the next iteration.
    """
    variables_to_delete = ['iteration_data', 'training_data', 'testing_data']
    for var_name in variables_to_delete:
        try:
            del var_name
        except KeyError:
            pass
        

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
