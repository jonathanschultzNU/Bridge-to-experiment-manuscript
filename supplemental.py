import math
import numpy as np
import sys
from copy import deepcopy
import matplotlib.pyplot as plt
import pickle
from sklearn.metrics import f1_score
import subprocess
import torch
from torch.utils.data import DataLoader
import pandas as pd
import time
from sklearn.metrics import top_k_accuracy_score
from collections import Counter

# %% Classes

import torch.nn as nn
from torch.utils.data import Dataset

class PyTorchDataset(Dataset):
    def __init__(self, data, p, isTrain=True):
        '''
        Description: Takes data dictionary and formats necessary numpy & Torch arrays to be fed to PyTorch dataloader
        
        INPUTS
        -------
        data .......... dictionary containing arrays for the spectral data, system ID #s, labels, and t2 values
        p.............. dictionary containing all inputs of the job
        split_seed .... for shuffling train/test data (ensure they are shuffled with the same seed)
        isTrain ....... is it the training dataset?
        split ......... train/test split
        
        CLASS ATTRIBUTES
        -------
        self.imgs ............ 3D Torch array of all spectra, shape: [# systems x t2 points, w1 points, w3 points]
        self.img_labels ...... 1D Torch array of classes corresponding to spectra, shape: [# systems x t2 points,]
        self.img_IDnums ...... 1D Torch array of run ID numbers corresponding to spectra, shape: [# systems x t2 points,]
        self.img_t2s ......... 1D Torch array of t2 timepoints corresponding to spectra, shape: [# systems x t2 points,]
        self.num_imgs ........ total number of images in the dataset
        self.num_sys ......... total number of systems reflected in the dataset
        '''
        
        system_ID_numbers = data['system ID numbers'] 
        sys_data = data['spectra']
        classes = data['classes']
        t2 = data['t2']
        
        # shuffle lists/arrays in correspondence with one another
        rng = np.random.default_rng(seed = p['split_seed'])
        shuffle_indices = rng.permutation(len(system_ID_numbers)).astype(int)
        system_ID_numbers = system_ID_numbers[shuffle_indices]
        classes = classes[shuffle_indices]
        sys_data = sys_data[shuffle_indices]
        t2 = t2[shuffle_indices]
        
        split_point = int(p['train_test_split'] * len(system_ID_numbers))

        # select appropriate subset of data for train/test
        if isTrain:
            system_ID_numbers = system_ID_numbers[0:split_point]
            classes = classes[0:split_point]
            t2 = t2[0:split_point,:]
            sys_data = sys_data[0:split_point,:,:,:]
            subset = 'TRAINING'
        else:
            system_ID_numbers = system_ID_numbers[split_point:]
            classes = classes[split_point:]
            t2 = t2[split_point:,:]
            sys_data = sys_data[split_point:,:,:,:]
            subset = 'TESTING'
        
        # Calculate the shape of the final data array
        sys_data = sys_data.transpose(0, 3, 1, 2)
        num_sys, nt2s, nw, _ = sys_data.shape
        final_data_shape = (num_sys * nt2s, nw, nw)
        data_array = sys_data.reshape(final_data_shape)
        
        final_t2_shape = (num_sys * nt2s)
        t2_array = t2.reshape(final_t2_shape)
        
        if p['task'] == 'noise_addition':
        
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
                print_to_log_file(p['log filename'], f'ERROR: full mask has length {len(full_mask)} (expected {len(system_ID_numbers)*nt2s}). Now exiting.')
                sys.exit()
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
        
        print_to_log_file(p['log filename'], 
                               f'{subset} dataset summary:\nTotal spectra: {self.num_imgs}\nTotal systems: {self.num_sys}')
        
    def __getitem__(self, index):
        return self.imgs[index], self.img_labels[index], self.img_IDnums[index], self.img_t2s[index]

    def __len__(self):
        return self.num_imgs

class NeuralNet(nn.Module):

    def __init__(self, inputSize, hiddenSize, numClasses, dropout_val):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(inputSize,
                            hiddenSize)  # linear transform in hidden layer (between orig data and input into hidden) 1
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hiddenSize, numClasses)  # linear transform in output layer (between hidden 1 and out)
        self.dropout1 = nn.Dropout(dropout_val)
        
    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.dropout1(out)
        out = self.l2(out)
        return out

# %% Inputs and initializing

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


def initialize_check_system(p, central_data):
    '''
    DESCRIPTION: Prep to plot an example 2D spectrum per iteration
    INPUTS: 
        p ... 
        central_data ..........
        
    RETURNS:
        check_spectrum............ dictionary
    '''
    system_index_selected = find_image_index(central_data, p['check_system_ID'])
    if system_index_selected == None:
        import random
        system_index_selected = random.randint(0, central_data['Number of systems']-1)
    check_spectrum = {'system index selected': system_index_selected, 'w1': central_data['w1'][system_index_selected,:], 
                      'w3': central_data['w3'][system_index_selected,:], 't2': central_data['t2'][system_index_selected, 0], 
                      'system ID number': central_data['system ID numbers'][system_index_selected]}
    
    return check_spectrum


# %% Data loading and processing
        
def load_system_data(database_path, sys_ID):
    '''
    DESCRIPTION: loads system with specific run name from the directory data_path
    INPUTS: 
        database_path ... path to data
        sys_ID .......... ID number for system
        
    RETURNS:
        idata............ array of shape [1, nw1, nw3, nt2] containing 2DES spectra
    '''
    
    import os
    
    filepath = os.path.join(database_path, f'{sys_ID}.pkl')
    with open(filepath, "rb") as pklfile:
        fullEntry = pickle.load(pklfile)

    idata = fullEntry["Rw1w3Absorptive"]
    w1 = fullEntry['w1']
    w3 = fullEntry['w3']
    t2 = fullEntry['t2']
    
    return idata, w1, w3, t2

def load_central_dataset(runlist_file, p):
    central_ID_numbers = []
    f = open(runlist_file)
    data = f.readlines()
    for line in data:
        central_ID_numbers.append(int(line))
    f.close()
    
    central_data = {'system ID numbers': np.array(central_ID_numbers), 'system IDs': [f"runJS{str(i).zfill(7)}" for i in central_ID_numbers],
                    'Number of systems': len(central_ID_numbers)}
    
    start_time = time.perf_counter()
    print_to_log_file(p['log filename'], " ~~~~ STARTING to load data ~~~~ ")
    
    data_full = list()   # final shape = [Number of systems, w1, w3, t2]
    w1 = list()
    w3 = list()
    t2 = list()
    message_interval = math.ceil(central_data['Number of systems'] // 10)
    for index, system_ID in enumerate(central_data['system IDs']):
        data_temp, w1_temp, w3_temp, t2_temp = load_system_data(p['data_path'], system_ID)
        
        if p['t2_truncate'] == 'True':
            data_temp = data_temp[:,:,0:int(p['t2_truncate_ind'])]
            t2_temp = t2_temp[0:int(p['t2_truncate_ind'])]
        
        data_full.append(data_temp)
        w1.append(w1_temp)
        w3.append(w3_temp)
        t2.append(t2_temp)
        if (index+1) % message_interval == 0:
            completion_percentage = 100*(index+1)/central_data['Number of systems']
            print_to_log_file(p['log filename'], f"{completion_percentage:.0f}% loaded")
    
    central_data.update(
        {'spectra': np.array(data_full), 'w1': np.array(w1), 'w3': np.array(w3),
         't2': np.array(t2), 'nw1': len(w1[0].squeeze()), 
         'nw3': len(w3[0].squeeze()), 'nt2': len(t2[0].squeeze())}
        )
    
    time_elapsed = np.round((time.perf_counter() - start_time))
    print_to_log_file(p['log filename'], f''' ~~~~ FINISHED loading data after {time_elapsed} sec ~~~~ 
Number of systems: {central_data['Number of systems']} 
Data shape:
{central_data['spectra'].shape}
loading time per system: {np.round(1000*time_elapsed/central_data['Number of systems'], decimals = 2)} ms per spectrum''')

    return central_data


def classify_central_dataset(p, central_data):
    
    numClasses = len(p['class_bounds'])-1
    class_information = {'N': numClasses, 
               'bounds': [(p['class_bounds'][i], p['class_bounds'][i+1]) for i in range(numClasses)],
               'numbers': np.linspace(0, numClasses-1, numClasses)}
    
    labelsDf = pd.read_csv(p['data_labels'], index_col=0).loc[central_data['system IDs'], :]
    print_to_log_file(p['log filename'], f'''labels df:\n{labelsDf.index}''')

    all_classes = list()    
    for system in labelsDf.index:
        J = labelsDf.loc[system, "J"]
        current_class = -1
        for i, (lower, upper) in enumerate(class_information['bounds']):
            if J < 0:
                if lower < J <= upper:
                    current_class = i
                    break
            elif J == 0:
                if lower < J < upper:
                    current_class = i
                    break
            elif J > 0:
                if lower <= J < upper:
                    current_class = i
                    break

        if current_class == -1:
            print_to_log_file(p['log filename'], "ERROR: An electronic coupling is out of range (last value of J was: {J}). Now exiting.")
            sys.exit()
        else:
            all_classes.append(current_class)
    
    central_data['classes'] = np.array(all_classes)

    if central_data['spectra'].shape[0] != len(central_data['classes']):
        raise Exception("ERROR: Fewer classes than spectra")
        sys.exit()
        
    return central_data, class_information


def self_normalize_datasets(data_dict, logfilename):
    
    maxvals = []
    for i in range(data_dict['Number of systems']):
        temp = deepcopy(data_dict['spectra'][i,:,:,:])
        mv = np.max(np.max(np.max(np.abs(temp))))
        maxvals.append(mv) # log maximum signals  
        
    print_to_log_file(logfilename, f'''
Largest max signal = {np.max(maxvals)}
Smallest max signal = {np.min(maxvals)}''')

    if all(element == np.float64(1) for element in maxvals):
        print_to_log_file(logfilename, 'Data already normalized.')  
    
    else:
        print_to_log_file(logfilename, 'Now self-normalizing data for each system...')       
    
        for i in range(data_dict['Number of systems']):
            data_dict['spectra'][i,:,:,:] = data_dict['spectra'][i,:,:,:]/maxvals[i]
    
        maxvals = []
        for i in range(data_dict['Number of systems']):
            temp = deepcopy(data_dict['spectra'][i,:,:,:])
            mv = np.max(np.max(np.abs(temp)))
            maxvals.append(mv) # log maximum signals  
            
        print_to_log_file(logfilename, f'''Post normalization:
    Largest signal = {np.max(maxvals)}
    Smallest signal = {np.min(maxvals)}''')    
        
    return data_dict


def pollute_data(p, central_data, iteration_inds):
    
    data = deepcopy(central_data)
    data = self_normalize_datasets(data, p['log filename'])
    
    if p['num_vars'] == 1:
        
        iteration_number = iteration_inds[0]
    
        if p['task'] == 'noise_addition':
            data, SNR_summary = add_noise(data, p['log filename'], p['noise_method'], p['noise_fraction'][iteration_number], p['SNR_filter'], p['noise_threshold'])
            return data, SNR_summary
        
        elif p['task'] == 'pump_bandwidth':
            data = apply_pump(data, p['pump_center'], p['pump_bandwidth'][iteration_number])
            return data, None
        
        elif p['task'] == 'pump_center':
            data = apply_pump(data, p['pump_center'][iteration_number], p['pump_bandwidth'])
            return data, None
        
        elif p['task'] == 'ablation':
            from functions_ablation import ablate
            data = ablate(data, p['window_width'], p['window_centers'], iteration_number)
            return data, None
    
    elif p['num_vars'] == 2:
        
        if p['task'] == 'dual_pump':
            data = apply_pump(data, p['pump_center'][iteration_inds[1]], p['pump_bandwidth'][iteration_inds[0]])
            return data, None
        


def add_noise(data_dict, logfilename, method, noise_fraction, SNR_filter, threshold):
    '''
    Parameters
    ----------
    data_dict .........
    logfilename .......
    method ............
    noise_fraction ....
    SNR_filter ........
    threshold .........

    Returns
    -------
    data_dict .........
    
    Notes
    -------
    Seeds for noise addition are based on the runID (+10000 for floor noise, 
                                                     +20000 for intensity noise)
    '''
    
    # generate arrays to log widths of clean spectra and noise
    data_dict['noise width'] = np.zeros((data_dict['Number of systems'], data_dict['nt2']))
    data_dict['signal width'] = np.zeros((data_dict['Number of systems'], data_dict['nt2']))
    
    # SeedVec = np.arange(start = startSeed, stop = startSeed + data_dict['Number of systems'], step = 1, dtype = int)
    SeedVec = deepcopy(data_dict['system ID numbers'])
    
    # signal + max(signal) * noise
    if method == 'max_scaling':
            
        # list of maximum signals as a function of system
        maxvals = []
        for i in range(data_dict['Number of systems']):
            temp = deepcopy(data_dict['spectra'][i,:,:,:])
            mv = np.max(np.max(np.abs(temp)))
            maxvals.append(mv)
        
        absolute_max = np.max(maxvals)
        
        for i in range(data_dict['Number of systems']):
            
            rng = np.random.default_rng(seed = SeedVec[i] + 10000)
            
            # generate new random noise array for each system
            noise_array = rng.normal(0, noise_fraction*absolute_max, size=(1, data_dict['nw1'], data_dict['nw3'], data_dict['nt2']))
            noise_array = np.float64(noise_array)
            
            # log signal and noise widths
            for j in range(data_dict['nt2']):
                indv_spec = deepcopy(data_dict['spectra'][i,:,:,j])
                data_dict['signal width'][i][j] = np.mean(np.abs(indv_spec))
                
                data_dict['noise width'][i][j] = noise_fraction*absolute_max
            
            # add noise
            data_dict['spectra'][i,:,:,:] = data_dict['spectra'][i,:,:,:] + noise_array
    
    # signal + signal * noise
    elif method == 'profile_scaling': 
        
        for i in range(data_dict['Number of systems']):
            
            # generate new random noise array for each system
            rng = np.random.default_rng(seed = SeedVec[i] + 10000)
            
            noise_temp = rng.normal(0, noise_fraction, size=(1, data_dict['nw1'], data_dict['nw3'], data_dict['nt2']))
            noise_temp = np.float64(noise_temp)
            
            temp = deepcopy(data_dict['spectra'][i,:,:,:])
            noise_array = np.multiply(temp, noise_temp)
            
            # log signal and noise widths
            for j in range(data_dict['nt2']):
                
                indv_spec = deepcopy(data_dict['spectra'][i,:,:,j])
                data_dict['signal width'][i][j] = np.mean(np.abs(indv_spec))
                
                indv_noise = deepcopy(noise_array[i,:,:,j])
                data_dict['noise width'][i][j] = np.mean(np.abs(indv_noise))
              
            # add noise    
            data_dict['spectra'][i,:,:,:] = data_dict['spectra'][0,:,:,:] + noise_array 
            
    else:
        print_to_log_file(logfilename, " ERROR: invalid method specified. Now exiting...")
        sys.exit()
    
    if SNR_filter:
        
        SNR_summary = dict()
        SNR_summary['per system'] = dict()
        SNR_summary['per system']['System IDs'] = list()
        SNR_summary['per system']['SNR at t2 = 0'] = list()
        SNR_summary['per system']['SNR at t2 = end'] = list()
        SNR_summary['per system']['avg SNR of all t2'] = list()
        SNR_summary['per system']['avg SNR of kept t2'] = list()
        SNR_summary['per system']['avg SNR of dropped t2'] = list()
        
        data_dict['dropped images'] = {}
        spec_count = 0
        drop_count = 0
        avg_drop_SNR = 0
        avg_keep_SNR = 0
        avg_total_SNR = 0
        
        for i in range(data_dict['Number of systems']):
            ID_temp = data_dict['system ID numbers'][i]
            
            data_dict['dropped images'][ID_temp] = np.ones(data_dict['nt2'], dtype=bool)
            drop_count_perSys = 0
            avg_keep_SNR_perSys = 0
            avg_drop_SNR_perSys = 0
            avg_total_SNR_perSys = 0
            
            SNR_summary['per system']['System IDs'].append(ID_temp.copy())
            
            for j in range(data_dict['nt2']):
                noise_width = data_dict['noise width'][i][j]
                signal_width = data_dict['signal width'][i][j]
                
                spec_count += 1
                SNR = signal_width/(noise_width + 0.0000000001) # avoid dividing by zero
                
                if j == 0:
                    SNR_summary['per system']['SNR at t2 = 0'].append(SNR)
                elif j == data_dict['nt2']-1:
                    SNR_summary['per system']['SNR at t2 = end'].append(SNR)
                    
                avg_total_SNR += SNR 
                avg_total_SNR_perSys += SNR
                
                if SNR < threshold:
                    data_dict['dropped images'][ID_temp][j] = False
                    drop_count += 1
                    drop_count_perSys += 1
                    avg_drop_SNR += SNR
                    avg_drop_SNR_perSys += 1
                else:
                    avg_keep_SNR += SNR 
                    avg_keep_SNR_perSys += SNR 
                    
            SNR_summary['per system']['avg SNR of all t2'].append(avg_total_SNR_perSys/data_dict['nt2'])
            
            if drop_count_perSys > 0:
                SNR_summary['per system']['avg SNR of kept t2'].append(avg_keep_SNR_perSys/(data_dict['nt2']-drop_count_perSys))
                SNR_summary['per system']['avg SNR of dropped t2'].append(avg_drop_SNR_perSys/drop_count_perSys)
                
        SNR_summary['number of total spectra'] = spec_count
        SNR_summary['avg SNR of all spectra'] = avg_total_SNR/spec_count
        SNR_summary['avg SNR of t2 = 0 spectra'] = np.mean(SNR_summary['per system']['SNR at t2 = 0'])
        SNR_summary['avg SNR of kept spectra'] = avg_keep_SNR/(spec_count-drop_count)
        
        if drop_count > 0:
            
            SNR_summary['number of dropped spectra'] = drop_count
            SNR_summary['avg SNR of dropped spectra'] = avg_drop_SNR/drop_count
            
            print_to_log_file(logfilename, f'''\n
     ! ! ! ! ! DROP WARNING ! ! ! ! !
     
spectra dropped (low SNR): {drop_count} of {spec_count} ({100*drop_count/spec_count} %)
Average SNR of dropped spectra: {avg_drop_SNR/drop_count}
Average SNR of kept spectra: {avg_keep_SNR/(spec_count-drop_count)}
Average SNR of all spectra: {avg_total_SNR/spec_count}\n''')

        else:
            print_to_log_file(logfilename, f'''\n
No spectra were dropped due to low SNR
Average SNR of all spectra: {avg_total_SNR/spec_count}\n''')
   
    return data_dict, SNR_summary


def apply_pump(data_dict, carrier, bandwidth):
    '''
    Parameters
    ----------
    data_dict ...... copy of centralized data dictionary
    carrier ........ center frequency [cm-1] of the pulse 
    bandwidth ...... spectral bandwidth [cm-1] of the pulse (full-width at half-maximum)

    Returns
    -------
    data_dict ...... modified data dictionary

    '''
    constant = 10000
    carrier = carrier/constant
    bandwidth = bandwidth/constant
    
    data_dict['pump spectra'] = {}
    
    for index, ID in enumerate(data_dict['system ID numbers']):
        w1 = deepcopy(data_dict['w1'][index])
        w1 = w1/constant
   
        pump_spectrum = np.exp((-4*np.log(2)*np.square(w1-carrier))/bandwidth**2)  # Gaussian envelope
        data_dict[ID] = deepcopy(pump_spectrum)

        pump_convolution_2d = np.tile(pump_spectrum,(len(w1),1))
        pump_convolution_3d = np.dstack([pump_convolution_2d]*data_dict['nt2'])
        
        data_dict['spectra'][index,:,:,:] = np.multiply(data_dict['spectra'][index,:,:,:], pump_convolution_3d)

    return data_dict
    
# %% Operational

def print_to_log_file(filename, string):
    with open(filename, "a") as logfile: 
        logfile.write(f'{string}\n')
        

def read_input_file(f):
    from pathlib import Path
    p = {}
    data = f.readlines()
    keys = ['class_bounds', 'noise_fraction', 'pump_bandwidth', 'pump_center']
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


def recursive_filter(system_Df, current_system, current_filter, filter_counter, logfilename):
    '''
    DESCRIPTION: Uses user-specified filters to determine if a dataset should be included
    INPUTS:
        system_Df ........... dataframe of system labels
        current_system ...... system ID
        current_filter ...... filter criteria
        filter_counter ...... which filter among full filter list
        logfilename ......... name of .OREOSlog file)
        
    RETURNS:
        filter_result ....... [true/false] whether system should be loaded
    '''
    
    filter_result = False
    if isinstance(current_filter[1], str):
        check = system_Df.loc[current_system, current_filter[0]]
        if current_filter[0] == 'vib_freq':
            entries = check.count(",") + 1 
            if entries == int(current_filter[1]):
                filter_result = True
        else:
            if check.find(current_filter[1]) != -1:
                filter_result = True
    elif isinstance(current_filter[1], int):
        check = system_Df.loc[current_system, current_filter[0]]
        if int(check) == current_filter[1]: 
            filter_result = True
    else:
        if logfilename is not None:
            print_to_log_file(logfilename, f"WARNING: Filter values must be of type string or int. Filter request {filter_counter+1} not applied.")
        else:
            print(f"WARNING: Filter values must be of type string or int. Filter request {filter_counter+1} not applied.")
    return filter_result


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
        For `average=None`, the value is a tuple of (F1 score list, label list).
    """
    # Validate average_methods_list
    valid_methods = ['micro', 'macro', 'weighted', None]
    if not all(method in valid_methods for method in average_methods_list):
        raise ValueError(f"Invalid averaging method(s) in {average_methods_list}. "
                         f"Supported methods are: {valid_methods}")
        
    from sklearn.utils.multiclass import unique_labels

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

def remove_workspace_variables():
    # remove large variables before heading into next iteration  
    variables_to_delete = ['iteration_data', 'iteration_results',
                           'training_data', 'training_dict', 'train_running_dict',
                           'testing_data', 'testing_dict', 'testing_running_dict',
                           'testing_in_training_dict', 'train_test_running_dict', 'pollute_summary']
    
    for var_name in variables_to_delete:
        try:
            del var_name
        except KeyError:
            pass
    
    

# %% Plotting

def create_and_save_line_plot(x, y, title, x_label, y_label, ymin, ymax, save_path):
    """
    Creates a simple line plot and saves it to the specified path.

    Args:
        x (array-like): X-axis data.
        y (array-like): Y-axis data.
        title (str): Title for the plot.
        save_path (str): Path to save the plot (including filename and extension).

    Returns:
        None
    """
    plt.figure(figsize=(8, 6))  # Set the figure size (optional)
    plt.plot(x, y, linewidth=2)
    plt.title(title, fontsize=20, fontweight='bold')
    plt.xlabel(x_label, fontsize=18)
    plt.ylabel(y_label, fontsize=18)
    plt.ylim((ymin, ymax))
    plt.xticks(fontname='DejaVu Sans', fontsize=16)
    plt.yticks(fontname='DejaVu Sans', fontsize=16)
    plt.grid(True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
def imagesc(data,title,save=False):
        
    fig, ax = fig, ax = plt.subplots()
    ax.imshow(data, extent=[0,data.shape[1],0,data.shape[0]], cmap='seismic')
    ax.set_title(f'{title}')
    plt.tight_layout()
    if save:
        fig.savefig(f'{title}.png')
    return fig

def plot_spectrum(data_dict, save_path):
    
    minsize = 16
    medsize = minsize+2
    bigsize = minsize+6
    
    plt.rc('font', size = minsize)          # controls default text sizes
    plt.rc('axes', titlesize = bigsize)     # fontsize of the axes title
    plt.rc('axes', labelsize = medsize)     # fontsize of the x and y labels
    plt.rc('xtick', labelsize = minsize)    # fontsize of the tick labels
    plt.rc('ytick', labelsize = minsize)    # fontsize of the tick labels
    plt.rc('figure', titlesize = bigsize)   # fontsize of the figure title
    ystr = "$\omega_{3} /2 \pi c (10^{3} cm^{-1})$"
    xstr = "$\omega_{1} /2 \pi c (10^{3} cm^{-1})$"
    
    maxval = np.max(np.abs(data_dict['spectrum']))
    
    fig, ax =plt.subplots(ncols=1,nrows=1, figsize=(7, 7))
    ax.contour(data_dict['w1']/1000, data_dict['w3']/1000, data_dict['spectrum'], levels=np.arange(-maxval, maxval, step=0.1*maxval), colors='k', linestyles='dashed', linewidths=0.25)
    ax.contourf(data_dict['w1']/1000, data_dict['w3']/1000, data_dict['spectrum'], levels=np.arange(-maxval, maxval, step=0.01*maxval), cmap='seismic')
    ax.set_ylabel(ystr)
    ax.set_xlabel(xstr)
    ax.set_title(f"{data_dict['system ID number']}, t2 = {data_dict['t2']} fs")
    ax.set_aspect('equal',adjustable='box')
    ax.set_xlim([data_dict['w1'].min()/1000, data_dict['w1'].max()/1000])
    ax.set_ylim([data_dict['w3'].min()/1000, data_dict['w3'].max()/1000])
            
    fig.tight_layout()
    plt.ioff()
    plt.savefig(save_path, dpi=300)
    plt.close(fig)

def save_checked_spectrum(p, check_spectrum, iteration_data, iteration_number):
    if p['save_2D_plots'] == 'True':
        
        import os
        save_loc = os.path.join(p['outputs_path'],"2D_check_plots")
        if not os.path.exists(save_loc):
            os.makedirs(save_loc)
            
        if (iteration_number+1) % p['spec_save_interval'] == 0:
            try:
                check_spectrum['spectrum'] = iteration_data['spectra'][check_spectrum['system index selected'],:,:,0]
                plot_spectrum(check_spectrum, f"{save_loc}/job{p['Slurm job ID']}_spec_{iteration_number+1}.png")
                with open(f"{save_loc}/job{p['Slurm job ID']}_spec_{iteration_number+1}.pkl", 'wb') as f:
                    pickle.dump(check_spectrum, f)
                
            except Exception as e:
                print_to_log_file(p['log filename'], f'An error occured while generating or saving 2D spectrum: {e}')


def save_ml_reports(p, training_dict, testing_in_training_dict, iteration_number):
 if p['save_ML_report_images'] == 'True':
     try:
         import os
         save_loc = os.path.join(p['outputs_path'],"Training_plots")
         if not os.path.exists(save_loc):
             os.makedirs(save_loc)
             
         xvect = np.arange(p['num_epochs']) + 1
         keys_to_plot = ['accuracy', 'micro f1', 'macro f1']
         for key in keys_to_plot:
             create_and_save_line_plot(xvect, training_dict[key], key, 'epoch', f'{key} (%)', 0, 100, 
                                       f"{save_loc}/job{p['Slurm job ID']}_train_{key.strip()}_{iteration_number+1}.png")
         
         #save loss plot
         create_and_save_line_plot(xvect, training_dict['loss per image'], 'loss per image', 'epoch', 'loss per image', 0, np.max(training_dict['loss per image']), 
                                   f"{save_loc}/job{p['Slurm job ID']}_train_loss_per_image_{iteration_number+1}.png")
         
         for key in keys_to_plot:
             create_and_save_line_plot(xvect, testing_in_training_dict[key], key, 'epoch', f'{key} (%)', 0, 100,
                                       f"{save_loc}/job{p['Slurm job ID']}_traintest_{key.strip()}_{iteration_number+1}.png")
         
     except Exception as e:
         print_to_log_file(p['log filename'], f"An error occurred while generating or saving plots: {e}")
         
         
             
# %% Machine learning

def get_num_ml_iterations(p):
    """
    Determine number of ml iterations

    Parameters:
    p: A dictionary containing required parameters.
    """
    
    if p['task'] == 'noise_addition':
        p['total_passes'] = len(p['noise_fraction'])
        p['num_vars'] = 1
        
    elif p['task'] == 'pump_bandwidth':    
        p['total_passes'] = len(p['pump_bandwidth'])
        p['num_vars'] = 1
        
    elif p['task'] == 'pump_center':    
        p['total_passes'] = len(p['pump_center'])
        p['num_vars'] = 1
        
    elif p['task'] == 'dual_pump':    
        p['total_passes'] = len(p['pump_center']) * len(p['pump_bandwidth'])
        p['passes per variable'] = {1: len(p['pump_bandwidth']),
                                    2: len(p['pump_center'])
                                    }
        p['num_vars'] = 2
    
    print_to_log_file(p['log filename'], f"ML iterations requested: {p['total_passes']}")
    

def format_pytorch_datasets(p, class_distributions, iteration_data):
    
    print_to_log_file(p['log filename'],'Formatting data for Pytorch...')
    
    training_data = PyTorchDataset(iteration_data, p, isTrain=True)
    testing_data = PyTorchDataset(iteration_data, p, isTrain=False)
    
    total_number_spectra = len(testing_data) + len(training_data)
    print_to_log_file(p['log filename'], f'total number of spectra for training and testing = {total_number_spectra}')
    
    # Log class distribution
    training_classes = []
    for i in range(len(training_data)):
       check_imgs, check_labels, check_IDs, check_t2s = training_data[i]
       training_classes.append(check_labels.item())

    testing_classes = []
    for i in range(len(testing_data)):
       check_imgs, check_labels, check_IDs, check_t2s = testing_data[i]
       testing_classes.append(check_labels.item())
    
    class_distributions['training'] = Counter(training_classes)
    class_distributions['testing'] = Counter(testing_classes)
    
    return training_data, testing_data, class_distributions



def initialize_new_ml_iteration(p, iteration_number, iteration_inds, accuracy_df, f1_df):

    if p['num_vars'] == 1:
        
        print_to_log_file(p['log filename'],f"----------------\nIteration {iteration_number+1} of {p['total_passes']}")
        iteration_variable_name = p['scan_item_df']
        
        if p['task'] == 'ablation':
            if iteration_number == 0:
                iteration_variable_value = None
            else:
                iteration_variable_value = p[p['scan_key']][iteration_number-1]
        else:
            iteration_variable_value = p[p['scan_key']][iteration_number]
            
        iteration_results = {'scanned item': iteration_variable_name, 
                             'scanned item value': iteration_variable_value
                             }
        # Update summary dataframes
        accuracy_df.at[iteration_number + 1, iteration_variable_name] = iteration_results['scanned item value']
        f1_df.at[iteration_number + 1, iteration_variable_name] = iteration_results['scanned item value']
        
    elif p['num_vars'] == 2:
        
        print_to_log_file(p['log filename'],f"----------------\nIteration {iteration_number+1} of {p['total_passes']}")
        
        iteration_variable1_name = p['scan_item_df'][1]
        iteration_variable2_name = p['scan_item_df'][2]
        
        iteration_variable1_value = p[p['scan_key'][1]][iteration_inds[0]]
        iteration_variable2_value = p[p['scan_key'][2]][iteration_inds[1]]
        
        iteration_results = {'scanned item 1': iteration_variable1_name, 
                             'scanned item 1 value': iteration_variable1_value,
                             'scanned item 2': iteration_variable2_name, 
                             'scanned item 2 value': iteration_variable2_value
                             }
        
        # Update summary dataframes
        accuracy_df.at[iteration_number + 1, p['scan_item_df'][1]] = iteration_results['scanned item 1 value']     
        accuracy_df.at[iteration_number + 1, p['scan_item_df'][2]] = iteration_results['scanned item 2 value']
        f1_df.at[iteration_number + 1, p['scan_item_df'][1]] = iteration_results['scanned item 1 value']
        f1_df.at[iteration_number + 1, p['scan_item_df'][2]] = iteration_results['scanned item 2 value']
        
    
    return iteration_results, accuracy_df, f1_df



def ML_iterations(p, central_data, classes, check_spectrum, accuracy_df, f1_df):
    
    from custom_classes import NeuralNet
    import torch.nn as nn
    
    torch.manual_seed(p['torch_seed'])
    network_details = {'input size': p['inSize'], 'hidden size': p['hiddenSize'], 'epochs': p['num_epochs'],
                       'batch size': p['batchSize'], 'learning rate': p['lr'], 'dropout': p['p_dropout']}

    if p['num_vars'] == 1:
    
        for iteration_number in range(p['total_passes']):
            
            class_distributions = {'training': [], 
                                   'testing': []}
            
            # Initialize the results dictionary for this iteration
            iteration_results, accuracy_df, f1_df = initialize_new_ml_iteration(p = p,
                                                                                iteration_number = iteration_number,
                                                                                iteration_inds=None,
                                                                                accuracy_df = accuracy_df,
                                                                                f1_df = f1_df)
            
            iteration_data, pollute_summary = pollute_data(p = p,
                                                           central_data = central_data, 
                                                           iteration_inds=[iteration_number])                      # Generate polluted data
            
            save_checked_spectrum(p, check_spectrum, iteration_data, iteration_number)                             # Save spectrum if requested
            
            # Format datasets for Pytorch
            training_data, testing_data, class_distributions = format_pytorch_datasets(p, class_distributions, iteration_data)
            
            # ----- Set up model -------
            
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
            train_loader = DataLoader(dataset=training_data, batch_size=p['batchSize'], shuffle=True)   # dataloaders automatically shuffle and split data up into batches of requested size
            test_loader = DataLoader(dataset=testing_data, batch_size=p['batchSize'], shuffle=False)
                
            model = NeuralNet(p['inSize'], p['hiddenSize'], classes['N'], p['p_dropout']).to(device)
            
            flatten = nn.Flatten()                                                                      # convert from w1 x w3 (2D) to w1w3 (1D)
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=p['lr'])
            
            # ----- Train model -----
            
            training_dict, testing_in_training_dict = initialize_training(p, train_loader, test_loader) 
            start_time = time.perf_counter()
            model.train()                                                                               # ensure model is set to training state
            
            for epoch_index in range(p['num_epochs']):
                
                # set up dictionary with running lists (to be updated through the batches)
                training_running_dict = initialize_running_dict(p, 'training')
    
                # step through batches
                for batch_index, (images, labels, IDs, t2s) in enumerate(train_loader):
                    
                    images = flatten(images).to(device)
                    labels = labels.to(device)        
                    outputs = model(images)             # forward steps through the neurons
                    loss = criterion(outputs, labels)
                    
                    # analyze and record before proceeding
                    update_running_dict(training_running_dict, IDs, t2s, labels, outputs, stage='training', loss=loss)
                    
                    optimizer.zero_grad()               # gradients need to be zeroed out
                    loss.backward()                     # backwards steps to optimize the weights and biases
                    optimizer.step()
                    
                # log outcomes after all batches complete
                log_epoch_outcomes(epoch_index, training_dict, training_running_dict, isTrain=True)
                
                # test once per epoch
                with torch.no_grad():
                    
                    # set up dictionary with running lists (to be updated through the batches)
                    train_test_running_dict = initialize_running_dict(p, 'testing in training')
                    
                    for batch_index, (images, labels, IDs, t2s) in enumerate(test_loader):
                        
                        images = flatten(images).to(device)
                        labels = labels.to(device)
                        outputs = model(images)
                        update_running_dict(train_test_running_dict, IDs, t2s, labels, outputs, stage='testing in training')
                        
                    # log outcomes after all batches complete
                    log_epoch_outcomes(epoch_index, testing_in_training_dict, train_test_running_dict)
                
                # end epoch
                print_to_log_file(p['log filename'], f"  completed {epoch_index+1} of {p['num_epochs']} epochs")
                
            # end training
            print_to_log_file(p['log filename'], f'Time allocated to training: {np.round((time.perf_counter() - start_time))} sec')
            update_dataframes_post_training(iteration_number, accuracy_df, f1_df, 
                                                 training_dict, testing_in_training_dict)
            # save figures for loss and accuracy (if user requested)
            save_ml_reports(p, training_dict, testing_in_training_dict, iteration_number)
            
            
            # ----- Test model -----
                
            model.eval()            # set the model to evaluation mode
            with torch.no_grad():   # gradients turned off
                
                testing_running_dict = initialize_running_dict(p, 'testing')
                for batch_index, (images, labels, IDs, t2s) in enumerate(test_loader):
                    
                    images = flatten(images).to(device)
                    labels = labels.to(device)
                    outputs = model(images)
                    
                    update_running_dict(testing_running_dict, IDs, t2s, labels, outputs, stage='testing')
                
                # end testing and record results
                testing_dict = log_testing_outcomes(testing_running_dict, classes, len(test_loader))
                    
            # ----- End iteration: process and save results -----
            update_dataframes_post_testing(p, iteration_number, iteration_results, testing_dict, accuracy_df, f1_df)
            
            save_iteration_outputs(p, network_details, classes, class_distributions, 
                                        training_dict, testing_in_training_dict, testing_dict, 
                                        iteration_number, iteration_results, pollute_summary)
                
            # remove large variables before heading into next iteration  
            variables_to_delete = ['iteration_data', 'iteration_results',
                                   'training_data', 'training_dict', 'train_running_dict',
                                   'testing_data', 'testing_dict', 'testing_running_dict',
                                   'testing_in_training_dict', 'train_test_running_dict', 'pollute_summary']
            
            for var_name in variables_to_delete:
                try:
                    del var_name
                except KeyError:
                    pass  

    elif p['num_vars'] == 2:
        
        count = 0
        for pass1_ind in range(p['passes per variable'][1]):
            for pass2_ind in range(p['passes per variable'][2]):
            
                class_distributions = {'training': [], 
                                       'testing': []}
                
                # Initialize the results dictionary for this iteration
                iteration_results, accuracy_df, f1_df = initialize_new_ml_iteration(p = p, 
                                                                                    iteration_number = count,
                                                                                    iteration_inds=[pass1_ind, pass2_ind],
                                                                                    accuracy_df = accuracy_df,
                                                                                    f1_df = f1_df)
                # Generate polluted data
                iteration_data, pollute_summary = pollute_data(p = p, 
                                                               central_data = central_data, 
                                                               iteration_inds=[pass1_ind, pass2_ind])
                
                # Save spectrum if requested
                save_checked_spectrum(p, check_spectrum, iteration_data, iteration_number = count)                             
                
                # Format datasets for Pytorch
                training_data, testing_data, class_distributions = format_pytorch_datasets(p, class_distributions, iteration_data)
                
                # ----- Set up model -------
                
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
                train_loader = DataLoader(dataset=training_data, batch_size=p['batchSize'], shuffle=True)   # dataloaders automatically shuffle and split data up into batches of requested size
                test_loader = DataLoader(dataset=testing_data, batch_size=p['batchSize'], shuffle=False)
                    
                model = NeuralNet(p['inSize'], p['hiddenSize'], classes['N'], p['p_dropout']).to(device)
                
                flatten = nn.Flatten()                                                                      # convert from w1 x w3 (2D) to w1w3 (1D)
                criterion = nn.CrossEntropyLoss()
                optimizer = torch.optim.Adam(model.parameters(), lr=p['lr'])
                
                # ----- Train model -----
                
                training_dict, testing_in_training_dict = initialize_training(p, train_loader, test_loader) 
                start_time = time.perf_counter()
                model.train()                                                                               # ensure model is set to training state
                
                for epoch_index in range(p['num_epochs']):
                    
                    # set up dictionary with running lists (to be updated through the batches)
                    training_running_dict = initialize_running_dict(p, 'training')
        
                    # step through batches
                    for batch_index, (images, labels, IDs, t2s) in enumerate(train_loader):
                        
                        images = flatten(images).to(device)
                        labels = labels.to(device)        
                        outputs = model(images)             # forward steps through the neurons
                        loss = criterion(outputs, labels)
                        
                        # analyze and record before proceeding
                        update_running_dict(training_running_dict, IDs, t2s, labels, outputs, stage='training', loss=loss)
                        
                        optimizer.zero_grad()               # gradients need to be zeroed out
                        loss.backward()                     # backwards steps to optimize the weights and biases
                        optimizer.step()
                        
                    # log outcomes after all batches complete
                    log_epoch_outcomes(epoch_index, training_dict, training_running_dict, isTrain=True)
                    
                    # test once per epoch
                    with torch.no_grad():
                        
                        # set up dictionary with running lists (to be updated through the batches)
                        train_test_running_dict = initialize_running_dict(p, 'testing in training')
                        
                        for batch_index, (images, labels, IDs, t2s) in enumerate(test_loader):
                            
                            images = flatten(images).to(device)
                            labels = labels.to(device)
                            outputs = model(images)
                            update_running_dict(train_test_running_dict, IDs, t2s, labels, outputs, stage='testing in training')
                            
                        # log outcomes after all batches complete
                        log_epoch_outcomes(epoch_index, testing_in_training_dict, train_test_running_dict)
                    
                    # end epoch
                    print_to_log_file(p['log filename'], f"  completed {epoch_index+1} of {p['num_epochs']} epochs")
                    
                # end training
                print_to_log_file(p['log filename'], f'Time allocated to training: {np.round((time.perf_counter() - start_time))} sec')
                update_dataframes_post_training(iteration_number = count, 
                                                accuracy_df = accuracy_df, 
                                                f1_df = f1_df, 
                                                training_dict = training_dict, 
                                                testing_in_training_dict = testing_in_training_dict)
                
                # save figures for loss and accuracy (if user requested)
                save_ml_reports(p = p, 
                                training_dict = training_dict, 
                                testing_in_training_dict = testing_in_training_dict, 
                                iteration_number = count)
                
                # ----- Test model -----
                    
                model.eval()            # set the model to evaluation mode
                with torch.no_grad():   # gradients turned off
                    
                    testing_running_dict = initialize_running_dict(p, 'testing')
                    for batch_index, (images, labels, IDs, t2s) in enumerate(test_loader):
                        
                        images = flatten(images).to(device)
                        labels = labels.to(device)
                        outputs = model(images)
                        
                        update_running_dict(testing_running_dict, IDs, t2s, labels, outputs, stage='testing')
                    
                    # end testing and record results
                    testing_dict = log_testing_outcomes(testing_running_dict, classes, len(test_loader))
                        
                # ----- End iteration: process and save results -----
                update_dataframes_post_testing(p = p, 
                                               iteration_number = count, 
                                               iteration_results = iteration_results, 
                                               testing_dict = testing_dict, 
                                               accuracy_df = accuracy_df, 
                                               f1_df = f1_df)
                
                save_iteration_outputs(p = p, 
                                       network_details = network_details, 
                                       classes = classes, 
                                       class_distributions = class_distributions, 
                                       training_dict = training_dict, 
                                       testing_in_training_dict = testing_in_training_dict, 
                                       testing_dict = testing_dict, 
                                       iteration_number = count, 
                                       iteration_results = iteration_results, 
                                       pollute_summary = pollute_summary)
                
                remove_workspace_variables()
                count += 1
                
    return p, accuracy_df, f1_df


def pollution_iterations(p, central_data):
    '''Function to

    '''
    if p['num_vars'] == 1:
    
        for iteration_number in range(p['total_passes']):
            
            iteration_data, pollute_summary = pollute_data(p = p,
                                                           central_data = central_data, 
                                                           iteration_inds=[iteration_number])                      # Generate polluted data
 
            print_to_log_file(p['log filename'],f"----------------\nIteration {iteration_number+1} of {p['total_passes']}")
            iteration_variable_name = p['scan_item_df']
            
            iteration_variable_value = p[p['scan_key']][iteration_number]
                
            iteration_results = {'scanned item': iteration_variable_name, 
                                 'scanned item value': iteration_variable_value
                                 }
            
            log_pollution_results(p, iteration_results, pollute_summary)
                
            # remove large variables before heading into next iteration  
            variables_to_delete = ['iteration_data', 'iteration_results', 'pollute_summary']
            
            for var_name in variables_to_delete:
                try:
                    del var_name
                except KeyError:
                    pass  
                
    else:
        print_to_log_file(p['log filename'],'ERROR: Can only do SNR analysis for one scanned variable. Exiting...')
        sys.exit()


# %% Logging ML outputs


def initialize_training(p, train_loader, test_loader):
    
    print_to_log_file(p['log filename'],'TRAINING STAGE')
    
    n_batches_training = len(train_loader)
    n_batches_testing = len(test_loader)
    print_to_log_file(p['log filename'], f'{n_batches_training} batches in train loader, {n_batches_testing} in test loader...''')
    
    # dictionaries to keep track of items per epoch per batch
    training_dict = {'number of batches': n_batches_training,
                     'epoch index': [], 
                     'system IDs': [],
                     't2s': [],
                     'labels': [],
                     'predictions': [],
                     'number of images': [],
                     'number of correct predictions': [],
                     'accuracy': [],
                     'total loss': [],
                     'loss per image': [],
                     'micro f1': [], 
                     'macro f1': [],
                     'weighted f1': [],
                     'f1 raw scores': [],
                     'f1 raw labels': [],
                     'f1 raw label counts': []
                     }
    
    testing_in_training_dict = {'number of batches': n_batches_testing,
                                'epoch index': [], 
                                'system IDs': [],
                                't2s': [],
                                'labels': [],
                                'predictions': [],
                                'number of images': [],
                                'number of correct predictions': [],
                                'accuracy': [],
                                'micro f1': [], 
                                'macro f1': [],
                                'weighted f1': [],
                                'f1 raw scores': [],
                                'f1 raw labels': [],
                                'f1 raw label counts': []
                                }

    return training_dict, testing_in_training_dict


def initialize_running_dict(p, stage):

    if stage == 'training':
        running_dict = {'number of images': [], 
                        'number of correct predictions': [],
                        'system IDs': [],
                        't2s': [],
                        'labels': [], 
                        'predictions': [], 
                        'loss': [],
                        }
        
    elif stage == 'testing in training':
        running_dict = {'number of images': [], 
                        'number of correct predictions': [],
                        'system IDs': [],
                        't2s': [],
                        'labels': [], 
                        'predictions': [], 
                        }
        
    elif stage == 'testing':
        running_dict = {'number of images': [], 
                        'number of correct predictions': [],
                        'system IDs': [],
                        't2s': [],
                        'labels': [], 
                        'predictions': [], 
                        'predictions with probabilities': []
                        }   
        
    else:
        print_to_log_file(p['log filename'],"ERROR: unsupported stage requested. Exiting.")
        raise Exception("ERROR: Unsupported stage requested.")
        sys.exit()
    
    return running_dict


def update_running_dict(running_dict, IDs, t2s, labels, outputs, stage, loss=None):
    '''
    Update running analytics per batch

    '''

    _, predictions = torch.max(outputs, 1)
    sm = torch.nn.Softmax(dim=1)
    probabilities = sm(outputs).cpu().detach().numpy()  # Convert to probabilities
    
    number_of_images = labels.shape[0]
    number_correct = (predictions == labels.view_as(predictions)).sum().item()
    
    running_dict['number of images'].append(number_of_images)
    running_dict['number of correct predictions'].append(number_correct)
    running_dict['system IDs'].extend(IDs.squeeze().cpu().numpy())
    running_dict['t2s'].extend(t2s.squeeze().cpu().numpy())
    running_dict['labels'].extend(labels.cpu().numpy())
    running_dict['predictions'].extend(predictions.cpu().numpy())
    
    if stage == 'training':
        running_dict['loss'].append(loss.item())
        
    elif stage == 'testing':
        running_dict['predictions with probabilities'].append(probabilities)     # Accumulate predictions and labels
        


def log_epoch_outcomes(epoch_index, stage_dict, running_dict, isTrain=False):
    '''
    Log training outcomes at the end of each epoch
    '''
    
    n_images_per_epoch = np.sum(running_dict['number of images'])
    n_correct_per_epoch = np.sum(running_dict['number of correct predictions'])
    
    stage_dict['epoch index'].append(epoch_index)
    stage_dict['number of images'].append(n_images_per_epoch)
    stage_dict['system IDs'].append(running_dict['system IDs'])
    stage_dict['t2s'].append(running_dict['t2s'])
    stage_dict['labels'].append(running_dict['labels'])
    stage_dict['predictions'].append(running_dict['predictions'])
    
    stage_dict['number of correct predictions'].append(n_correct_per_epoch)
    stage_dict['accuracy'].append(100 * n_correct_per_epoch / n_images_per_epoch)

    # calculate f1 scores at the end of each epoch
    f1_scores = f1_score_custom(running_dict['labels'], 
                                running_dict['predictions'], 
                                average_methods_list=['micro', 'macro', 'weighted', None]
                                )

    stage_dict['micro f1'].append(f1_scores['micro'])   
    stage_dict['macro f1'].append(f1_scores['macro'])   
    stage_dict['weighted f1'].append(f1_scores['weighted'])    
    stage_dict['f1 raw scores'].append(f1_scores[None][0])
    stage_dict['f1 raw labels'].append(f1_scores[None][1])
    stage_dict['f1 raw label counts'].append(f1_scores[None][2])
    # stage_dict['batch-level details'].append(running_dict)
    
    if isTrain:
        stage_dict['total loss'].append(np.sum(running_dict['loss']))
        stage_dict['loss per image'].append(np.sum(running_dict['loss'])/n_images_per_epoch)
        
        
def log_testing_outcomes(running_dict, classes, n_batches):
    '''
    Log testing outcomes after all batches have been iterated
    '''
    
    n_images = np.sum(running_dict['number of images'])
    n_correct = np.sum(running_dict['number of correct predictions'])
    
    results_dict = {'number of batches': n_batches,
                    'number of images': n_images,
                    'system IDs': running_dict['system IDs'],
                    't2s': running_dict['t2s'],
                    'labels': running_dict['labels'],
                    'predictions': running_dict['predictions'],
                    'number of correct predictions': n_correct,
                    'accuracy': 100 * n_correct / n_images
                    }
    
    all_probabilities = np.concatenate(running_dict['predictions with probabilities'], axis=0)  # Shape: (total_samples, n_classes)
    results_dict['predictions with probabilities'] = all_probabilities
    results_dict['top2 accuracy'] = 100 * top_k_accuracy_score(np.array(running_dict['labels']), all_probabilities, k=2, labels=classes['numbers'], normalize=True)
    
    # calculate f1 scores at the end of each epoch
    f1_scores = f1_score_custom(running_dict['labels'], 
                                running_dict['predictions'], 
                                average_methods_list=['micro', 'macro', 'weighted', None]
                                )

    results_dict.update({'micro f1': f1_scores['micro'],
                         'macro f1': f1_scores['macro'],
                         'weighted f1': f1_scores['weighted'],   
                         'f1 raw scores': f1_scores[None][0],
                         'f1 raw labels': f1_scores[None][1],
                         'f1 raw label counts': f1_scores[None][2]
                         })
    
    return results_dict


def save_iteration_outputs(p, network_details, classes, class_distributions, 
                           training_dict, testing_in_training_dict, testing_dict, 
                           iteration_number, iteration_results, pollute_summary):
  
    if p['save_ML_output'] == 'True':
        
        print_to_log_file(p['log filename'], f'Saving results for iteration {iteration_number+1}...')
        
        iteration_results.update({'network details': network_details, 'classes': classes, 'class distributions': class_distributions,
                                  'TRAINING details': training_dict, 'TESTING IN TRAINING details': testing_in_training_dict, 'TESTING details': testing_dict,
                                  'pollution summary': pollute_summary})
        
        with open(f"{p['outputs_path']}/job{p['Slurm job ID']}_ITERATION{iteration_number+1}.pkl", 'wb') as f:
            pickle.dump(iteration_results, f)      
        
        print_to_log_file(p['log filename'],'Results saved...\n')
        
        
def log_pollution_results(p, iteration_number, iteration_results, pollute_summary):
        
    print_to_log_file(p['log filename'], f'Saving results for iteration {iteration_number+1}...')
    iteration_results.update({'pollution summary': pollute_summary})
    
    with open(f"{p['outputs_path']}/job{p['Slurm job ID']}_ITERATION{iteration_number+1}.pkl", 'wb') as f:
        pickle.dump(iteration_results, f)      
    
    print_to_log_file(p['log filename'],'Results saved...\n')

    
def update_dataframes_post_testing(p, iteration_number, iteration_results, testing_dict, accuracy_df, f1_df):
    
    if p['num_vars'] == 1:
        print_to_log_file(p['log filename'], f'''~~~ RESULTS ~~~
{p['scan_item_df']} = {iteration_results['scanned item value']}''')

    elif p['num_vars'] == 2:
        print_to_log_file(p['log filename'], f'''~~~ RESULTS ~~~
{p['scan_item_df'][1]} = {iteration_results['scanned item 1 value']}
{p['scan_item_df'][2]} = {iteration_results['scanned item 2 value']}''')
     
    print_to_log_file(p['log filename'], f'''
 correct predictions = {testing_dict['number of correct predictions']}
total spectra tested = {testing_dict['number of images']}
            accuracy = {np.round(testing_dict['accuracy'], decimals=2)}
            micro f1 = {np.round(testing_dict['micro f1'], decimals=2)}
            macro f1 = {np.round(testing_dict['macro f1'], decimals=2)}
         weighted f1 = {np.round(testing_dict['weighted f1'], decimals=2)}
                top2 = {np.round(testing_dict['top2 accuracy'], decimals=2)}''')

    accuracy_df.at[iteration_number+1, 'Test accuracy'] = np.round(testing_dict['accuracy'], decimals=3)
    accuracy_df.at[iteration_number+1, 'Test top2 accuracy'] = np.round(testing_dict['top2 accuracy'], decimals=3)
    f1_df.at[iteration_number+1, 'Test micro f1'] = np.round(testing_dict['micro f1'], decimals=3)
    f1_df.at[iteration_number+1, 'Test macro f1'] = np.round(testing_dict['macro f1'], decimals=3)
    f1_df.at[iteration_number+1, 'Test weighted f1'] = np.round(testing_dict['weighted f1'], decimals=3)


def update_dataframes_post_training(iteration_number, accuracy_df, f1_df, training_dict, testing_in_training_dict):
    
    accuracy_df.at[iteration_number+1, 'Train accuracy'] = training_dict['accuracy'][-1]
    f1_df.at[iteration_number+1, 'Train micro f1'] = training_dict['micro f1'][-1]
    f1_df.at[iteration_number+1, 'Train macro f1'] = training_dict['macro f1'][-1]
    f1_df.at[iteration_number+1, 'Train weighted f1'] = training_dict['weighted f1'][-1]
    
    accuracy_df.at[iteration_number+1, 'Train-test accuracy'] = testing_in_training_dict['accuracy'][-1]
    f1_df.at[iteration_number+1, 'Train-test micro f1'] = testing_in_training_dict['micro f1'][-1]
    f1_df.at[iteration_number+1, 'Train-test macro f1'] = testing_in_training_dict['macro f1'][-1]
    f1_df.at[iteration_number+1, 'Train-test weighted f1'] = testing_in_training_dict['weighted f1'][-1]
    

# %% Debugging

def job_monitor_with_slurm(p):
    try:
        # Run the top command and capture the output
        top_output = subprocess.check_output(["top","-bn1"], text=True)
        
        # Split the output into lines
        lines = top_output.strip().split('\n')
        
        # Parse the output and create a DataFrame
        data = [line.split() for line in lines[7:]]  # Skip the header lines
        columns = ["PID", "USER", "PR", "NI", "VIRT", "RES", "SHR", "S", "%CPU", "%MEM", "TIME+", "COMMAND", ""]
        df = pd.DataFrame(data, columns=columns)
        
        # Iterate through each cell in the DataFrame
        for index, row in df.iterrows():
                # Check if the cell contains the sequence
            if 'python' in row["COMMAND"]:
                cpu_percent = df.loc[index, '%CPU']
                mem_percent = df.loc[index, '%MEM']
                    
        print_to_log_file(p['log filename'],f'''
JOB PERFORMANCE STATS
CPU usage: {cpu_percent} %
memory usage: {mem_percent} %
''')
    
    except:
        print_to_log_file(p['log filename'],"  Unable to obtain job stats... Moving on.")
        
def print_variable_info(logfilename, variable):

    class_name = type(variable).__name__
    print_to_log_file(logfilename,f"Class: {class_name}")

    if isinstance(variable, np.ndarray):
        print_to_log_file(logfilename,f"Size: {variable.size}\nShape: {variable.shape}\nDimensions: {variable.ndim}\nData Type: {variable.dtype}")
        
    elif isinstance(variable, float) or isinstance(variable, int):
        print_to_log_file(logfilename,f"Type: {type(variable)}\nValue: {variable}")
        
    elif isinstance(variable, list):
        print_to_log_file(logfilename,f"Length: {len(variable)}\nDimensions: {get_list_dimensions(variable)}")
        
    elif isinstance(variable, torch.Tensor):
        print_to_log_file(logfilename,f"Size: {variable.size()}\nShape: {variable.shape}\nData Type: {variable.dtype}")
        
    elif isinstance(variable, DataLoader):
        print_to_log_file(logfilename,f"Number of batches: {len(variable)}")
        
        # Check if the DataLoader is iterable to get the batch size
        try:
            batch_size = next(iter(variable))[0].size(0)  # Assuming the batch contains tensors
            print_to_log_file(logfilename,f"Batch size: {batch_size}")
        except StopIteration:
            print_to_log_file(logfilename,"Unable to determine batch size")
            
    else:
        try:
            print_to_log_file(logfilename,f"Data Type: {variable.dtype}")
        except:
            print_to_log_file(logfilename,"Error in assessing variable")

def get_list_dimensions(lst):
    if not isinstance(lst, list):
        return 0
    return 1 + max(get_list_dimensions(item) for item in lst)

def print_all_keys(logfilename, dictionary):
    print_to_log_file(logfilename,'-------------------------\nDictionary information:\n')
    for key, value in dictionary.items():
        print_to_log_file(logfilename, f'\nkey: {key}')
        print_variable_info(logfilename, value)
        
    print_to_log_file(logfilename,'--------------------------------')
    
        # ~~~~~~~~ RANDOMIZED ORGANIZATIONAL CHECK ~~~~~~~~~~~
        
#         print_to_log_file(f'{JOBNAME}.log', 'Initiating random spectrum check...')
#         random_batch_index = random.randint(0, len(Training_data)/batchSize-1)
#         random_index_in_batch = random.randint(0, batchSize-1)
        
#         for i, batch in enumerate(train_loader):
#             if i == random_batch_index:
#                 image, label, ID, t2 = batch[0][random_index_in_batch], batch[1][random_index_in_batch], batch[2][random_index_in_batch], batch[3][random_index_in_batch]
#                 break

#         image = image.cpu().detach().numpy()
#         label = label.cpu().detach().numpy()
#         ID = ID.cpu().detach().numpy()
#         t2 = t2.cpu().detach().numpy().item()
        
#         image_check_result, label_check_result = check_spectrum_against_central(central_data, image, label, ID, t2)
#         found_ID, found_t2, found_error = find_image_in_central_data(central_data, image)
#         print_to_log_file(f'{JOBNAME}.log', f'''
# random image selection:
#     ID: {ID}
#     t2: {t2}
                          
# image check: {image_check_result}
# label check: {label_check_result}

# results of image search:
#     ID: {found_ID}
#     t2: {found_t2}
#     mean absolute error:  {found_error}''')       
        
#         if image_check_result == True and label_check_result == True:
#             print_to_log_file(f'{JOBNAME}.log', 'PASSED!')
#         else:
#             print_to_log_file(f'{JOBNAME}.log', 'ERROR: Random spectrum check FAILED. Now exiting.')        
#             sys.exit()

# %% Image operations

def are_images_equal(image1, image2, epsilon=1e-6):
    diff = np.abs(image1 - image2)
    return np.all(diff < epsilon)

def check_spectrum_against_central(central_data, spectrum, classification, ID, t2):
    '''
    Parameters
    ----------
    spectrum : 2D spectrum sent from data loader
    classification : associated class sent from data loader
    ID : associated run ID sent from data loater
    t2 : associated waiting time sent from data loader

    Returns
    -------
    bool : data loader spectrum = central dataset spectrum? True/False
    bool : data loader classification = central dataset classification? True/False
    '''
   
    sys_index = np.where(central_data['system ID numbers'] == ID)
    sys_index = sys_index[0].item()
    t2_index = np.where(central_data['t2'][sys_index] == t2)
    t2_index = t2_index[0].item()
    
    central_class = central_data['classes'][sys_index] 
    sys_spectra = central_data['spectra'][sys_index,:,:,:]
    central_spectrum = sys_spectra[:,:,t2_index].squeeze()
    
    imagesc(central_spectrum,'Central', True)
    imagesc(spectrum,'DataLoader', True)

    if are_images_equal(central_spectrum, spectrum, epsilon=1e-2):
        if central_class == classification:
            return True, True
        else:
            return True, False
    else:
        if central_class == classification:
            return False, True
        else:
            return False, False

def find_image_in_central_data(central_data, image):

    ID_of_image = None
    t2_of_image = None
    error_val = None
    for i in range(central_data['Number of systems']):
        for j in range(central_data['nt2s']):
            img_temp = central_data['spectra'][i,:,:,j].squeeze()
            if are_images_equal(img_temp, image, epsilon=1e-6):
                ID_of_image = central_data['System IDs'][i]
                t2_of_image = central_data['t2'][i][j]
                error_val = mean_absolute_error(img_temp, image)
                break
            
    return ID_of_image, t2_of_image, error_val

def find_image_index(central_data, ID):
    '''
    Parameters
    ----------
    '''
    sys_index = np.where(central_data['system ID numbers'] == ID)[0]
    if len(sys_index) == 0:
        sys_index = None
    else:
        sys_index = sys_index.item()
    
    return sys_index

def mean_absolute_error(image1, image2):
    return np.mean(np.abs(image1 - image2))

# %% Empty



