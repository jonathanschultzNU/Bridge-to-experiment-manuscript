"""
Module containing functions for data preprocessing and augmentation, including noise addition and self-normalization.
"""

from copy import deepcopy
import numpy as np
from utils import print_to_log_file
import sys


def add_noise(data_dict: dict, logfilename: str, method: str, noise_fraction: float, SNR_filter: bool, threshold: float) -> tuple:
    """
    Adds noise to the dataset based on the specified method.
    
    Parameters:
        data_dict (dict): The dataset dictionary.
        logfilename (str): Path to the log file.
        method (str): Noise method ('additive' or 'intensity-dependent').
        noise_fraction (float): Fraction of noise to be added.
        SNR_filter (bool): Whether to filter low SNR spectra.
        threshold (float): SNR threshold for filtering.
    
    Returns:
        tuple: (Updated dataset, SNR summary dictionary)
    """
    
    # generate arrays to log widths of clean spectra and noise
    data_dict['noise width'] = np.zeros((data_dict['Number of systems'], data_dict['nt2']))
    data_dict['signal width'] = np.zeros((data_dict['Number of systems'], data_dict['nt2']))
    
    # SeedVec = np.arange(start = startSeed, stop = startSeed + data_dict['Number of systems'], step = 1, dtype = int)
    SeedVec = deepcopy(data_dict['system ID numbers'])
    
    if method not in ['max_scaling', 'profile_scaling']:
        print_to_log_file(logfilename, " ERROR: invalid noise method specified. Now exiting...")
        sys.exit()
    
    
    if method == 'additive':
        # signal + max(signal) * noise
            
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
    
    
    elif method == 'intensity-dependent': 
        # signal + signal * noise
        
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
    
    if not SNR_filter:
        return data_dict, {}
        
    # SNR Filtering
    SNR_summary = {
        'per system': {
            'System IDs': [],
            'SNR at t2 = 0': [],
            'SNR at t2 = end': [],
            'avg SNR of all t2': [],
            'avg SNR of kept t2': [],
            'avg SNR of dropped t2': [],
        },
        'dropped images': {},
    }
    
    spec_count, drop_count, avg_drop_SNR, avg_keep_SNR, avg_total_SNR = 0, 0, 0, 0, 0
    data_dict['dropped images'] = {}
    
    for i, system_ID in enumerate(data_dict['system ID numbers']):
        
        data_dict['dropped images'][system_ID] = np.ones(data_dict['nt2'], dtype=bool)
        drop_count_perSys, avg_keep_SNR_perSys, avg_drop_SNR_perSys, avg_total_SNR_perSys = 0, 0, 0, 0
        
        SNR_summary['per system']['System IDs'].append(system_ID.copy())
        
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
                data_dict['dropped images'][system_ID][j] = False
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
 
spectra dropped (low SNR): {drop_count} of {spec_count} ({100*drop_count/spec_count:.2f} %)
Average SNR of dropped spectra: {avg_drop_SNR/drop_count}
Average SNR of kept spectra: {avg_keep_SNR/(spec_count-drop_count)}
Average SNR of all spectra: {avg_total_SNR/spec_count:.2f}\n''')

    else:
        print_to_log_file(logfilename, f'''\n
No spectra were dropped due to low SNR. Average SNR of all spectra: {avg_total_SNR/spec_count:.2f}\n''')
   
    return data_dict, SNR_summary


def apply_pump(data_dict: dict, carrier: float, bandwidth: float) -> dict:
    """
    Applies a pump modulation to the dataset.
    
    Parameters:
        data_dict (dict): The dataset dictionary.
        carrier (float): Center frequency of the pump.
        bandwidth (float): Spectral bandwidth of the pump.
    
    Returns:
        dict: Modified dataset with applied pump effects.
    """
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


def pollute_data(p: dict, central_data: dict, iteration_inds: list) -> tuple:
    """
    Applies data modifications such as noise addition or pump effects.
    
    Parameters:
        p (dict): Configuration parameters.
        central_data (dict): The central dataset.
        iteration_inds (list): Indices for the iteration process.
    
    Returns:
        tuple: (Modified dataset, Summary of modifications applied)
    """
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
    
    elif p['num_vars'] == 2:
        
        if p['task'] == 'dual_pump':
            data = apply_pump(data, p['pump_center'][iteration_inds[1]], p['pump_bandwidth'][iteration_inds[0]])
            return data, None
        


def self_normalize_datasets(data_dict: dict, logfilename: str) -> dict:
    """
    Normalizes each dataset in place to ensure consistent maximum intensity.
    
    Parameters:
        data_dict (dict): The dataset dictionary.
        logfilename (str): Path to log file for tracking normalization.
    
    Returns:
        dict: Normalized dataset.
    """
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

