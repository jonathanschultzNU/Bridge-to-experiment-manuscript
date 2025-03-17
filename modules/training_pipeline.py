"""
Module to manage the training and testing of the ML model, including dataset formatting and performance evaluation.
"""

import pandas as pd
import torch
import time
import numpy as np
from torch.utils.data import DataLoader
from modules.data_augmentation import pollute_data
from modules.data_processing import format_pytorch_datasets
from modules.visualization import save_checked_spectrum
from modules.utils import print_to_log_file, remove_workspace_variables
from modules.results_reporting import save_training_reports, save_iteration_outputs, initialize_training, log_epoch_outcomes, log_testing_outcomes, update_running_dict, update_dataframes_post_training, update_dataframes_post_testing, initialize_running_dict

# . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 

def get_num_ml_iterations(p, dirs):
    """
    Determine number of ml iterations

    Parameters:
    p: A dictionary containing required parameters.
    """
    
    if p['task'] == 'noise':
        p['total passes'] = len(p['noise fraction'])
        p['num_vars'] = 1
        
    elif p['task'] == 'bandwidth':    
        p['total passes'] = len(p['bandwidth'])
        p['num_vars'] = 1
        
    elif p['task'] == 'center frequency':    
        p['total passes'] = len(p['center frequency'])
        p['num_vars'] = 1
        
    elif p['task'] == 'bandwidth and center frequency':    
        p['total passes'] = len(p['bandwidth']) * len(p['center frequency'])
        p['passes per variable'] = {'bandwidth': len(p['bandwidth']),
                                    'center frequency': len(p['center frequency'])
                                    }
        p['num_vars'] = 2
    
    print_to_log_file(dirs['log file'], f"ML iterations requested: {p['total passes']}")
    

# . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 

def initialize_new_ml_iteration(p: dict, dirs: dict, iteration_number: int, iteration_inds: list, accuracy_df: pd.DataFrame, f1_df: pd.DataFrame) -> tuple:
    """
    Initializes a new ML iteration by setting up the required parameters.
    
    Parameters:
        p (dict): Configuration parameters.
        iteration_number (int): Current iteration number.
        iteration_inds (list, optional): Indices used for multi-variable iterations.
        accuracy_df (pd.DataFrame): Dataframe for accuracy logging.
        f1_df (pd.DataFrame): Dataframe for F1-score logging.
    
    Returns:
        tuple: (Iteration results dictionary, updated accuracy_df, updated f1_df)
    """
    if p['num_vars'] == 1:
        
        print_to_log_file(dirs['log file'],f"----------------\nIteration {iteration_number+1} of {p['total passes']}")
        iteration_variable_name = p['scan key']
        iteration_variable_value = p[p['scan key']][iteration_number]
            
        iteration_results = {'scanned item': iteration_variable_name, 
                             'scanned item value': iteration_variable_value
                             }
        # Update summary dataframes
        accuracy_df.at[iteration_number + 1, iteration_variable_name] = iteration_results['scanned item value']
        f1_df.at[iteration_number + 1, iteration_variable_name] = iteration_results['scanned item value']
        
    elif p['num_vars'] == 2:
        
        print_to_log_file(dirs['log file'],f"----------------\nIteration {iteration_number+1} of {p['total passes']}")
        
        iteration_variable1_name = p['scan key']['1']
        iteration_variable2_name = p['scan key']['2']
        
        iteration_variable1_value = p[p['scan key']['1']][iteration_inds[0]]
        iteration_variable2_value = p[p['scan key']['2']][iteration_inds[1]]
        
        iteration_results = {'scanned item 1': iteration_variable1_name, 
                             'scanned item 1 value': iteration_variable1_value,
                             'scanned item 2': iteration_variable2_name, 
                             'scanned item 2 value': iteration_variable2_value
                             }
        
        # Update summary dataframes
        accuracy_df.at[iteration_number + 1, p['scan key']['1']] = iteration_results['scanned item 1 value']     
        accuracy_df.at[iteration_number + 1, p['scan key']['2']] = iteration_results['scanned item 2 value']
        f1_df.at[iteration_number + 1, p['scan key']['1']] = iteration_results['scanned item 1 value']
        f1_df.at[iteration_number + 1, p['scan key']['2']] = iteration_results['scanned item 2 value']
        
    
    return iteration_results, accuracy_df, f1_df


# . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 

def perform_ml_iterations(p: dict, dirs: dict, central_data: dict, classes: dict, check_spectrum: dict, accuracy_df: pd.DataFrame, f1_df: pd.DataFrame) -> tuple:
    """
    Runs multiple machine learning iterations with different configurations.
    
    Parameters:
        p (dict): Configuration parameters.
        central_data (dict): Central dataset.
        classes (dict): Class labels and boundaries.
        check_spectrum (dict): Example spectrum for debugging.
        accuracy_df (pd.DataFrame): Dataframe to store accuracy results.
        f1_df (pd.DataFrame): Dataframe to store F1-score results.
    
    Returns:
        tuple: (Updated parameters, accuracy dataframe, F1-score dataframe)
    """
    from modules.ml_model import NeuralNet
    import torch.nn as nn
    
    torch.manual_seed(p['torch seed'])
    network_details = {'input size': p['input size'], 'hidden size': p['hidden layer size'],
                       'epochs': p['number of epochs'],'batch size': p['batch size'], 
                       'learning rate': p['learning rate'], 'dropout': p['dropout probability']}
    
    # variables to remove at the end of each iteration  
    variables_to_delete = ['iteration_data', 'iteration_results',
                           'training_data', 'training_dict', 'train_running_dict',
                           'testing_data', 'testing_dict', 'testing_running_dict',
                           'testing_in_training_dict', 'train_test_running_dict', 'pollute_summary']

    if p['num_vars'] == 1:
    
        for iteration_number in range(p['total passes']):
            
            class_distributions = {'training': [], 
                                   'testing': []}
            
            # Initialize the results dictionary for this iteration
            iteration_results, accuracy_df, f1_df = initialize_new_ml_iteration(p = p,
                                                                                dirs = dirs,
                                                                                iteration_number = iteration_number,
                                                                                iteration_inds=None,
                                                                                accuracy_df = accuracy_df,
                                                                                f1_df = f1_df)
            
            # Generate polluted data
            iteration_data, pollute_summary = pollute_data(p = p,
                                                           dirs = dirs,
                                                           central_data = central_data, 
                                                           iteration_inds=[iteration_number])
            
            # Save spectrum if requested
            save_checked_spectrum(p, dirs, check_spectrum, iteration_data, iteration_number)
            
            # Format datasets for Pytorch
            training_data, testing_data, class_distributions = format_pytorch_datasets(p, dirs, class_distributions, iteration_data)
            
            # Set up model
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            train_loader = DataLoader(dataset=training_data, batch_size=p['batch size'], shuffle=True)
            test_loader = DataLoader(dataset=testing_data, batch_size=p['batch size'], shuffle=False)
            model = NeuralNet(p['input size'], p['hidden layer size'], classes['N'], p['dropout probability']).to(device)
            
            flatten = nn.Flatten()  # convert from w1 x w3 (2D) to w1w3 (1D)
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=p['learning rate'])
            
            # ----- Train model -----
            
            training_dict, testing_in_training_dict = initialize_training(p, dirs, train_loader, test_loader) 
            start_time = time.perf_counter()
            model.train()                                                                               # ensure model is set to training state
            
            for epoch_index in range(p['number of epochs']):
                
                # set up dictionary with running lists (to be updated through the batches)
                training_running_dict = initialize_running_dict(p, dirs, 'training')
    
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
                    train_test_running_dict = initialize_running_dict(p, dirs, 'testing in training')
                    
                    for batch_index, (images, labels, IDs, t2s) in enumerate(test_loader):
                        
                        images = flatten(images).to(device)
                        labels = labels.to(device)
                        outputs = model(images)
                        update_running_dict(train_test_running_dict, IDs, t2s, labels, outputs, stage='testing in training')
                        
                    # log outcomes after all batches complete
                    log_epoch_outcomes(epoch_index, testing_in_training_dict, train_test_running_dict)
                
                # end epoch
                print_to_log_file(dirs['log file'], f"  completed {epoch_index+1} of {p['number of epochs']} epochs")
                
            # end training
            print_to_log_file(dirs['log file'], f'Time allocated to training: {np.round((time.perf_counter() - start_time))} sec')
            update_dataframes_post_training(iteration_number, accuracy_df, f1_df, 
                                                 training_dict, testing_in_training_dict)
            
            # save figures for loss and accuracy (if user requested)
            save_training_reports(p, dirs, training_dict, testing_in_training_dict, iteration_number)
            
            # ----- Test model -----
                
            model.eval()            # set the model to evaluation mode
            with torch.no_grad():   # gradients turned off
                
                testing_running_dict = initialize_running_dict(p, dirs, 'testing')
                for batch_index, (images, labels, IDs, t2s) in enumerate(test_loader):
                    
                    images = flatten(images).to(device)
                    labels = labels.to(device)
                    outputs = model(images)
                    
                    update_running_dict(testing_running_dict, IDs, t2s, labels, outputs, stage='testing')
                
                # end testing and record results
                testing_dict = log_testing_outcomes(testing_running_dict, classes, len(test_loader))
                    
            # ----- End iteration: process and save results -----
            update_dataframes_post_testing(p = p, 
                                           dirs = dirs, 
                                           iteration_number = iteration_number, 
                                           iteration_results = iteration_results, 
                                           testing_dict = testing_dict, 
                                           accuracy_df = accuracy_df, 
                                           f1_df = f1_df)
            
            save_iteration_outputs(p = p, 
                                   dirs = dirs,
                                   network_details = network_details, 
                                   classes = classes, 
                                   class_distributions = class_distributions, 
                                   training_dict = training_dict, 
                                   testing_in_training_dict = testing_in_training_dict, 
                                   testing_dict = testing_dict, 
                                   iteration_number = iteration_number, 
                                   iteration_results = iteration_results, 
                                   pollute_summary = pollute_summary)
                
            remove_workspace_variables(variables_to_delete)

    elif p['num_vars'] == 2:
        
        count = 0
        for pass_index_bandwidth in range(p['passes per variable']['bandwidth']):
            for pass_index_center_frequency in range(p['passes per variable']['center frequency']):
            
                class_distributions = {'training': [], 
                                       'testing': []}
                
                # Initialize the results dictionary for this iteration
                iteration_results, accuracy_df, f1_df = initialize_new_ml_iteration(p = p, 
                                                                                    dirs = dirs,
                                                                                    iteration_number = count,
                                                                                    iteration_inds=[pass_index_bandwidth, pass_index_center_frequency],
                                                                                    accuracy_df = accuracy_df,
                                                                                    f1_df = f1_df)
                # Generate polluted data
                iteration_data, pollute_summary = pollute_data(p = p, 
                                                               dirs = dirs,
                                                               central_data = central_data, 
                                                               iteration_inds=[pass_index_bandwidth, pass_index_center_frequency])
                
                # Save spectrum if requested
                save_checked_spectrum(p, dirs, check_spectrum, iteration_data, iteration_number = count)                             
                
                # Format datasets for Pytorch
                training_data, testing_data, class_distributions = format_pytorch_datasets(p, dirs, class_distributions, iteration_data)
                
                # ----- Set up model -------
                
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
                train_loader = DataLoader(dataset=training_data, batch_size=p['batch size'], shuffle=True)   # dataloaders automatically shuffle and split data up into batches of requested size
                test_loader = DataLoader(dataset=testing_data, batch_size=p['batch size'], shuffle=False)
                    
                model = NeuralNet(p['input size'], p['hidden layer size'], classes['N'], p['dropout probability']).to(device)
                
                flatten = nn.Flatten()                                                                      # convert from w1 x w3 (2D) to w1w3 (1D)
                criterion = nn.CrossEntropyLoss()
                optimizer = torch.optim.Adam(model.parameters(), lr=p['learning rate'])
                
                # ----- Train model -----
                
                training_dict, testing_in_training_dict = initialize_training(p, dirs, train_loader, test_loader) 
                start_time = time.perf_counter()
                model.train()                                                                               # ensure model is set to training state
                
                for epoch_index in range(p['number of epochs']):
                    
                    # set up dictionary with running lists (to be updated through the batches)
                    training_running_dict = initialize_running_dict(p, dirs, 'training')
        
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
                        train_test_running_dict = initialize_running_dict(p, dirs, 'testing in training')
                        
                        for batch_index, (images, labels, IDs, t2s) in enumerate(test_loader):
                            
                            images = flatten(images).to(device)
                            labels = labels.to(device)
                            outputs = model(images)
                            update_running_dict(train_test_running_dict, IDs, t2s, labels, outputs, stage='testing in training')
                            
                        # log outcomes after all batches complete
                        log_epoch_outcomes(epoch_index, testing_in_training_dict, train_test_running_dict)
                    
                    # end epoch
                    print_to_log_file(dirs['log file'], f"  completed {epoch_index+1} of {p['number of epochs']} epochs")
                    
                # end training
                print_to_log_file(dirs['log file'], f'Time allocated to training: {np.round((time.perf_counter() - start_time))} sec')
                update_dataframes_post_training(iteration_number = count, 
                                                accuracy_df = accuracy_df, 
                                                f1_df = f1_df, 
                                                training_dict = training_dict, 
                                                testing_in_training_dict = testing_in_training_dict)
                
                # save figures for loss and accuracy (if user requested)
                save_training_reports(p = p, 
                                      dirs = dirs,
                                      training_dict = training_dict, 
                                      testing_in_training_dict = testing_in_training_dict, 
                                      iteration_number = count)
                
                # ----- Test model -----
                    
                model.eval()            # set the model to evaluation mode
                with torch.no_grad():   # gradients turned off
                    
                    testing_running_dict = initialize_running_dict(p, dirs, 'testing')
                    for batch_index, (images, labels, IDs, t2s) in enumerate(test_loader):
                        
                        images = flatten(images).to(device)
                        labels = labels.to(device)
                        outputs = model(images)
                        
                        update_running_dict(testing_running_dict, IDs, t2s, labels, outputs, stage='testing')
                    
                    # end testing and record results
                    testing_dict = log_testing_outcomes(testing_running_dict, classes, len(test_loader))
                        
                # ----- End iteration: process and save results -----
                update_dataframes_post_testing(p = p,
                                               dirs = dirs,
                                               iteration_number = count, 
                                               iteration_results = iteration_results, 
                                               testing_dict = testing_dict, 
                                               accuracy_df = accuracy_df, 
                                               f1_df = f1_df)
                
                save_iteration_outputs(p = p, 
                                       dirs = dirs,
                                       network_details = network_details, 
                                       classes = classes, 
                                       class_distributions = class_distributions, 
                                       training_dict = training_dict, 
                                       testing_in_training_dict = testing_in_training_dict, 
                                       testing_dict = testing_dict, 
                                       iteration_number = count, 
                                       iteration_results = iteration_results, 
                                       pollute_summary = pollute_summary)
                
                remove_workspace_variables(variables_to_delete)
                count += 1
                
    return p, accuracy_df, f1_df