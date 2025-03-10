"""
Module to manage the training and testing of the ML model, including dataset formatting and performance evaluation.
"""

import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import top_k_accuracy_score
from collections import Counter
import time
import numpy as np
import pickle
import sys
from data_augmentation import pollute_data
from ml_model import PyTorchDataset, f1_score_custom
from visualization import save_checked_spectrum, create_and_save_line_plot
from utils import print_to_log_file, remove_workspace_variables

# . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 

def get_num_ml_iterations(p, dirs):
    """
    Determine number of ml iterations

    Parameters:
    p: A dictionary containing required parameters.
    """
    
    if p['task'] == 'noise_addition':
        p['total passes'] = len(p['noise_fraction'])
        p['num_vars'] = 1
        
    elif p['task'] == 'pump_bandwidth':    
        p['total passes'] = len(p['pump_bandwidth'])
        p['num_vars'] = 1
        
    elif p['task'] == 'pump_center':    
        p['total passes'] = len(p['pump_center'])
        p['num_vars'] = 1
        
    elif p['task'] == 'dual_pump':    
        p['total passes'] = len(p['pump_center']) * len(p['pump_bandwidth'])
        p['passes per variable'] = {1: len(p['pump_bandwidth']),
                                    2: len(p['pump_center'])
                                    }
        p['num_vars'] = 2
    
    print_to_log_file(dirs['log file'], f"ML iterations requested: {p['total passes']}")
    
# . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 

def ML_iterations(p: dict, dirs: dict, central_data: dict, classes: dict, check_spectrum: dict, accuracy_df: pd.DataFrame, f1_df: pd.DataFrame) -> tuple:
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
    from ml_model import NeuralNet
    import torch.nn as nn
    
    torch.manual_seed(p['torch seed'])
    network_details = {'input size': p['input size'], 'hidden size': p['hidden layer size'], 'epochs': p['number of epochs'],
                       'batch size': p['batch size'], 'learning rate': p['learning rate'], 'dropout': p['dropout probability']}

    if p['num_vars'] == 1:
    
        for iteration_number in range(p['total passes']):
            
            class_distributions = {'training': [], 
                                   'testing': []}
            
            # Initialize the results dictionary for this iteration
            iteration_results, accuracy_df, f1_df = initialize_new_ml_iteration(p = p,
                                                                                iteration_number = iteration_number,
                                                                                iteration_inds=None,
                                                                                accuracy_df = accuracy_df,
                                                                                f1_df = f1_df)
            
            # Generate polluted data
            iteration_data, pollute_summary = pollute_data(p = p,
                                                           central_data = central_data, 
                                                           iteration_inds=[iteration_number])
            
            # Save spectrum if requested
            save_checked_spectrum(p, dirs, check_spectrum, iteration_data, iteration_number)
            
            # Format datasets for Pytorch
            training_data, testing_data, class_distributions = format_pytorch_datasets(p, class_distributions, iteration_data)
            
            # Set up model
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            train_loader = DataLoader(dataset=training_data, batch_size=p['batch size'], shuffle=True)
            test_loader = DataLoader(dataset=testing_data, batch_size=p['batch size'], shuffle=False)
            model = NeuralNet(p['input size'], p['hidden layer size'], classes['N'], p['dropout probability']).to(device)
            
            flatten = nn.Flatten()  # convert from w1 x w3 (2D) to w1w3 (1D)
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=p['learning rate'])
            
            # ----- Train model -----
            
            training_dict, testing_in_training_dict = initialize_training(p, train_loader, test_loader) 
            start_time = time.perf_counter()
            model.train()                                                                               # ensure model is set to training state
            
            for epoch_index in range(p['number of epochs']):
                
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
                print_to_log_file(dirs['log file'], f"  completed {epoch_index+1} of {p['number of epochs']} epochs")
                
            # end training
            print_to_log_file(dirs['log file'], f'Time allocated to training: {np.round((time.perf_counter() - start_time))} sec')
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
            remove_workspace_variables(variables_to_delete)

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
                save_checked_spectrum(p, dirs, check_spectrum, iteration_data, iteration_number = count)                             
                
                # Format datasets for Pytorch
                training_data, testing_data, class_distributions = format_pytorch_datasets(p, class_distributions, iteration_data)
                
                # ----- Set up model -------
                
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
                train_loader = DataLoader(dataset=training_data, batch_size=p['batch size'], shuffle=True)   # dataloaders automatically shuffle and split data up into batches of requested size
                test_loader = DataLoader(dataset=testing_data, batch_size=p['batch size'], shuffle=False)
                    
                model = NeuralNet(p['input size'], p['hidden layer size'], classes['N'], p['dropout probability']).to(device)
                
                flatten = nn.Flatten()                                                                      # convert from w1 x w3 (2D) to w1w3 (1D)
                criterion = nn.CrossEntropyLoss()
                optimizer = torch.optim.Adam(model.parameters(), lr=p['learning rate'])
                
                # ----- Train model -----
                
                training_dict, testing_in_training_dict = initialize_training(p, train_loader, test_loader) 
                start_time = time.perf_counter()
                model.train()                                                                               # ensure model is set to training state
                
                for epoch_index in range(p['number of epochs']):
                    
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
                    print_to_log_file(dirs['log file'], f"  completed {epoch_index+1} of {p['number of epochs']} epochs")
                    
                # end training
                print_to_log_file(dirs['log file'], f'Time allocated to training: {np.round((time.perf_counter() - start_time))} sec')
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

# . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 

def format_pytorch_datasets(p: dict, class_distributions: dict, iteration_data: dict) -> tuple:
    """
    Prepares datasets for PyTorch training and testing.
    
    Parameters:
        p (dict): Configuration parameters.
        class_distributions (dict): Dictionary storing distribution of classes in training/testing.
        iteration_data (dict): The dataset for the current iteration.
    
    Returns:
        tuple: (Training dataset, Testing dataset, Updated class distributions)
    """
    
    print_to_log_file(dirs['log file'],'Formatting data for Pytorch...')
    
    training_data = PyTorchDataset(iteration_data, p, isTrain=True)
    testing_data = PyTorchDataset(iteration_data, p, isTrain=False)
    
    total_number_spectra = len(testing_data) + len(training_data)
    print_to_log_file(dirs['log file'], f'total number of spectra for training and testing = {total_number_spectra}')
    
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

# . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 

def initialize_new_ml_iteration(p: dict, iteration_number: int, iteration_inds: list, accuracy_df: pd.DataFrame, f1_df: pd.DataFrame) -> tuple:
    """
    Initializes a new ML iteration by setting up the required parameters.
    
    Parameters:
        p (dict): Configuration parameters.
        iteration_number (int): Current iteration number.
        iteration_inds (list): Indices for the iteration process.
        accuracy_df (pd.DataFrame): Dataframe for accuracy logging.
        f1_df (pd.DataFrame): Dataframe for F1-score logging.
    
    Returns:
        tuple: (Iteration results dictionary, updated accuracy_df, updated f1_df)
    """
    if p['num_vars'] == 1:
        
        print_to_log_file(dirs['log file'],f"----------------\nIteration {iteration_number+1} of {p['total passes']}")
        iteration_variable_name = p['scan_item_df']
        
        if p['task'] == 'ablation':
            if iteration_number == 0:
                iteration_variable_value = None
            else:
                iteration_variable_value = p[p['scan key']][iteration_number-1]
        else:
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
        iteration_variable2_value = p[p['scan key']['1']][iteration_inds[1]]
        
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
        print_to_log_file(dirs['log file'],"ERROR: unsupported stage requested. Exiting.")
        raise Exception("ERROR: Unsupported stage requested.")
        sys.exit()
    
    return running_dict

# . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 

def initialize_training(p, train_loader, test_loader):
    
    print_to_log_file(dirs['log file'],'TRAINING STAGE')
    
    n_batches_training = len(train_loader)
    n_batches_testing = len(test_loader)
    print_to_log_file(dirs['log file'], f'{n_batches_training} batches in train loader, {n_batches_testing} in test loader...''')
    
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

# . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 

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

# . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 
        
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

# . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 

def save_iteration_outputs(p, network_details, classes, class_distributions, 
                           training_dict, testing_in_training_dict, testing_dict, 
                           iteration_number, iteration_results, pollute_summary):
  
    if p['save_ML_output'] == 'True':
        
        print_to_log_file(dirs['log file'], f'Saving results for iteration {iteration_number+1}...')
        
        iteration_results.update({'network details': network_details, 'classes': classes, 'class distributions': class_distributions,
                                  'TRAINING details': training_dict, 'TESTING IN TRAINING details': testing_in_training_dict, 'TESTING details': testing_dict,
                                  'pollution summary': pollute_summary})
        
        with open(f"{dirs['outputs']}/job{p['jobname']}_ITERATION{iteration_number+1}.pkl", 'wb') as f:
            pickle.dump(iteration_results, f)      
        
        print_to_log_file(dirs['log file'],'Results saved...\n')
        
    pass

# . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .         

def save_ml_reports(p, training_dict, testing_in_training_dict, iteration_number):
    if p['save_ML_report_images'] == 'True':
        try:
            import os
            save_loc = os.path.join(dirs['outputs'],"Training_plots")
            if not os.path.exists(save_loc):
                os.makedirs(save_loc)
                
            xvect = np.arange(p['number of epochs']) + 1
            keys_to_plot = ['accuracy', 'micro f1', 'macro f1']
            for key in keys_to_plot:
                create_and_save_line_plot(xvect, training_dict[key], key, 'epoch', f'{key} (%)', 0, 100, 
                                          f"{save_loc}/job{p['jobname']}_train_{key.strip()}_{iteration_number+1}.png")
            
            #save loss plot
            create_and_save_line_plot(xvect, training_dict['loss per image'], 'loss per image', 'epoch', 'loss per image', 0, np.max(training_dict['loss per image']), 
                                      f"{save_loc}/job{p['jobname']}_train_loss_per_image_{iteration_number+1}.png")
            
            for key in keys_to_plot:
                create_and_save_line_plot(xvect, testing_in_training_dict[key], key, 'epoch', f'{key} (%)', 0, 100,
                                          f"{save_loc}/job{p['jobname']}_traintest_{key.strip()}_{iteration_number+1}.png")

        except Exception as e:
            print_to_log_file(dirs['log file'], f"An error occurred while generating or saving plots: {e}")
    
    pass
    
# . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .            
        
def update_dataframes_post_testing(p, iteration_number, iteration_results, testing_dict, accuracy_df, f1_df):
    
    if p['num_vars'] == 1:
        print_to_log_file(dirs['log file'], f'''~~~ RESULTS ~~~
{p['scan_item_df']} = {iteration_results['scanned item value']}''')

    elif p['num_vars'] == 2:
        print_to_log_file(dirs['log file'], f'''~~~ RESULTS ~~~
{p['scan key']['1']} = {iteration_results['scanned item 1 value']}
{p['scan key']['2']} = {iteration_results['scanned item 2 value']}''')
     
    print_to_log_file(dirs['log file'], f'''
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

# . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 

def update_dataframes_post_training(iteration_number, accuracy_df, f1_df, training_dict, testing_in_training_dict):
    
    accuracy_df.at[iteration_number+1, 'Train accuracy'] = training_dict['accuracy'][-1]
    f1_df.at[iteration_number+1, 'Train micro f1'] = training_dict['micro f1'][-1]
    f1_df.at[iteration_number+1, 'Train macro f1'] = training_dict['macro f1'][-1]
    f1_df.at[iteration_number+1, 'Train weighted f1'] = training_dict['weighted f1'][-1]
    
    accuracy_df.at[iteration_number+1, 'Train-test accuracy'] = testing_in_training_dict['accuracy'][-1]
    f1_df.at[iteration_number+1, 'Train-test micro f1'] = testing_in_training_dict['micro f1'][-1]
    f1_df.at[iteration_number+1, 'Train-test macro f1'] = testing_in_training_dict['macro f1'][-1]
    f1_df.at[iteration_number+1, 'Train-test weighted f1'] = testing_in_training_dict['weighted f1'][-1]
    
# . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 
    
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

# . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .     