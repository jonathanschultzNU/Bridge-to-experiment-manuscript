"""
Module to handle the logging of results and saving of optional reports
"""

import torch
import numpy as np
import pickle
from modules.visualization import create_and_save_line_plot
from modules.utils import print_to_log_file
from modules.ml_model import f1_score_custom
from sklearn.metrics import top_k_accuracy_score

# . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 

def initialize_running_dict(p, dirs, stage):

    if stage == 'training':
        running_dict = {'number of images': [], 
                        'number of correct predictions': [],
                        'system ID numbers': [],
                        't2s': [],
                        'labels': [], 
                        'predictions': [], 
                        'loss': [],
                        }
        
    elif stage == 'testing in training':
        running_dict = {'number of images': [], 
                        'number of correct predictions': [],
                        'system ID numbers': [],
                        't2s': [],
                        'labels': [], 
                        'predictions': [], 
                        }
        
    elif stage == 'testing':
        running_dict = {'number of images': [], 
                        'number of correct predictions': [],
                        'system ID numbers': [],
                        't2s': [],
                        'labels': [], 
                        'predictions': [], 
                        'predictions with probabilities': []
                        }   
        
    else:
        error_message = f"ERROR: Unsupported stage '{stage}' requested. Valid options: ['training', 'testing in training', 'testing']."
        print_to_log_file(dirs['log file'], error_message)
        raise ValueError(error_message)
    
    return running_dict

# . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 

def initialize_training(p, dirs, train_loader, test_loader):
    
    print_to_log_file(dirs['log file'],'TRAINING STAGE')
    
    n_batches_training = len(train_loader)
    n_batches_testing = len(test_loader)
    print_to_log_file(dirs['log file'], f'{n_batches_training} batches in train loader, {n_batches_testing} in test loader...''')
    
    # dictionaries to keep track of items per epoch per batch
    training_dict = {'number of batches': n_batches_training,
                     'epoch index': [], 
                     'system ID numbers': [],
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
                                'system ID numbers': [],
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
    stage_dict['system ID numbers'].append(running_dict['system ID numbers'])
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
                    'system ID numbers': running_dict['system ID numbers'],
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

def save_iteration_outputs(p, dirs, network_details, classes, class_distributions, 
                           training_dict, testing_in_training_dict, testing_dict, 
                           iteration_number, iteration_results, pollute_summary):
  
    if p['save iteration outputs'] == 'full':
        
        try:
        
            import os
            save_loc = os.path.join(dirs['outputs'],"ML_reports")
            if not os.path.exists(save_loc):
                os.makedirs(save_loc)
            
            print_to_log_file(dirs['log file'], f'Saving full results for iteration {iteration_number+1}...')
            
            iteration_results.update({'network details': network_details, 'classes': classes, 'class distributions': class_distributions,
                                      'TRAINING details': training_dict, 'TESTING IN TRAINING details': testing_in_training_dict, 'TESTING details': testing_dict,
                                      'pollution summary': pollute_summary})
            
            with open(f"{save_loc}/ITERATION_{iteration_number+1}.pkl", 'wb') as f:
                pickle.dump(iteration_results, f)      
            
            print_to_log_file(dirs['log file'],'Results saved...\n')
            
        except Exception as e:
            print_to_log_file(dirs['log file'], f"An error occurred while saving iteration results: {e}")
        
        
    if p['save iteration outputs'] == 'partial':
        
        try:    
            import os
            save_loc = os.path.join(dirs['outputs'],"ML_reports")
            if not os.path.exists(save_loc):
                os.makedirs(save_loc)
            
            print_to_log_file(dirs['log file'], f'Saving partial results for iteration {iteration_number+1}...')
            
            iteration_results.update({'network details': network_details, 
                                      'classes': classes, 
                                      'class distributions': class_distributions,
                                      'TESTING details': testing_dict,
                                      'pollution summary': pollute_summary})
            
            with open(f"{save_loc}/ITERATION_{iteration_number+1}.pkl", 'wb') as f:
                pickle.dump(iteration_results, f)      
            
            print_to_log_file(dirs['log file'],'Results saved...\n')
            
        except Exception as e:
            print_to_log_file(dirs['log file'], f"An error occurred while generating or saving training reports: {e}")
            
    else:
        pass

# . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .         

def save_training_reports(p, dirs, training_dict, testing_in_training_dict, iteration_number):
    
    if p['save training reports'] == 'True':
        try:
            import os
            save_loc = os.path.join(dirs['outputs'],"Training_reports")
            if not os.path.exists(save_loc):
                os.makedirs(save_loc)
                
            xvect = np.arange(p['number of epochs']) + 1
            keys_to_plot = ['accuracy', 'micro f1', 'macro f1']
            for key in keys_to_plot:
                create_and_save_line_plot(xvect, training_dict[key], key, 'epoch', f'{key} (%)', 0, 100, 
                                          f"{save_loc}/train_{key.strip()}_{iteration_number+1}.png")
            
            #save loss plot
            create_and_save_line_plot(xvect, training_dict['loss per image'], 'loss per image', 'epoch', 'loss per image', 0, np.max(training_dict['loss per image']), 
                                      f"{save_loc}/train_loss_per_image_{iteration_number+1}.png")
            
            for key in keys_to_plot:
                create_and_save_line_plot(xvect, testing_in_training_dict[key], key, 'epoch', f'{key} (%)', 0, 100,
                                          f"{save_loc}/traintest_{key.strip()}_{iteration_number+1}.png")

        except Exception as e:
            print_to_log_file(dirs['log file'], f"An error occurred while generating or saving training reports: {e}")
    
    else:
        pass

    
# . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .            
        
def update_dataframes_post_testing(p, dirs, iteration_number, iteration_results, testing_dict, accuracy_df, f1_df):
    
    if p['num_vars'] == 1:
        print_to_log_file(dirs['log file'], f'''~~~ RESULTS ~~~
{p['scan key']} = {iteration_results['scanned item value']}''')

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
    running_dict['system ID numbers'].extend(IDs.squeeze().cpu().numpy())
    running_dict['t2s'].extend(t2s.squeeze().cpu().numpy())
    running_dict['labels'].extend(labels.cpu().numpy())
    running_dict['predictions'].extend(predictions.cpu().numpy())
    
    if stage == 'training':
        running_dict['loss'].append(loss.item())
        
    elif stage == 'testing':
        running_dict['predictions with probabilities'].append(probabilities)     # Accumulate predictions and labels

# . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .     