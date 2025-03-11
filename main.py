'''
Main code for Bridge-to-experiment-manuscript repo
'''

import os
import pickle
from modules.config import get_directories
from modules.ml_model import get_num_ml_iterations
from modules.utils import print_to_log_file, check_inputs, convert_parameter_datatypes, initalize_dataframes, read_input_file, get_git_info
from modules.data_processing import load_central_dataset, classify_central_dataset
from modules.debugging_tools import initialize_check_system
from modules.training_pipeline import ML_iterations

# . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 

if __name__ == "__main__":
    
    verbose = True
    dirs = get_directories()
    
    # Read inputs
    with open(os.path.join(dirs['working'], "input.txt"), "r") as f:
        p = read_input_file(f)

    dirs['outputs'] = os.path.join(dirs['working'], f"job{p['jobname']}_Outputs")
    dirs['log file'] = os.path.join(dirs['outputs'], f"job{p['jobname']}.log")
    os.makedirs(dirs['outputs'], exist_ok=True)
    
    # set up log file
    with open(dirs['log file'], "w") as f: 
        f.write("Log file initiated.")
    
    print_to_log_file(dirs['log file'], f'''\n
Git Repository Information:
   {get_git_info()}

Working directory: {dirs['working']}
job name : {p['jobname']}''')

    # Process inputs    
    check_inputs(p, dirs)
    convert_parameter_datatypes(p)
    get_num_ml_iterations(p, dirs)
    
    # Load and classify central dataset
    central_data = load_central_dataset(p, dirs['dataset']) 
    central_data, class_information = classify_central_dataset(p, dirs['labels'], central_data)
    
    # Initialize objects for later
    p, accuracy_df, f1_df = initalize_dataframes(p)
    check_spectrum = initialize_check_system(p, central_data)
    
    # ML iterations      
    p, accuracy_df, f1_df = ML_iterations(
        p = p,
        dirs = dirs,
        central_data = central_data,
        classes = class_information,
        check_spectrum = check_spectrum, 
        accuracy_df = accuracy_df, 
        f1_df = f1_df
        )
    
    # Save final results                         
    accuracy_df.to_csv(f"{dirs['outputs']}/accuracies.csv", index_label = 'Iteration')
    f1_df.to_csv(f"{dirs['outputs']}/F1scores.csv", index_label = 'Iteration')
    
    # Save inputs
    with open(f"{dirs['outputs']}/inputs.pkl", 'wb') as f:
        pickle.dump(p, f)
        
    print_to_log_file(dirs['log file'],'''~~~ All iterations complete! ~~~\n\nNow exiting.''')