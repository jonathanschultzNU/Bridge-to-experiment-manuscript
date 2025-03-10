'''
TASK: mapping couplings in molecular dimers 
  
ML APPROACH: Feed-forward neural network (FFNN)

CAPABILITIES:
    - Noise can be introduced to spectra
    - Spectra with low signal-to-noise ratio can be dropped
    - Amount of noise can be scanned to determine effect(s) on FFNN performance
        
KEY NOTES:  
    - Clean data are loaded into a 'central dataset' (avoids re-loading data between scan points)
    - The number of t2 timepoints along the waiting time dimension in each dataset MUST be consistent across all datasets
'''

import os
import pickle

# Import from modules
from modules.config import get_directories
from modules.ml_model import get_num_ml_iterations
from modules.utils import print_to_log_file, check_inputs, convert_parameter_datatypes, initalize_dataframes, read_input_file, get_git_info
from modules.data_processing import load_central_dataset, classify_central_dataset
from modules.debugging_tools import initialize_check_system
from modules.training_pipeline import ML_iterations


if __name__ == "__main__":
    
    verbose = True
    dirs = get_directories()
    
    # Read inputs
    f = open(os.paths.join(dirs['working'], "input.txt"))
    p = read_input_file(f)
    f.close()
    
    runlist_file = os.paths.join(dirs['working'], "runlist.txt")
    if not runlist_file:
        runlist_file = None

    dirs['outputs'] = os.path.join(dirs['working'], f"job{p['jobname']}_Outputs")
    dirs['log file'] = os.path.join(dirs['outputs'], f"job{dirs['jobname']}.log")
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
    central_data = load_central_dataset(runlist_file, p) 
    central_data, class_information = classify_central_dataset(p, central_data)
    
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
    accuracy_df.to_csv(f"{dirs['outputs']}/job{p['jobname']}_Accuracies.csv", index_label = 'Iteration')
    f1_df.to_csv(f"{dirs['outputs']}/job{p['jobname']}_F1scores.csv", index_label = 'Iteration')
    
    # save inputs
    with open(f"{dirs['outputs']}/job{p['jobname']}_inputs.pkl", 'wb') as f:
        pickle.dump(p, f)
        
    print_to_log_file(dirs['log file'],'''~~~ All iterations complete! ~~~\n\nNow exiting.''')