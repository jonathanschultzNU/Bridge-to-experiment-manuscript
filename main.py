'''
TASK: mapping couplings in molecular dimers 
  
ML APPROACH: Feed-forward neural network (FFNN)

CAPABILITIES:
    Noise can be introduced to spectra
    Spectra with low signal-to-noise ratio can be dropped
    Amount of noise can be scanned to determine effect(s) on FFNN performance
        
KEY NOTES:  
    Clean data are loaded into a 'central dataset' (avoids re-loading data between scan points)
    The number of t2 timepoints along the waiting time dimension in each dataset MUST be consistent across all datasets
'''

import os
import sys
import pickle
import supplemental as fcns


if __name__ == "__main__":
    
    # UNCOMMENT IF RUNNING LOCALLY
    # if len(sys.argv) == 1:  # Default behavior if no args are passed
    #     sys.argv = ["ml_main.py", "additive_test_local.inp", "runlist_local.txt"]  # Replace with desired arguments
    
    verbose = True
    input_file = sys.argv[1]
    runlist_file = sys.argv[2]
    
    # Read and check inputs
    f = open(input_file)
    p = fcns.read_input_file(f)
    f.close()
    
    p['working directory'] = os.getcwd()
    p['Slurm job ID'] = os.environ.get("SLURM_JOB_ID")    # COMMENT IF RUNNING LOCALLY
    # p['Slurm job ID'] = 12345                               # UNCOMMENT IF RUNNING LOCALLY
    
    p['outputs_path'] = os.path.join(p['working directory'], f"job{p['Slurm job ID']}_Outputs")
    if not os.path.exists(p['outputs_path']):
        os.makedirs(p['outputs_path'])
        
    p['log filename'] = os.path.join(p['outputs_path'], f"job{p['Slurm job ID']}.OREOSlog")
    
    # set up log file
    with open(p['log filename'], "w") as f: 
        f.write("ML2DVS-Feedforward-Variable")
    print(f"See {p['log filename']} for further job details.", flush=True)
    
    fcns.print_to_log_file(p['log filename'], f'''\n
Branch: FFNN_scan_editing
Working directory: {p['working directory']}
job ID : {p['Slurm job ID']}''')


    # Process inputs    
    fcns.check_inputs(p)
    fcns.convert_parameter_datatypes(p)
    fcns.get_num_ml_iterations(p)
    
    # Load and classify central dataset
    central_data = fcns.load_central_dataset(runlist_file, p) 
    central_data, class_information = fcns.classify_central_dataset(p, central_data)
    
    p, accuracy_df, f1_df = fcns.initalize_dataframes(p)
    check_spectrum = fcns.initialize_check_system(p, central_data)
    
    # ML iterations      
    p, accuracy_df, f1_df = fcns.ML_iterations(p = p,
                                            central_data = central_data,
                                            classes = class_information,
                                            check_spectrum = check_spectrum, 
                                            accuracy_df = accuracy_df, 
                                            f1_df = f1_df)
    
    # Save final results                         
    accuracy_df.to_csv(f"{p['outputs_path']}/job{p['Slurm job ID']}_Accuracies.csv",
                       index_label = 'Iteration'
                       )
    f1_df.to_csv(f"{p['outputs_path']}/job{p['Slurm job ID']}_F1scores.csv",
                 index_label = 'Iteration'
                 )
    
    # save inputs
    with open(f"{p['outputs_path']}/job{p['Slurm job ID']}_inputs.pkl", 'wb') as f:
        pickle.dump(p, f)   
        
    fcns.print_to_log_file(p['log filename'],'''~~~ All iterations complete! ~~~\n\nNow exiting.''')