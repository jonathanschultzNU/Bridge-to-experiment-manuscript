import os

# . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 

def get_directories():
    """
    Returns a dictionary of important directories.
    Users can set 'DATABASE_DIR' as an environment variable or modify database_dir below.
    """
    
    # Specify database bath
    database_dir = os.path.join(os.path.dirname(os.getcwd()), "Mini_database")
    
    # Validate that database_dir exists
    if not os.path.exists(database_dir):
        print(f"ERROR: The specified database directory '{database_dir}' does not exist.")
    
    # Define directories
    dirs = {'working': os.getcwd(),
            'database': database_dir,
            'dataset': os.path.join(database_dir, 'mini_dataset.pkl'),
            'labels': os.path.join(database_dir, 'mini_dataset_system_parameters.csv')
            }
    
    return dirs
