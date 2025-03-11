import os

# . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 

def get_directories():
    """
    Returns a dictionary of important directories.
    Users can set 'DATABASE_DIR' as an environment variable or modify database_dir below.
    """
    
    # Allow setting database directory via environment variable
    # database_dir = os.getenv("DATABASE_DIR", "/path/to/database")  # Default if not set
    
    # OR: Manual specification
    database_dir = os.path.join(os.path.dirname(os.getcwd()), "Mini_database")
    
    # Ensure the user didn't forget to set the path
    if database_dir == "/path/to/database":
        print("WARNING: Database directory is using the default placeholder. Please update 'config.py'.")
    
    # Validate that database_dir exists
    if not os.path.exists(database_dir):
        print(f"ERROR: The specified database directory '{database_dir}' does not exist.")
    
    # Define directories
    dirs = {'working': os.getcwd(),
            'database': database_dir,
            'dataset': os.path.join(database_dir, '/data_subset.pkl'),
            'labels': os.path.join(database_dir, 'data_subset_system_parameters.csv')
            }
    
    return dirs
