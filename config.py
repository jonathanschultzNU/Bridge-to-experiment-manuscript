import os

def get_directories():

    database_dir = '...'
    
    dirs = {'working': os.getcwd(),
            'database': database_dir,
            'labels': os.path.join(database_dir, 'labels.csv'),
            'Outputs': os.path.join(os.getcwd(), 'Outputs')
            }
    
    return dirs
