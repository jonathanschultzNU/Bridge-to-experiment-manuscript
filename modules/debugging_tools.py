"""
Module with functions for validation, debugging, and sanity checks of dataset consistency.
"""

import numpy as np

# . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 

def find_image_index(central_data: dict, ID: int) -> int:
    """
    Finds the index of a dataset entry by its system ID.

    Parameters:
        central_data (dict): Dictionary containing dataset information, including system index.
        ID (int): The system ID number to search for.

    Returns:
        int: Index of the dataset entry if found, otherwise None.
    """
    
    sys_index = np.where(central_data['system ID numbers'] == ID)[0]
    return sys_index.item() if len(sys_index) > 0 else None

# . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 

def initialize_check_system(p, central_data):
    """
    Selects a system from the dataset to serve as an example for visualization.

    If the specified `check system ID` exists in the dataset, it is used. Otherwise, 
    a random system is selected.

    Parameters:
        p (dict): Dictionary containing user-defined parameters, including 'check system ID'.
        central_data (dict): Dictionary containing dataset information, including system index and spectral data.

    Returns:
        dict: A dictionary containing the selected system's details:
            - 'system index selected' (int): The index of the selected system in `central_data`.
            - 'w1' (numpy.ndarray): The w1 axis values for the selected system.
            - 'w3' (numpy.ndarray): The w3 axis values for the selected system.
            - 't2' (float): The t2 waiting time value for the selected system.
            - 'system ID number' (int): The system ID of the selected system.
    """
    
    system_index_selected = find_image_index(central_data, p['check-system ID'])
    
    if system_index_selected == None:
        import random
        system_index_selected = random.randint(0, central_data['Number of systems']-1)
        
    check_spectrum = {'system index selected': system_index_selected, 
                      'w1': central_data['w1'][system_index_selected,:], 
                      'w3': central_data['w3'][system_index_selected,:], 
                      't2': central_data['t2'][system_index_selected, 0], 
                      'system ID number': central_data['system ID numbers'][system_index_selected]}
    
    return check_spectrum

