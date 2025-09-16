# -*- coding: utf-8 -*-
"""
@author: Ignacio Erazo
"""
import pickle
from scipy.stats import gmean
import numpy as np

"""
This file is meant to be used to obtain the results for tables 4 and 5 of 
the paper.
The functions below can be expanded to retrieve more granular information from
the experiments, as most data is stored in the pickle files. Please see the
'run_family_setups_small_experiments.py' file to see what information is 
stored in the pickle files.
"""

def open_pickle_object_and_pass_data(
        file_path: str
        ) -> object:
    """
    Function to open the pickle files and pass the content as a python object.
    """
    try:
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
        print("Data loaded successfully:")
        return data
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")
    return

def print_results_for_a_given_experiment(
        path_for_results: str,
        results_were_computed_with_small_experiments_script: bool=True
    ):
    """
    Function to print the results for the experiment; particularly the ones
    in the tables 4 and 5 of the paper.
    """
    data = open_pickle_object_and_pass_data(file_path=path_for_results)
    # Used to characterize the instances
    replications = data[0]
    scenario = data[1]
    list_methods = data[2]
    # Relevant information in the context of Tables 4 and 5
    gaps = data[3]
    # Not in the tables, but can be studied
    running_times = data[4]
    times_equal_benchmark = data[5]
    best_lower_bound = data[6]
    best_heuristic = data[7]
    columns_generated = data[8]
    print(f'The number of replications for the experiment {replications}')
    print('The scenario had:',
          f'{scenario[0]} orders,',
          f'{scenario[1]} machines,',
          f'{scenario[2]} families,',
          f'{scenario[3]} uniform parameter for orders,',
          f'{scenario[4]} uniform parameter families,',
          f'{scenario[5]} uniform parameter arrival')
    
    # In this scenario we are concerned with the ordering given by the pickle
    # files in this repo
    if results_were_computed_with_small_experiments_script:
        lower_bound_indices = {0, 1, 3, 5, 7, 9}
        print('Going over the lower bounds')
        for index, method in enumerate(list_methods):
            if index in lower_bound_indices:
                print(f'For {method = } the geometric gap was {gmean(gaps[index]): .3f}, and min gap {np.min(gaps[index]): .3f}')
        
        print('Going over the heuristic results')
        for index, method in enumerate(list_methods):
            if index not in lower_bound_indices:
                print(f'For {method = } the gap was {gmean(gaps[index]) :.3f}, and max gap {np.max(gaps[index]) :.3f}')

    # We do not assume ordering of the methods
    else:
        print(f'The list of methods {list_methods}')
        for index, method in enumerate(list_methods):
            print(f'For {method = } the gap was {gmean(gaps[index]): .3f}')
    return

if __name__ == '__main__':
    #file_path = "results/family_setups_table_results/results_(160, 15, 15, (1, 101), (0, 51), (0, 10))_120_sec.pickle"
    list_paths = [
        "results/family_setups_small_experiments_results/results_(15, 2, 3, (1, 100), (0, 51), (0, 26)).pickle",
        "results/family_setups_small_experiments_results/results_(15, 2, 3, (1, 100), (0, 51), (0, 101)).pickle",
        "results/family_setups_small_experiments_results/results_(15, 2, 3, (1, 100), (0, 101), (0, 26)).pickle",
        "results/family_setups_small_experiments_results/results_(15, 2, 3, (1, 100), (0, 101), (0, 101)).pickle",
        "results/family_setups_small_experiments_results/results_(15, 5, 3, (1, 100), (0, 51), (0, 26)).pickle",
        "results/family_setups_small_experiments_results/results_(15, 5, 3, (1, 100), (0, 51), (0, 101)).pickle",
        "results/family_setups_small_experiments_results/results_(15, 5, 3, (1, 100), (0, 101), (0, 26)).pickle",
        "results/family_setups_small_experiments_results/results_(15, 5, 3, (1, 100), (0, 101), (0, 101)).pickle",
        "results/family_setups_small_experiments_results/results_(15, 2, 5, (1, 100), (0, 51), (0, 26)).pickle",
        "results/family_setups_small_experiments_results/results_(15, 2, 5, (1, 100), (0, 51), (0, 101)).pickle",
        "results/family_setups_small_experiments_results/results_(15, 2, 5, (1, 100), (0, 101), (0, 26)).pickle",
        "results/family_setups_small_experiments_results/results_(15, 2, 5, (1, 100), (0, 101), (0, 101)).pickle",
        "results/family_setups_small_experiments_results/results_(15, 5, 5, (1, 100), (0, 51), (0, 26)).pickle",
        "results/family_setups_small_experiments_results/results_(15, 5, 5, (1, 100), (0, 51), (0, 101)).pickle",
        "results/family_setups_small_experiments_results/results_(15, 5, 5, (1, 100), (0, 101), (0, 26)).pickle",
        "results/family_setups_small_experiments_results/results_(15, 5, 5, (1, 100), (0, 101), (0, 101)).pickle"
    ]
    for file_path in list_paths:
        print(file_path)
        data = print_results_for_a_given_experiment(path_for_results=file_path)
        print('\n')