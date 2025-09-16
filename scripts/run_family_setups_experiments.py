# -*- coding: utf-8 -*-
"""
@author: Ignacio Erazo
"""
import numpy as np
import sys
import os
from tqdm import tqdm
import time
import itertools
import pickle
from scipy.stats import gmean

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from src.family_setups.tool_functions import create_input_family_setups

from src.family_setups.lower_bounds import lower_bound_improved_families

from src.family_setups.regular_formulations import naive_formulation_families, \
    naive_quadratic_interval_formulation_families

from src.family_setups.set_cover_methods import colgen_set_cover_families, \
    colgen_set_cover_families_strong
    
from src.family_setups.regular_formulations_column_generation import \
    colgen_naive_families, colgen_flow_families


PATH_TO_STORE_RESULTS = "results/family_setups_small_experiments_pickle_results_computed_by_script/"

def save_object_in_pickle(
        object_being_saved: object,
        name_file: str,
        path_to_store_file: str=PATH_TO_STORE_RESULTS,
        extension: str = '.pickle'
        ) -> None:
    """
    This function is used to save a python object into the hard drive. 
    The object will be saved as default as a .pickle file, which can be open 
    easily afterwards.

    Args:
        objectBeingSaved [object]: Just a python object to save
        nameFile [str]: Name to use for the file where the object will live
        extension [str]: Extension if needed something different than .pickle
    
    Returns:
        None, the file will be saved in current directory. 
    """
    assert type(name_file) == str and type(extension) == str, \
        "You need to provide a valid name file and extension as strings"
    # make sure the directory exists
    os.makedirs(name=path_to_store_file, exist_ok=True)
    # store the file
    pickleBucket = open(path_to_store_file + name_file + extension, "wb")
    pickle.dump(object_being_saved, pickleBucket)
    pickleBucket.close()
    return    


do_small_experiments = True
do_large_experiments = True


if __name__ == '__main__':
    if do_small_experiments:
        replications = 25
        number_orders = [15]
        number_families = [2, 5]
        number_vehicles = [3, 5]
        params_orders = [(1, 101)]
        params_families = [(0, 51), (0, 101)]
        params_arrivals = [(0, 26), (0, 51), (0, 101)]
        
        all_scenarios = itertools.product(number_orders, number_families, 
            number_vehicles, params_orders, params_families, params_arrivals)
        
        list_methods = ['IP Naive', 
            'CG Form 1 y 2 (Naive) LP', 'CG Form 1 y 2 (Naive) Heuristic',
            'CG Form Flow LP', 'CG Form Flow Heuristic', 
            'CG Form Set Cover LP STRONG', 'CG Form Set Cover STRONG Heuristic',
            'CG Form Set Cover WEAK LP', 'CG Form Set Cover WEAK Heuristic',
            'Lower Bound for Families', 'Quadratic interval IP']
        
        iteration = 0
        for scenario in all_scenarios:
            iteration += 1
            # creating data structures to hold our results
            gaps = np.zeros(shape=(len(list_methods), replications))
            running_times = np.zeros(shape=(len(list_methods), replications))
            columns_generated = np.zeros(shape=(len(list_methods), replications))
            times_equal_benchmark = np.zeros(shape=(len(list_methods), replications))
            best_lower_bound = np.zeros(shape=(len(list_methods), replications))
            best_heuristic = np.zeros(shape=(len(list_methods), replications))
            
            # Start processing this scenario
            print(f'Starting scenario (orders, families, vehicles, (params order, params fam y params arriv))')
            print(f'{scenario = }; {iteration = }')
            for rep in tqdm(range(replications)):
                np.random.seed(rep + iteration * 10000)
                # Create input
                arrivals, family_setup_times, processing_times_orders, \
                    orders_to_families_assignment = create_input_family_setups(
                    number_orders=scenario[0],
                    number_families=scenario[1],
                    params_distribution_orders={'low': scenario[3][0], 'high': scenario[3][1]},
                    params_distribution_families={'low': scenario[4][0], 'high': scenario[4][1]},
                    params_distribution_interarrivals={'low': scenario[5][0], 'high': scenario[5][1]},
                    )
                
                # parameters
                params = {'number_orders': scenario[0], 'number_families': scenario[1],
                    'number_vehicles': scenario[2],
                    'arrivals': arrivals,
                    'processing_times_orders': processing_times_orders,
                    'orders_to_families_assignment': orders_to_families_assignment,
                    'family_setup_times': family_setup_times}

                # Start solving the IP fully
                start = time.time()
                gaps[0, rep] = naive_formulation_families(lp_relaxation=False,
                    **params)
                end = time.time()
                running_times[0, rep] = end - start
                
                # Parameters for the heuristic and column generations
                params['p_feas'] = 0.5
                params['do_heuristic'] = True
                
                # Colgen naive\
                gaps[1, rep], gaps[2, rep], running_times[1, rep],\
                    running_times[2, rep], columns_generated[1, rep]\
                        = colgen_naive_families(**params)

                # Flow
                gaps[3, rep], gaps[4, rep], running_times[3, rep],\
                    running_times[4, rep], columns_generated[3, rep]\
                        = colgen_flow_families(**params)
                
                # Set Cover methods
                gaps[5, rep], gaps[6, rep], running_times[5, rep],\
                    running_times[6, rep], columns_generated[5, rep]\
                        = colgen_set_cover_families_strong(**params)
                
                start = time.time()
                gaps[7, rep], gaps[8, rep], running_times[7, rep],\
                    running_times[8, rep], columns_generated[7, rep]\
                        = colgen_set_cover_families(**params)
                running_times[7, rep] = time.time() - start
                
                # LB Improved families
                start = time.time()
                gaps[9, rep] = lower_bound_improved_families(
                    number_orders=scenario[0],
                    number_vehicles=scenario[2],
                    arrivals=arrivals, 
                    processing_times_orders=processing_times_orders, 
                    orders_to_families_assignment=orders_to_families_assignment,
                    family_setup_times=family_setup_times,
                    fixed_cost=0
                    )
                running_times[9, rep] = time.time() - start
                
                # Quadratic Interval IP
                start = time.time()
                gaps[10, rep] = naive_quadratic_interval_formulation_families(
                    number_orders=scenario[0],
                    number_vehicles=scenario[2],
                    number_families=scenario[1],
                    arrivals=arrivals,
                    processing_times_orders=processing_times_orders, 
                    orders_to_families_assignment=orders_to_families_assignment,
                    family_setup_times=family_setup_times,
                    fixed_cost=0,
                    lp_relaxation=False,
                    log_console=0,
                    maximum_running_time=0)
                running_times[10, rep] = time.time() - start
                
                # Check what is the best LB and heuristic
                best_lb = np.max(gaps[[1, 3, 5, 7, 9], rep])
                for i in [1, 3, 5, 7, 9]:
                    if gaps[i, rep] >= best_lb - 0.000001:
                        best_lower_bound[i, rep] = 1

                best_heur = np.min(gaps[[2, 4, 6, 8, 10], rep])
                for i in [2, 4, 6, 8, 10]:
                    if gaps[i, rep] <= best_heur + 0.000001:
                        best_heuristic[i, rep] = 1

            # obtain our results
            gaps[:, :] = gaps[:, :] / gaps[0, :]
            redefined_gaps = np.where(gaps > 1.0001, 0, gaps)
            times_equal_benchmark = np.where(redefined_gaps >= 0.9999, 1, 0) 
            

            print('Printing the results for lower bounds in this scenario:')
            order_for_results = [0, 1, 3, 5, 7, 9]
            for number in order_for_results:
                print(f'Method {list_methods[number]} has:')
                print(f'max gap: {np.max(gaps[number, :]):.3f}; min gap {np.min(gaps[number, :]):.3f}')
                print(f'Gmean gap {gmean(gaps[number, :]):.3f}, avg run time: {np.mean(running_times[number, :]):.3f}')
                print(f'Percentage of times it meets the optimal {100*np.mean(times_equal_benchmark[number, :]):.3f}')
                print(f'Percent best lb {100*np.mean(best_lower_bound[number, :]):.3f}')
                print(f'Percent best Heuristic {100*np.mean(best_heuristic[number, :]):.3f}')
                print(f'Mean columns generated are: {np.mean(columns_generated[number, :]):.3f} \n')
            print('\n')
            
            print('Printing the results for heuristics in this scenario:')
            order_for_results = [2, 4, 6, 8, 10]
            for number in order_for_results:
                print(f'Method {list_methods[number]} has:')
                print(f'max gap: {np.max(gaps[number, :]):.3f}; min gap {np.min(gaps[number, :]):.3f}')
                print(f'Gmean gap {gmean(gaps[number, :]):.3f}, avg run time: {np.mean(running_times[number, :]):.3f}')
                print(f'Percentage of times it meets the optimal {100*np.mean(times_equal_benchmark[number, :]):.3f}')
                print(f'Percent best lb {100*np.mean(best_lower_bound[number, :]):.3f}')
                print(f'Percent best Heuristic {100*np.mean(best_heuristic[number, :]):.3f}')
                print(f'Mean columns generated are: {np.mean(columns_generated[number, :]):.3f} \n')
            print('\n')
            
            # Store our results
            save_object_in_pickle(
                object_being_saved=[replications, scenario, list_methods, gaps, 
                running_times, times_equal_benchmark, best_lower_bound, best_heuristic,
                columns_generated],
                name_file=f'results_{scenario}')



    if do_large_experiments:
        replications = 25
        number_orders = [80, 120, 160]
        number_families = [5, 10, 15]
        number_vehicles = [5, 10, 15]
        params_orders = [(1, 101)]
        params_families = [(0, 51), (0, 101)]
        params_arrivals = [(0, 10), (0, 26)]
        
        all_scenarios = itertools.product(number_orders, number_vehicles, number_families, 
            params_orders, params_families, params_arrivals)
        
        list_methods = [
            'CG Form 1 y 2 (Naive) LP', 'CG Form 1 y 2 (Naive) Heuristic',
            'Lower Bound for Families', 'Quadratic interval IP',
            'Best Heuristic']
        
        max_run_time = 120
        iteration = 0
        for scenario in all_scenarios:
            iteration += 1
            # Creating the data structures to store results for this scenario
            gaps = np.zeros(shape=(len(list_methods), replications))
            running_times = np.zeros(shape=(len(list_methods), replications))
            columns_generated = np.zeros(shape=(len(list_methods), replications))
            times_equal_benchmark = np.zeros(shape=(len(list_methods), replications))
            best_lower_bound = np.zeros(shape=(len(list_methods), replications))
            best_heuristic = np.zeros(shape=(len(list_methods), replications))
            exceeds_running_time = np.zeros(shape=(len(list_methods), replications))
            
            print(f'Starting scenario (orders, vehicles, families, (params order, params fam y params arriv))')
            print(f'{scenario = }; {iteration = }')
            for rep in tqdm(range(replications)):
                np.random.seed(rep + iteration * 10000)
                
                arrivals, family_setup_times, processing_times_orders, \
                    orders_to_families_assignment = create_input_family_setups(
                    number_orders=scenario[0],
                    number_families=scenario[2],
                    params_distribution_orders={'low': scenario[3][0], 'high': scenario[3][1]},
                    params_distribution_families={'low': scenario[4][0], 'high': scenario[4][1]},
                    params_distribution_interarrivals={'low': scenario[5][0], 'high': scenario[5][1]},
                    )
                    
                params = {'number_orders': scenario[0], 'number_families': scenario[2],
                    'number_vehicles': scenario[1],
                    'arrivals': arrivals,
                    'processing_times_orders': processing_times_orders,
                    'orders_to_families_assignment': orders_to_families_assignment,
                    'family_setup_times': family_setup_times}
                
                
                params['p_feas'] = 0.5
                
                # Colgen naive\
                gaps[0, rep], gaps[1, rep], running_times[0, rep],\
                    running_times[1, rep], columns_generated[0, rep]\
                        = colgen_naive_families(do_heuristic=True,
                        maximum_running_time=max_run_time, **params)
                
                if running_times[1, rep] >= max_run_time:
                    exceeds_running_time[1, rep] = 1
                
                # LB Improved families
                start = time.time()
                gaps[2, rep] = lower_bound_improved_families(
                    number_orders=scenario[0],
                    number_vehicles=scenario[1],
                    arrivals=arrivals, 
                    processing_times_orders=processing_times_orders, 
                    orders_to_families_assignment=orders_to_families_assignment,
                    family_setup_times=family_setup_times,
                    fixed_cost=0
                    )
                running_times[2, rep] = time.time() - start
                
                # Quadratic Interval IP
                start = time.time()
                gaps[3, rep] = naive_quadratic_interval_formulation_families(
                    number_orders=scenario[0],
                    number_vehicles=scenario[1],
                    number_families=scenario[2],
                    arrivals=arrivals,
                    processing_times_orders=processing_times_orders, 
                    orders_to_families_assignment=orders_to_families_assignment,
                    family_setup_times=family_setup_times,
                    fixed_cost=0,
                    lp_relaxation=False,
                    log_console=0,
                    maximum_running_time=max_run_time)
                running_times[3, rep] = time.time() - start
                
                if running_times[3, rep] >= max_run_time:
                    exceeds_running_time[3, rep] = 1
                    
                gaps[4, rep] = min(gaps[1, rep], gaps[3, rep])
                running_times[4, rep] = running_times[1, rep] + running_times[3, rep]
                
                best_lb = np.max(gaps[[0, 2], rep])
                for i in [0, 2]:
                    if gaps[i, rep] >= best_lb - 0.000001:
                        best_lower_bound[i, rep] = 1

                best_heur = np.min(gaps[[1, 3, 4], rep])
                for i in [1, 3, 4]:
                    if gaps[i, rep] <= best_heur + 0.000001:
                        best_heuristic[i, rep] = 1

                gaps[:, rep] = gaps[:, rep] / best_lb 
            redefined_gaps = np.where(gaps > 1.0001, 0, gaps)
            times_equal_benchmark = np.where(redefined_gaps >= 0.9999, 1, 0) 
            
            print('Printing the results for lower bounds in this scenario:')
            order_for_results = [0, 2]
            for number in order_for_results:
                print(f'Method {list_methods[number]} has:')
                print(f'max gap: {np.max(gaps[number, :]):.3f}; min gap {np.min(gaps[number, :]):.3f}')
                print(f'Gmean gap {gmean(gaps[number, :]):.3f}, avg run time: {np.mean(running_times[number, :]):.3f}')
                print(f'Percentage of times it meets the optimal {100*np.mean(times_equal_benchmark[number, :]):.3f}')
                print(f'Percent best lb {100*np.mean(best_lower_bound[number, :]):.3f}')
                print(f'Percent best Heuristic {100*np.mean(best_heuristic[number, :]):.3f}')
                print(f'Mean columns generated are: {np.mean(columns_generated[number, :]):.3f} \n')
            print('\n')
            
            print('Printing the results for heuristics in this scenario:')
            order_for_results = [1, 3, 4]
            for number in order_for_results:
                print(f'Method {list_methods[number]} has:')
                print(f'max gap: {np.max(gaps[number, :]):.3f}; min gap {np.min(gaps[number, :]):.3f}')
                print(f'Gmean gap {gmean(gaps[number, :]):.3f}, avg run time: {np.mean(running_times[number, :]):.3f}')
                print(f'Heuristic exceeds run time (%) {100*np.mean(exceeds_running_time[number, :]):.3f}')
                print(f'Percentage of times it meets the optimal {100*np.mean(times_equal_benchmark[number, :]):.3f}')
                print(f'Percent best lb {100*np.mean(best_lower_bound[number, :]):.3f}')
                print(f'Percent best Heuristic {100*np.mean(best_heuristic[number, :]):.3f}')
                print(f'Mean columns generated are: {np.mean(columns_generated[number, :]):.3f} \n')
            print('\n')
            
            save_object_in_pickle(
                object_being_saved=[replications, scenario, list_methods, gaps, 
                running_times, times_equal_benchmark, best_lower_bound, best_heuristic,
                columns_generated, exceeds_running_time],
                name_file=f'results_{scenario}_{max_run_time}_sec')
    
    
    
    
    