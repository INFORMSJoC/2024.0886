# -*- coding: utf-8 -*-
"""
@author: Ignacio Erazo
"""
from gurobipy import *
import numpy as np
import time
import copy
import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from src.tactical_design_sdd.order_arrival_functions import \
    arrivals_constant_rate, arrivals_inhomogeneous_u_shaped, \
    arrivals_inhomogeneous_early

from src.tactical_design_sdd.solution_methods import dp_concave_and_sum_and_fixed, \
    ip_naive_formulation_interval_solvable, DualCallback

from src.tactical_design_sdd.figure_functions import \
    retrieve_dispatch_times_given_list_of_dispatches, \
    plotting_dispatches_results_for_three_different_fleet_sizes

def g(x):
    """ 
    Dispatching times function based on cardinality for the SDD experiments.
    1 unit of time corresponds to 6 minutes
    """
    return 0.25*x + 4*np.sqrt(x) + 1.67

DISPATCHING_TIME_PARAMETERS_SDD_EXPERIMENT = {
    'fixed_cost': 0,
    'C1': 0,
    'C2': 0,
    # SDD relies only on g(cardinality) dispatching times 
    'C3': 1,
    'g': g
}

BASELINE_CASE_TO_INITIAL_ORDERS_ARRIVAL_PROCESS = {
    'Baseline (i)': arrivals_constant_rate,
    'Baseline (ii)': arrivals_inhomogeneous_early,
    # This is not pictured in the paper, but also an interesting scenario
    'Baseline (iii)': arrivals_inhomogeneous_u_shaped
}

# For the expected time between orders, each unit is 6 minutes; i.e. 2 units = 12 min
BASELINE_CASE_TO_INITIAL_ORDERS_PARAMETERS = {
    # Baseline (i) has no parameters beyond number of orders
    'Baseline (i)': {'num_orders': 50},
    'Baseline (ii)': {'num_orders': 50,
                      'num_orders_early': 20,
                      'expected_time_between_orders_early': 2,
                      'expected_time_between_orders_late': 1/3},
    'Baseline (iii)': {'num_orders': 50,
                      'num_orders_early': 15,
                      'expected_time_between_orders_early': 1/3,
                      'num_orders_mid_day': 20,
                      'expected_time_between_orders_mid_day': 2,
                      'expected_time_between_orders_late': 1/3}
}

# Changing this parameter modifies the economies of scale of orders that
# arrive after 2 PM. We kept it as constant for our experiments in the paper
# but it can be any arbitrary constant > 0.
# As mentioned in the definition of g(x), one unit of time is equal to 6 minutes.
EXPECTED_TIME_BETWEEN_ORDERS_AFTER_2_PM = 1

# To get some deeper logs based on our functions and the experiment
PRINT_TO_CONSOLE_PARTIAL_RESULTS = False

# If we desire to replicate the figure of the paper
OBTAIN_PAPER_FIGURE = True
PATH_TO_STORE_FIGURES = "results/figures_sdd_obtained_by_script/"

if __name__ == '__main__':
    print("Starting the SDD experiment for scenario:")

    # Create directory where we store results, if desired
    if OBTAIN_PAPER_FIGURE:
        os.makedirs(name=PATH_TO_STORE_FIGURES, exist_ok=True)

    # Setup the baseline case to run
    run_baseline_i = True
    scenario_run = ''
    if run_baseline_i:
        arrival_process = BASELINE_CASE_TO_INITIAL_ORDERS_ARRIVAL_PROCESS['Baseline (i)']
        params_arrival_process = BASELINE_CASE_TO_INITIAL_ORDERS_PARAMETERS['Baseline (i)']
        scenario_run += 'Baseline (i)'
    else:
        arrival_process = BASELINE_CASE_TO_INITIAL_ORDERS_ARRIVAL_PROCESS['Baseline (ii)']
        params_arrival_process = BASELINE_CASE_TO_INITIAL_ORDERS_PARAMETERS['Baseline (ii)']
        scenario_run += 'Baseline (ii)'
    
    
    # Compute the arrivals based on the arrival proocess
    order_arrival_times = arrival_process(**params_arrival_process)
    num_orders = params_arrival_process['num_orders']

    # No need to have individual dispatching times given we only have a 
    # subset cardinality component (see g(x))
    individual_dispatch_times = [0 for _ in range(num_orders)]

    # We solve optimally the problem for a single vehicle, and record
    # the optimal makespan of such solution
    desired_makespan, optimal_dispatches_per_vehicle, cardinality_dispatches, \
        start_and_end_for_dispatches = \
            dp_concave_and_sum_and_fixed(
                order_arrival_times,
                individual_dispatch_times,
                need_to_debug=False,
                **DISPATCHING_TIME_PARAMETERS_SDD_EXPERIMENT)
    
    # Transform into list of lists
    for i in range(0, len(start_and_end_for_dispatches)):
        list_version = []
        for j in range(0, len(start_and_end_for_dispatches[i])):
            list_version.append(start_and_end_for_dispatches[i][j])
        start_and_end_for_dispatches[i] = list_version[:]
    
    if PRINT_TO_CONSOLE_PARTIAL_RESULTS:
        print('Original makespan for 1 vehicle is: ' + str(desired_makespan))
    
    # We initialize the parameters for the tests on multiple vehicles
    dualcallback = DualCallback(makespan_to_compare_dual_bound_against=desired_makespan)
    
    all_params_for_ip = {key: value for key, value \
                         in DISPATCHING_TIME_PARAMETERS_SDD_EXPERIMENT.items()}
    
    # We parameters from period runs; this is useful if we want to study the
    # efficiency of a dispatching fleet (in terms of distance driven), given a 
    # maximum (feasible) makespan and a set of orders to serve
    params_last_run = None

    # Useful if we want to reproduce paper results, including the SDD figures
    vehicles_to_optimal_solution_for_plotting = {
        1: [start_and_end_for_dispatches, 
            cardinality_dispatches, 
            len(order_arrival_times)]
    }

    initial_z_variables_solution = []

    # We can solve for as many vehicles as we want
    for num_vehicles in range(2, 5): 
        print(f'Starting optimization procedure for {num_vehicles} vehicles and {num_orders} orders')

        # This boolean will make sure we dispatch as many orders as we can
        # given some number of vehicles
        solve_optimization = True
        while solve_optimization:
            # Update the parameters for this run
            all_params_for_ip['number_of_orders'] = num_orders
            all_params_for_ip['order_arrival_times'] = order_arrival_times
            all_params_for_ip['individual_dispatch_times'] = individual_dispatch_times
            all_params_for_ip['callback'] = dualcallback
            all_params_for_ip['number_of_vehicles'] = num_vehicles
            all_params_for_ip['desired_makespan'] = desired_makespan
            all_params_for_ip['initial_z_variables_solution'] = initial_z_variables_solution
            
            # optimize
            start=time.time()
            current_makespan, initial_z_variables_solution, optimal_dispatches_per_vehicle = ip_naive_formulation_interval_solvable(
                print_debug_to_console=PRINT_TO_CONSOLE_PARTIAL_RESULTS,
                **all_params_for_ip)
            end=time.time()

            if PRINT_TO_CONSOLE_PARTIAL_RESULTS:
                print('Finished optimization with '+str(num_orders) + ' orders and makespan ' + str(current_makespan) +' and took (s) '+str(end-start))
                print('')

            # If we can dispatch to all orders with this number of vehicles, 
            # then we increase the number of orders
            if current_makespan < desired_makespan:
                # We update the last run parameters before preparing the
                # parameters for the next run
                params_last_run = copy.deepcopy(all_params_for_ip)
                params_last_run['minimize_total_dispatching_time'] = True

                # We append a new order to the next run, since we know we can
                # deliver to the current number of orders within desired makespan
                order_arrival_times.append(order_arrival_times[-1] + EXPECTED_TIME_BETWEEN_ORDERS_AFTER_2_PM)
                individual_dispatch_times.append(0)
                num_orders += 1
                
            # Otherwise we study the efficiency (distance driven and dispatches
            # structure) of serving these orders given these orders and their arrivals
            else:
                if PRINT_TO_CONSOLE_PARTIAL_RESULTS:
                    print(f'{num_vehicles} vehicles could not dispatch to {len(order_arrival_times)} orders within makespan {desired_makespan}')
                    print('')
                    print('Now solving the IP that minimizes distance, for the maximum achievable',
                        f'number of orders that can be done with {num_vehicles} vehicles under makespan {desired_makespan}')
                # We study efficiency by using the last parameters (params_last_run)
                # known to be feasible for this number of vehicles
                total_distance_driven, sol, optimal_dispatches_per_vehicle = ip_naive_formulation_interval_solvable(
                    print_debug_to_console=PRINT_TO_CONSOLE_PARTIAL_RESULTS,
                    **params_last_run)
                
                previous_order_arrival_times = params_last_run['order_arrival_times']

                # We compute our dispatches' data for plotting
                start_and_end_for_dispatches, cardinality_of_dispatches \
                    = retrieve_dispatch_times_given_list_of_dispatches(
                        number_of_vehicles=num_vehicles,
                        makespan_to_consider=desired_makespan,
                        g_dispatch_times_function=g,
                        order_arrival_times=previous_order_arrival_times,
                        batches_dispatched_by_each_vehicle=optimal_dispatches_per_vehicle)

                vehicles_to_optimal_solution_for_plotting[num_vehicles] = [
                    start_and_end_for_dispatches, 
                    cardinality_of_dispatches, 
                    len(previous_order_arrival_times)]
                
                # By switching this flag, we will pass to the next iteration
                # increasing the number of vehicles (as we cannot serve all the
                # orders with the current number of vehicles)
                solve_optimization = False


    # We can plot the results for any three different fleet sizes. 
    # This function assumes the first cardinality will be the single vehicle,
    # and is optimized (visually-wise) for cardinalities 1, 2 and 4.
    # The paper figures were made by hard-coding the values, so the lines across
    # each fleet were longer at the bottom. This figure will have the same lines
    # they will just not be sorted (making visuals a bit worse in our opinion)
    # but the solution is the same.
    if OBTAIN_PAPER_FIGURE:
        verts = plotting_dispatches_results_for_three_different_fleet_sizes(
            order_arrival_times=order_arrival_times,
            schedule_dispatches_fleet_size_1=vehicles_to_optimal_solution_for_plotting[1][0],
            cardinality_of_dispatches_fleet_size_1=vehicles_to_optimal_solution_for_plotting[1][1],
            number_of_orders_done_with_fleet_size_1=vehicles_to_optimal_solution_for_plotting[1][2],
            fleet_size_1=1,
            schedule_dispatches_fleet_size_2=vehicles_to_optimal_solution_for_plotting[2][0],
            cardinality_of_dispatches_fleet_size_2=vehicles_to_optimal_solution_for_plotting[2][1],
            number_of_orders_done_with_fleet_size_2=vehicles_to_optimal_solution_for_plotting[2][2],
            fleet_size_2=2,
            schedule_dispatches_fleet_size_3=vehicles_to_optimal_solution_for_plotting[4][0],
            cardinality_of_dispatches_fleet_size_3=vehicles_to_optimal_solution_for_plotting[4][1],
            number_of_orders_done_with_fleet_size_3=vehicles_to_optimal_solution_for_plotting[4][2],
            fleet_size_3=4,
            makespan_for_figure=desired_makespan,
            start_of_day=9, # day in experiment starts at 9 am
            unit_arrivals_to_hours=0.1, # 1 unit is exactly 0.1 hr = 6 minutes.
            name=PATH_TO_STORE_FIGURES + scenario_run)
