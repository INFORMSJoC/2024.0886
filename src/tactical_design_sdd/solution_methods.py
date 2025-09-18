# -*- coding: utf-8 -*-
"""
@author: Ignacio Erazo
"""
from gurobipy import *
import numpy as np
import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

EPSILON = 0.0001

class DualCallback():
    def __init__(self, 
            makespan_to_compare_dual_bound_against: float
            ) -> None:
        self.dual_bound = makespan_to_compare_dual_bound_against
        return
    
    def __call__(self, model, where):
        if where == GRB.Callback.MIPNODE:
            val = model.cbGet(GRB.Callback.MIPNODE_OBJBND)
            if val > self.dual_bound:
                model.terminate()
        return

def dp_concave_and_sum_and_fixed(
        order_arrival_times: list[float],
        individual_dispatch_times: list[float],
        g: callable,
        fixed_cost: float,
        C1: float,
        C2: float,
        C3: float,
        need_to_debug: bool=False
        ) -> tuple[float, list[float], list[tuple[float, float]]]:
    """
    This function returns the optimal FIFO solution when the dispatching times
    function for a subset S of orders is:
    f(S) = C1 * \sum(i in S) individual_dispatch_times_i 
                + C2 * \max(i in S) individual_dispatch_times_i
                + C3 * g(cardinality of S)
                + fixed_cost
    
    We specifically return:
        - optimal makespan
        - the intervals representing the orders covered by each dispatch
        - number of orders in each of the dispatches of the optimal solution
        - list with each entry being the (start, end) of a dispatch in optimal solution
    """
    # Initialize utilities and our matrix for optimal FIFO solution makespan
    number_of_orders = len(order_arrival_times)
    cardinality_utility = np.ones(number_of_orders)
    optimal_fifo_dp_objectives = np.zeros((number_of_orders, number_of_orders))
    
    # Obtain the solutions for border condition with 1 dispatch of orders [0 to i]
    for i in range(0, number_of_orders):
        optimal_fifo_dp_objectives[0, i] = order_arrival_times[i]\
            + C1 * (sum(individual_dispatch_times[0: i+1])) \
            + C2 * (max(individual_dispatch_times[0: i+1])) \
            + C3 * (g(sum(cardinality_utility[0: i+1]))) \
            + fixed_cost
        
    # Solve for all the other entries
    for i in range(1, number_of_orders):
        for j in range(i, number_of_orders):
            optimal_fifo_dp_objectives[i,j] = \
                max(order_arrival_times[j], min(optimal_fifo_dp_objectives[0:i, i-1])) \
                    + C1 * (sum(individual_dispatch_times[i: j+1])) \
                    + C2 * (max(individual_dispatch_times[i: j+1])) \
                    + C3 * g(sum(cardinality_utility[i: j+1])) \
                    + fixed_cost
    
    # Retrieve the solution makespan
    makespan = min(optimal_fifo_dp_objectives[:, number_of_orders-1])
    
    # Retrieve the path (i.e. optimal solution) that achieves such makespan
    start_of_dispatch_indices = []
    values = [min(optimal_fifo_dp_objectives[:, number_of_orders-1])]
    # find the start order of the last dispatch
    index = np.argmin(optimal_fifo_dp_objectives[:, number_of_orders-1])
    start_of_dispatch_indices.append(index)
    while (index > 0):
        # find the start order of the previous dispatch
        index2 = np.argmin(optimal_fifo_dp_objectives[0:index, index-1])
        # insert the objective value at the end of such dispatch
        values.insert(0, min(optimal_fifo_dp_objectives[0:index, index-1]))
        index = index2
        # Add the start order of the dispatch into our list of dispatches for opt solution
        start_of_dispatch_indices.insert(0, index)
    
    # Create the list of intervals representing what the optimal solution looks like
    # in terms of what order is in what order in in what dispatch
    intervals = []
    for i in range(0, len(start_of_dispatch_indices)-1):
        intervals.append([start_of_dispatch_indices[i], start_of_dispatch_indices[i+1] - 1])
    # Add the final dispatch that ends with the final order
    intervals.append([start_of_dispatch_indices[len(start_of_dispatch_indices) - 1], number_of_orders - 1])
    
    # Create the time intervals needed for the dispatches
    # Initialize start
    dispatching_time_for_dispatches = [
        # Start of dispatch
        (order_arrival_times[intervals[0][1]], 
         order_arrival_times[intervals[0][1]] \
            + C1 * (sum(individual_dispatch_times[intervals[0][0]: intervals[0][1]+1])) \
            + C2 * (max(individual_dispatch_times[intervals[0][0]: intervals[0][1]+1])) \
            + C3 * g(1 + intervals[0][1] - intervals[0][0]) \
            + fixed_cost )]

    # Build up the next time intervals for the dispatches
    partial_max = dispatching_time_for_dispatches[0][1]
    for i in range(1, len(intervals)):
        dispatching_time_for_dispatches.append( 
            (max(order_arrival_times[intervals[i][1]], partial_max), 
             max(order_arrival_times[intervals[i][1]], partial_max) \
                + C1 * (sum(individual_dispatch_times[intervals[i][0]: intervals[i][1]+1])) \
                + C2 * (max(individual_dispatch_times[intervals[i][0]: intervals[i][1]+1])) \
                + C3 * g(1 + intervals[i][1] - intervals[i][0]) \
                + fixed_cost \
            )
        )
        # This is exactly equal to the last value we appended, second entry of the tuple
        partial_max = dispatching_time_for_dispatches[i][1]

    # Compute the cardinality of each of the dispatches
    cardinality_of_dispatches = []
    for i in range(0,len(intervals)):
        cardinality_of_dispatches.append(1 + intervals[i][1] - intervals[i][0])
    
    # Printing info to console, if needed
    if need_to_debug:
        print(start_of_dispatch_indices)
        print(intervals)
        print(dispatching_time_for_dispatches)
        print(cardinality_of_dispatches)
    
    # Transforming data into integers, instead of numpy integers
    for dispatch in range(len(intervals)):
        for index in range(len(intervals[dispatch])):
            intervals[dispatch][index] = int(intervals[dispatch][index])
    return makespan, intervals, cardinality_of_dispatches, dispatching_time_for_dispatches


def ip_naive_formulation_interval_solvable(
        number_of_orders: int,
        order_arrival_times: list[float],
        individual_dispatch_times: list[float],
        g: callable,
        callback: callable,
        fixed_cost: float,
        C1: float,
        C2: float,
        C3: float,
        number_of_vehicles: int,
        desired_makespan: float,
        initial_z_variables_solution: list=[],
        log_progress_to_console: int=0,
        print_debug_to_console: bool=False,
        minimize_total_dispatching_time: bool=False
        ):
    """
    This function implements formulation 1 of the paper. It solves the problem
    when only considering interval batches; so optimality is only guaranteed
    for interval-solvable functions.
    """
    # indexing the orders with paper notation
    order_indices = [order_number for order_number in range(1, number_of_orders+1)]
    
    # We create all the interval-type subsets and compute their dispatching time
    interval_subset_to_dispatching_time = dict()
    for i in range(0, number_of_orders):
        summed_time = 0
        max_time = 0
        for j in range(i, number_of_orders):
            # Update two of our dispatching time components
            summed_time += individual_dispatch_times[j]
            max_time = max(max_time, individual_dispatch_times[j])

            # We index by i+1 and j+1 because that corresponds to the max and
            # min order of the subset [i+1, j+1], which are the orders_indices for i, j
            interval_subset_to_dispatching_time[(i+1, j+1)] = C1 * summed_time \
                + C2 * max_time \
                + C3 * g(j - i + 1) \
                + fixed_cost

    # Creating the tuples that will define (subset, vehicle) variables
    subset_to_vehicle_variable_keys = list()
    for i in interval_subset_to_dispatching_time.keys():
        for j in range(0, number_of_vehicles):
            subset_to_vehicle_variable_keys.append((i, j))

    # Creating the gurobi Model
    model = Model()
    model.setParam("LogToConsole", log_progress_to_console)
    
    # Defining the indices for the t_{} variables
    indices_t_variables = []
    for i in range(0, number_of_orders):
        for k in range(0, number_of_vehicles):
            # order i, vehicle k = t_{i,k}
            indices_t_variables.append((i+1, k))
    # last variable to determine the makespan of each vehicle
    for k in range(0, number_of_vehicles):
        indices_t_variables.append((number_of_orders+1, k))
    # the overall makespan of the problem, >= to makespan of every vehicle
    indices_t_variables.append(number_of_orders+2)
    # Adding the variables
    t = model.addVars(indices_t_variables, name="time_dispatch")
    
    # Adding binary variables z for each (subset, vehicle) tuple
    z = model.addVars(subset_to_vehicle_variable_keys,
                        vtype=GRB.BINARY,
                        name="z")
    
    # In this scenario we know the desired makespan can be achieved for this
    # number_of_orders and number_of_vehicles, so we minimize distance driven
    if minimize_total_dispatching_time:
        objective = LinExpr()
        for subset_vehicle_tuple in subset_to_vehicle_variable_keys:
            objective += z[subset_vehicle_tuple] \
                * interval_subset_to_dispatching_time[subset_vehicle_tuple[0]]
        model.setObjective(objective, GRB.MINIMIZE) 
        # The new solution is feasible with makespan, we need to enforce it
        model.addConstr(t[number_of_orders+2] <= desired_makespan)

    # Otherwise if unknown, we minimize the makespan
    else:
        # Setting the makespan minimization objective
        model.setObjective(t[number_of_orders+2], GRB.MINIMIZE)  
        # We stop as soon as we know makespan is feasible
        model.params.BestObjStop = desired_makespan + EPSILON
    
    # Adding Constraints
    for i in range(0, number_of_orders):
        # Constraint on departure times t(i,k)>= r(i)
        for k in range(0, number_of_vehicles):
            model.addConstr(t[(i+1, k)] >= order_arrival_times[i]) 

            # Depart after the previous time t(i+1,k) >= t(i,k) + dispatches done by vehicle k
            # We add all in the left hand side
            constraint = LinExpr()
            for j in subset_to_vehicle_variable_keys:
                if j[0][1] == i+1 and j[1] == k:
                    constraint -= z[j] * interval_subset_to_dispatching_time[j[0]]
            constraint += t[(i+2,k)] - t[(i+1,k)]
            model.addConstr(constraint >= 0)

        # Covering Constraint \sum over all vehicles and subsets that cover order i >= 1
        constraint = LinExpr()
        # Index of constraints starts at i+1 because of order_indices
        for j in subset_to_vehicle_variable_keys:
            # This is implies order i is included in the dispatch, as inside first and last
            if i+1 >= j[0][0] and i+1 <= j[0][1]:
                constraint += z[j]
        model.addConstr(constraint == 1)
        

    # Contraint that says makespan >= makespan_vehicle[i]
    for k in range(0, number_of_vehicles):
        model.addConstr(t[number_of_orders+2] - t[(number_of_orders+1, k)] >= 0)
    
    if len(initial_z_variables_solution) > 0:
        total_variables_initialized = 0
        for k in initial_z_variables_solution:
            if k in subset_to_vehicle_variable_keys:
                z[k].Start = 1
                total_variables_initialized += 1
        if total_variables_initialized and print_debug_to_console:
            print('Succesfully initialized the given solution')
    else:
        if print_debug_to_console:
            print("There was no need to initialize a solution")


    if minimize_total_dispatching_time:
        model.optimize()
    else:
        # Optimize Model with the dualcallback, ensuring we can speed up the process
        # as the model will terminate early if the desired makespan cannot be met.
        # This does not apply to dispatching time minimization
        model.optimize(callback)

    # data structure to learn about the solution per vehicle
    optimal_dispatches_per_vehicle = []
    for vehicle in range(number_of_vehicles):
        optimal_dispatches_per_vehicle.append([])

    # optimal solution structure; can be used to initialize again the problem
    optimal_dispatches_solution = []

    for i in model.getVars():
        # Search for the dispatch decision variables that were selected
        if i.x > 0 + EPSILON and i.varname[0] == 'z':
            # Parsing the string name
            string = i.varname[2:-1]
            string = string.split(',')
            
            # Adding the tuple that is ((first_order_index, last_order_included), vehicle)
            # this is useful to solve again this problem as starting solution
            dispatch_tuple = ((int(string[0][1:]) , int(string[1][:-1])) , int(string[2]))
            optimal_dispatches_solution.append(dispatch_tuple)

            # This is useful to visualize what happens with each vehicle
            dispatch_tuple = [int(string[0][1:]) , int(string[1][:-1])]
            optimal_dispatches_per_vehicle[int(string[2])].append(dispatch_tuple)

    if print_debug_to_console:
        print(optimal_dispatches_per_vehicle)
    
    # retrieve the optimal makespan
    OBJ = model.getObjective()
    makespan = OBJ.getValue()
    
    # return the objective and the optimal dispatches solution
    return makespan, optimal_dispatches_solution, optimal_dispatches_per_vehicle