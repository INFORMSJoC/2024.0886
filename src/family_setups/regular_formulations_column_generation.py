# -*- coding: utf-8 -*-
"""
@author: Ignacio Erazo
"""
from __future__ import annotations
import gurobipy as gp
import numpy as np
import time

MAX_ITERATIONS_COLGEN = 250000
TOLERANCE_DUAL_PRIMAL_COLGEN = 0.0001

from src.family_setups.tool_functions import _compute_routing_time_families, \
    _initialize_singletons_families

from src.family_setups.create_variables_and_constraints import _create_variables_constraints_naive_formulation, \
    _create_variables_constraints_large_scale_naive_formulation, \
    _create_variables_constraints_low_symmetry_formulation, \
     _create_variables_constraints_flow_formulation

from src.family_setups.retrieve_dual_variables import _retrieve_dual_information_naive_formulation, \
    _retrieve_dual_information_low_symmetry_formulation, \
    _retrieve_dual_information_flow_formulation


""" Column generations with heuristics included """
def colgen_naive_families(
        number_orders: int,
        number_vehicles: int,
        number_families: int,
        arrivals: np.ndarray, 
        processing_times_orders: np.ndarray, 
        orders_to_families_assignment: np.ndarray,
        family_setup_times: np.ndarray,
        fixed_cost: int=0,
        log_console: int=0,
        do_heuristic: bool=True,
        p_feas: float=0.8,
        maximum_running_time: float|bool=False
        ):
    start_lp = time.time()
    # Will be useful to find orders within families when solving pricing problem
    dictionary_families_to_orders = {
        family: [index for index in range(number_orders) if \
        orders_to_families_assignment[index] == family] \
        for family in range(number_families)}
    
    # Initialize the problem
    mapping_orders_to_subsets_they_appear, mapping_max_order_to_subsets, \
        mapping_subsets_to_routing_times = _initialize_singletons_families(
        number_orders=number_orders,
        orders_to_families_assignment=orders_to_families_assignment,
        family_setup_times=family_setup_times,
        processing_times_orders=processing_times_orders,
        fixed_cost=fixed_cost
        )
    
    """ creating the LP and setting parameters """   
    model_lp = gp.Model()
    model_lp.setParam("LogToConsole", log_console)
    model_lp.setParam('MIPFocus', 2) 
    model_lp.setParam('Symmetry', 2)
    
    """ add constraints and variables """ 
    _create_variables_constraints_naive_formulation(model=model_lp,
        number_vehicles=number_vehicles, number_orders=number_orders, arrivals=arrivals,
        mapping_subsets_to_routing_times=mapping_subsets_to_routing_times,
        mapping_max_order_to_subsets=mapping_max_order_to_subsets,
        mapping_orders_to_subsets_they_appear=mapping_orders_to_subsets_they_appear,
        lp_relaxation=True
        )
    
    """ optimize master problem for the first time """
    model_lp.optimize()
    obj = model_lp.getObjective()
    objective_value_lp = obj.getValue()
    
    # Obtain the dual information
    alpha, beta, gamma = _retrieve_dual_information_naive_formulation(model_lp)
    alpha_in = np.zeros(shape=(number_orders, number_vehicles)) #i,k
    beta_in = np.zeros(shape=(number_orders, number_vehicles))  #i,k
    gamma_in = np.zeros(number_orders) #i
    arriv_in = np.reshape(arrivals, newshape=(number_orders, 1))
    objective_dual = np.sum(gamma_in) + np.sum(arriv_in*alpha_in)
    
    # Start column generation
    tolerance_eps = TOLERANCE_DUAL_PRIMAL_COLGEN
    max_iterations = MAX_ITERATIONS_COLGEN
    iterations = 0
    columns_added_in_pricing = set()
    # Termination criteria when our dual and primal bounds are equal
    while(abs(objective_value_lp-objective_dual) > tolerance_eps and\
        iterations <= max_iterations):
        iterations += 1
        if iterations % 2500 == 0:
            print(f'{iterations = } and {objective_value_lp = } and {objective_dual = }')
            
        alpha_prime = np.asarray(alpha).reshape((number_orders, number_vehicles))
        beta_prime = np.asarray(beta).reshape((number_orders, number_vehicles))
        gamma_prime = np.asarray(gamma)

        # Set our variables for the separation problem
        alpha_sep = p_feas*alpha_in + (1-p_feas)*alpha_prime
        beta_sep = p_feas*beta_in + (1-p_feas)*beta_prime
        gamma_sep = p_feas*gamma_in + (1-p_feas)*gamma_prime

        # Solve pricing problem
        best_objective = 0.0000001
        subset_to_enter = None
        vehicle_use = None
        for order in range(number_orders):
            # find smallest beta because we substract and seek the max value
            vehicle_max = np.argmin(beta_sep[order, :])
            beta_to_use = beta_sep[order, vehicle_max]
            order_list = {order}
            family_order = orders_to_families_assignment[order]
            setup_family = family_setup_times[family_order]
            gamma_objective = gamma_sep[order]
            routing_objective = setup_family + \
                processing_times_orders[order] + fixed_cost
            for order_j in dictionary_families_to_orders[family_order]:
                if order_j >= order:
                    break
                else:
                    if beta_to_use*processing_times_orders[order_j] \
                            < gamma_sep[order_j]:
                        order_list.add(order_j)
                        routing_objective += processing_times_orders[order_j]
                        gamma_objective += gamma_sep[order_j]
            
            total_objective = gamma_objective - beta_to_use*routing_objective
            if total_objective > best_objective:
                subset_to_enter = order_list
                best_objective = total_objective
                vehicle_use = vehicle_max

        # If we have a valid variable to enter, then we add it to the problem
        if subset_to_enter != None and \
                (tuple(sorted(subset_to_enter)), vehicle_use) not in \
                columns_added_in_pricing:
            subset_to_enter = sorted(subset_to_enter)
            columns_added_in_pricing.add((tuple(sorted(subset_to_enter)), vehicle_use))
            family_order = orders_to_families_assignment[subset_to_enter[-1]]
            routing_time = _compute_routing_time_families(
                order_list=subset_to_enter,
                processing_times_orders=processing_times_orders,
                family_setup=family_setup_times[family_order],
                fixed_cost=fixed_cost
                )
            
            # Add to constraints
            constraints_where_enters = [
                model_lp.getConstrByName(f'const1d[{order}]') for order in \
                subset_to_enter]
            coefficients_constraints = [1 for order in subset_to_enter]
            if subset_to_enter[-1] == number_orders-1:
                constraints_where_enters.append(
                        model_lp.getConstrByName(f'const1c[{vehicle_use}]'))
                coefficients_constraints.append(-routing_time)
            else:
                constraints_where_enters.append(model_lp.getConstrByName(
                    f'const1b_order{subset_to_enter[-1]}[{vehicle_use}]'))
                coefficients_constraints.append(-routing_time)
            
            # Register with our maps
            tupled = tuple(subset_to_enter)
            mapping_subsets_to_routing_times[tupled] = routing_time
            mapping_max_order_to_subsets[subset_to_enter[-1]].add(tupled)
            for order in subset_to_enter:
                mapping_orders_to_subsets_they_appear[order].add(tupled)
            
            model_lp.addVar(obj=0, name=f'x_{iterations}', 
                column=gp.Column(coefficients_constraints, constraints_where_enters))
            
            # Optimize the problem again and retrieve the dual information
            model_lp.optimize()
            obj = model_lp.getObjective()
            objective_value_lp = obj.getValue()
            alpha, beta, gamma = \
                _retrieve_dual_information_naive_formulation(model_lp)
                
        # Otherwise, alpha_sep was feasible and we adjust our alpha_in
        else:
            alpha_in = alpha_sep
            beta_in = beta_sep
            gamma_in = gamma_sep
            objective_dual = np.sum(gamma_in) + np.sum(arriv_in*alpha_in)
    
    end_lp = time.time()
    
    # If desired, we solve the problem heuristically with all the columns generated
    if do_heuristic:
        model_heuristic = gp.Model()
        model_heuristic.setParam("LogToConsole", log_console)
        model_heuristic.setParam('MIPFocus', 2) 
        model_heuristic.setParam('Symmetry', 2)
        
        if maximum_running_time:    
            model_heuristic.setParam('TimeLimit', maximum_running_time)      
        
        """ add constraints and variables """ 
        _create_variables_constraints_large_scale_naive_formulation(model=model_heuristic,
            number_vehicles=number_vehicles, number_orders=number_orders, arrivals=arrivals,
            mapping_subsets_to_routing_times=mapping_subsets_to_routing_times,
            mapping_max_order_to_subsets=mapping_max_order_to_subsets,
            mapping_orders_to_subsets_they_appear=mapping_orders_to_subsets_they_appear,
            lp_relaxation=False
            )
        
        """ optimize master problem for the first time """
        model_heuristic.optimize()
        obj = model_heuristic.getObjective()
        objective_value_heuristic = obj.getValue()
        
        end_heuristic = time.time()
        
        runtime_lp = end_lp - start_lp
        runtime_heur = end_heuristic - end_lp
        columns_generated = len(mapping_subsets_to_routing_times)
        
        return objective_value_lp, objective_value_heuristic, \
            runtime_lp, runtime_heur, columns_generated
    return objective_value_lp, -1

def colgen_low_symmetry_families(
        number_orders: int,
        number_vehicles: int,
        number_families: int,
        arrivals: np.ndarray, 
        processing_times_orders: np.ndarray, 
        orders_to_families_assignment: np.ndarray,
        family_setup_times: np.ndarray,
        fixed_cost: int=0,
        log_console: int=0,
        do_heuristic: bool=True,
        heuristic_formulation_1: bool=False,
        p_feas: float=0.8,
        maximum_running_time: float|bool=False
        ):
    start_lp = time.time()
    # Will be useful to find orders within families when doing pricing
    dictionary_families_to_orders = {
        family: [index for index in range(number_orders) if \
        orders_to_families_assignment[index] == family] \
        for family in range(number_families)}
    
    # Initialize the problem
    mapping_orders_to_subsets_they_appear, mapping_max_order_to_subsets, \
        mapping_subsets_to_routing_times = _initialize_singletons_families(
        number_orders=number_orders,
        orders_to_families_assignment=orders_to_families_assignment,
        family_setup_times=family_setup_times,
        processing_times_orders=processing_times_orders,
        fixed_cost=fixed_cost
        )
    
    """ creating the LP and setting parameters """   
    model_lp = gp.Model()
    model_lp.setParam("LogToConsole", log_console)
    model_lp.setParam('MIPFocus', 2)
    model_lp.setParam('Symmetry', 2)
    
    if maximum_running_time:    
        model_lp.setParam('TimeLimit', maximum_running_time)      
    
    """ add constraints and variables """ 
    _create_variables_constraints_low_symmetry_formulation(model=model_lp,
        number_vehicles=number_vehicles, number_orders=number_orders, arrivals=arrivals,
        mapping_subsets_to_routing_times=mapping_subsets_to_routing_times,
        mapping_max_order_to_subsets=mapping_max_order_to_subsets,
        mapping_orders_to_subsets_they_appear=mapping_orders_to_subsets_they_appear,
        fixed_cost=fixed_cost,
        processing_times_orders=processing_times_orders,
        family_setup_times=family_setup_times,
        lp_relaxation=True
        )
    
    """ optimize master problem for the first time """
    model_lp.optimize()
    obj = model_lp.getObjective()
    objective_value_lp = obj.getValue()
    
    # Obtain the dual information
    alpha, beta, epsilon, pi, gamma = \
        _retrieve_dual_information_low_symmetry_formulation(model_lp)
    
    """ 
    Both indices for alpha and beta come out orders as:
    order i, vehicle 0; order i vehicle 1, ... order i vehicle n,
    order i+1, vehicle 0; ... 
    """        
    alpha_in = np.zeros(shape=(number_orders, number_vehicles)) #i,k
    gamma_in = np.zeros(number_orders) #i
    epsilon_in = np.zeros(number_orders) #i
    pi_in = np.zeros(number_orders)
    arriv_in = np.reshape(arrivals,newshape=(number_orders,1))
    objective_dual = np.sum(gamma_in + pi_in) + np.sum(arriv_in*alpha_in)
    
    # Start column generation
    tolerance_eps = TOLERANCE_DUAL_PRIMAL_COLGEN
    max_iterations = MAX_ITERATIONS_COLGEN
    iterations = 0
    columns_added_in_pricing = set()
    # Termination criteria when our dual and primal bounds are equal
    while(abs(objective_value_lp-objective_dual) > tolerance_eps and\
            iterations <= max_iterations):
        iterations += 1
        if iterations % 2500 == 0:
            print(f'{iterations = } and {objective_value_lp = } and {objective_dual = }')
            
        alpha_prime = np.asarray(alpha).reshape((number_orders, number_vehicles))
        gamma_prime = np.asarray(gamma)
        epsilon_prime = np.asarray(epsilon)
        pi_prime = np.asarray(pi)

        # Set our variables for the separation problem
        alpha_sep = p_feas*alpha_in + (1-p_feas)*alpha_prime
        gamma_sep = p_feas*gamma_in + (1-p_feas)*gamma_prime
        epsilon_sep = p_feas*epsilon_in + (1-p_feas)*epsilon_prime
        pi_sep = p_feas*pi_in + (1-p_feas)*pi_prime

        # Solve pricing problem
        best_objective = 0.0000001
        subset_to_enter = None
        for order in range(number_orders):
            order_list = {order}
            family_order = orders_to_families_assignment[order]
            setup_family = family_setup_times[family_order]
            gamma_objective = gamma_sep[order]
            routing_objective = setup_family + \
                processing_times_orders[order] + fixed_cost
            for order_j in dictionary_families_to_orders[family_order]:
                if order_j >= order:
                    break
                else:
                    if epsilon_sep[order]*processing_times_orders[order_j] \
                            < gamma_sep[order_j]:
                        order_list.add(order_j)
                        routing_objective += processing_times_orders[order_j]
                        gamma_objective += gamma_sep[order_j]
            
            total_objective = gamma_objective - epsilon_sep[order]*routing_objective
            if total_objective > best_objective:
                subset_to_enter = order_list
                best_objective = total_objective
        
        # If we have a valid variable to enter, then we add it to the problem
        if subset_to_enter != None and tuple(sorted(subset_to_enter)) not in \
                columns_added_in_pricing:
            subset_to_enter = sorted(subset_to_enter)
            columns_added_in_pricing.add(tuple(sorted(subset_to_enter)))
            family_order = orders_to_families_assignment[subset_to_enter[-1]]
            routing_time = _compute_routing_time_families(
                order_list=subset_to_enter,
                processing_times_orders=processing_times_orders,
                family_setup=family_setup_times[family_order],
                fixed_cost=fixed_cost
                )
            
            # Add to constraints
            constraints_where_enters = [
                model_lp.getConstrByName(f'const2g[{order}]') for order in \
                subset_to_enter]
            coefficients_constraints = [1 for order in subset_to_enter]
            constraints_where_enters.append(
                model_lp.getConstrByName(f'const2d[{subset_to_enter[-1]}]'))
            coefficients_constraints.append(-routing_time)
            
            # Add to constraints
            tupled = tuple(subset_to_enter)
            mapping_subsets_to_routing_times[tupled] = routing_time
            mapping_max_order_to_subsets[subset_to_enter[-1]].add(tupled)
            for order in subset_to_enter:
                mapping_orders_to_subsets_they_appear[order].add(tupled)
            
            model_lp.addVar(obj=0, name=f'x_{iterations}', 
                column=gp.Column(coefficients_constraints, constraints_where_enters))
            
            # Optimize the problem again and retrieve the dual information
            model_lp.optimize()
            obj = model_lp.getObjective()
            objective_value_lp = obj.getValue()
            alpha, beta, epsilon, pi, gamma = \
                _retrieve_dual_information_low_symmetry_formulation(model_lp)
                
        # Otherwise, alpha_sep was feasible and we adjust our alpha_in
        else:
            alpha_in = alpha_sep
            gamma_in = gamma_sep
            epsilon_in = epsilon_sep
            pi_in = pi_sep
            objective_dual = np.sum(gamma_in + pi_in) + np.sum(arriv_in*alpha_in)
    
    end_lp = time.time()
    
    # Otherwise, alpha_sep was feasible and we adjust our alpha_in
    if do_heuristic:
        if heuristic_formulation_1:
            model_heuristic = gp.Model()
            model_heuristic.setParam("LogToConsole", log_console)
            model_heuristic.setParam('MIPFocus', 2) 
            model_heuristic.setParam('Symmetry', 2)
            
            if maximum_running_time:    
                model_heuristic.setParam('TimeLimit', maximum_running_time)      
            
            """ add constraints and variables """ 
            _create_variables_constraints_naive_formulation(model=model_heuristic,
                number_vehicles=number_vehicles, number_orders=number_orders, arrivals=arrivals,
                mapping_subsets_to_routing_times=mapping_subsets_to_routing_times,
                mapping_max_order_to_subsets=mapping_max_order_to_subsets,
                mapping_orders_to_subsets_they_appear=mapping_orders_to_subsets_they_appear,
                lp_relaxation=False
                )
            """ optimize master problem for the first time """
            model_heuristic.optimize()
            obj = model_heuristic.getObjective()
            objective_value_heuristic = obj.getValue()
            runtime_lp = end_lp - start_lp
            runtime_heur = end_heuristic - end_lp
            columns_generated = len(mapping_subsets_to_routing_times)
            
            return objective_value_lp, objective_value_heuristic, \
                runtime_lp, runtime_heur, columns_generated
        
        else:
            model_heuristic = gp.Model()
            model_heuristic.setParam("LogToConsole", log_console)
            model_heuristic.setParam('MIPFocus', 2) 
            model_heuristic.setParam('Symmetry', 2)
            
            if maximum_running_time:    
                model_heuristic.setParam('TimeLimit', maximum_running_time)      
            
            """ add constraints and variables """ 
            _create_variables_constraints_low_symmetry_formulation(model=model_heuristic,
                number_vehicles=number_vehicles, number_orders=number_orders, arrivals=arrivals,
                mapping_subsets_to_routing_times=mapping_subsets_to_routing_times,
                mapping_max_order_to_subsets=mapping_max_order_to_subsets,
                mapping_orders_to_subsets_they_appear=mapping_orders_to_subsets_they_appear,
                fixed_cost=fixed_cost,
                processing_times_orders=processing_times_orders,
                family_setup_times=family_setup_times,
                lp_relaxation=False
                )
            
            """ optimize master problem for the first time """
            model_heuristic.optimize()
            obj = model_heuristic.getObjective()
            objective_value_heuristic = obj.getValue()
            end_heuristic = time.time()
        
            runtime_lp = end_lp - start_lp
            runtime_heur = end_heuristic - end_lp
            columns_generated = len(mapping_subsets_to_routing_times)
            
            return objective_value_lp, objective_value_heuristic, \
                runtime_lp, runtime_heur, columns_generated

    return objective_value_lp, -1

def colgen_flow_families(
        number_orders: int,
        number_vehicles: int,
        number_families: int,
        arrivals: np.ndarray, 
        processing_times_orders: np.ndarray, 
        orders_to_families_assignment: np.ndarray,
        family_setup_times: np.ndarray,
        fixed_cost: int=0,
        log_console: int=0,
        do_heuristic: bool=True,
        p_feas: float=0.8,
        maximum_running_time: float|bool=False
        ):
    start_lp = time.time()
    # Will be useful to find orders within families when doing pricing
    dictionary_families_to_orders = {
        family: [index for index in range(number_orders) if \
        orders_to_families_assignment[index] == family] \
        for family in range(number_families)}
    
    # Initialize the problem
    mapping_orders_to_subsets_they_appear, mapping_max_order_to_subsets, \
        mapping_subsets_to_routing_times = _initialize_singletons_families(
        number_orders=number_orders,
        orders_to_families_assignment=orders_to_families_assignment,
        family_setup_times=family_setup_times,
        processing_times_orders=processing_times_orders,
        fixed_cost=fixed_cost
        )
    
    """ creating the LP and setting parameters """   
    model_lp = gp.Model()
    model_lp.setParam("LogToConsole", log_console)
    model_lp.setParam('MIPFocus', 2) 
    model_lp.setParam('Symmetry', 2)
    
    if maximum_running_time:    
        model_lp.setParam('TimeLimit', maximum_running_time)
    model_lp.setParam('NumericFocus', 2)
    
    
    """ add constraints and variables """ 
    _create_variables_constraints_flow_formulation(model=model_lp, 
        number_vehicles=number_vehicles, number_orders=number_orders,
        arrivals=arrivals,
        mapping_subsets_to_routing_times=mapping_subsets_to_routing_times,
        mapping_max_order_to_subsets=mapping_max_order_to_subsets,
        mapping_orders_to_subsets_they_appear=mapping_orders_to_subsets_they_appear,
        fixed_cost=fixed_cost,
        processing_times_orders=processing_times_orders,
        family_setup_times=family_setup_times,
        lp_relaxation=True
        )
    
    """ optimize master problem for the first time """
    model_lp.optimize()
    obj = model_lp.getObjective()
    objective_value_lp = obj.getValue()
    
    # Initialize the problem
    alpha, beta, epsilon, pi, gamma, epsilon_0, epsilon_n = \
        _retrieve_dual_information_flow_formulation(model_lp) 
    vector_of_indices_for_beta = [number_orders - i for i in range(number_orders)]
    vector_of_indices_for_beta = np.cumsum(vector_of_indices_for_beta)
    alpha_in = np.zeros(number_orders)
    beta_in = np.zeros(len(beta))
    gamma_in = np.zeros(number_orders)
    epsilon_in = np.zeros(number_orders)
    epsilon_0_in = 0
    epsilon_n_in = 0
    big_m = np.sum(processing_times_orders) + np.sum(family_setup_times) + fixed_cost
    objective_dual = number_vehicles * (epsilon_0_in + epsilon_n_in) + \
        np.sum(arrivals*alpha_in+gamma_in) - np.sum(big_m*beta_in)
        
    # Start column generation
    tolerance_eps = TOLERANCE_DUAL_PRIMAL_COLGEN
    max_iterations = MAX_ITERATIONS_COLGEN
    iterations = 0
    columns_added_in_pricing = set()
    # Termination criteria when our dual and primal bounds are equal
    while(abs(objective_value_lp-objective_dual) > tolerance_eps and\
        iterations <= max_iterations):
        iterations += 1
        if iterations % 2500 == 0:
            print(f'{iterations = } and {objective_value_lp = } and {objective_dual = }')
            
        alpha_prime = np.asarray(alpha)
        beta_prime = np.asarray(beta)
        gamma_prime = np.asarray(gamma)
        epsilon_prime = np.asarray(epsilon)
        
        # Set our variables for the separation problem
        alpha_sep = p_feas*alpha_in + (1-p_feas)*alpha_prime
        beta_sep = p_feas*beta_in + (1-p_feas)*beta_prime
        gamma_sep = p_feas*gamma_in + (1-p_feas)*gamma_prime
        epsilon_sep = p_feas*epsilon_in + (1-p_feas)*epsilon_prime
        epsilon_0_sep = p_feas*epsilon_0_in + (1-p_feas)*epsilon_0
        epsilon_n_sep = p_feas*epsilon_n_in + (1-p_feas)*epsilon_n
        
        # Solve pricing problem
        best_objective = 0.0000001
        subset_to_enter = None
        for order in range(number_orders):
            order_list = {order}
            family_order = orders_to_families_assignment[order]
            setup_family = family_setup_times[family_order]
            gamma_objective = gamma_sep[order] + epsilon_sep[order]
            routing_objective = setup_family + \
                processing_times_orders[order] + fixed_cost
            if order > 0:
                coefficient_beta = np.sum(beta_sep[
                    vector_of_indices_for_beta[order-1]:\
                    vector_of_indices_for_beta[order]])
            else:
                coefficient_beta = np.sum(beta_sep[:\
                    vector_of_indices_for_beta[order]])
                
            for order_j in dictionary_families_to_orders[family_order]:
                if order_j >= order:
                    break
                else:
                    if coefficient_beta*processing_times_orders[order_j] \
                            < gamma_sep[order_j]:
                        order_list.add(order_j)
                        routing_objective += processing_times_orders[order_j]
                        gamma_objective += gamma_sep[order_j]
            
            total_objective = gamma_objective - coefficient_beta*routing_objective
            if total_objective > best_objective:
                subset_to_enter = order_list
                best_objective = total_objective
        
        # If we have a valid variable to enter, then we add it to the problem
        if subset_to_enter != None and \
                tuple(sorted(subset_to_enter)) not in columns_added_in_pricing:
            subset_to_enter = sorted(subset_to_enter)
            columns_added_in_pricing.add(tuple(sorted(subset_to_enter)))
            family_order = orders_to_families_assignment[subset_to_enter[-1]]
            routing_time = _compute_routing_time_families(
                order_list=subset_to_enter,
                processing_times_orders=processing_times_orders,
                family_setup=family_setup_times[family_order],
                fixed_cost=fixed_cost
                )
            
            # Add to constraints
            constraints_where_enters = [
                model_lp.getConstrByName(f'const3d[{order}]') for order in \
                subset_to_enter]
            coefficients_constraints = [1 for order in subset_to_enter]
            
            constraints_where_enters.append(
                model_lp.getConstrByName(f'const3e[{subset_to_enter[-1]}]'))
            coefficients_constraints.append(1)
            
            for order_j in range(subset_to_enter[-1]+1, number_orders+1):
                constraints_where_enters.append(
                    model_lp.getConstrByName(f'const3g[{subset_to_enter[-1]},{order_j}]'))
                coefficients_constraints.append(-routing_time)
            
            # Register with our maps
            tupled = tuple(subset_to_enter)
            mapping_subsets_to_routing_times[tupled] = routing_time
            mapping_max_order_to_subsets[subset_to_enter[-1]].add(tupled)
            for order in subset_to_enter:
                mapping_orders_to_subsets_they_appear[order].add(tupled)
            
            model_lp.addVar(obj=0, name=f'x_{iterations}', 
                column=gp.Column(coefficients_constraints, constraints_where_enters))
            
            # Optimize the problem again and retrieve the dual information
            model_lp.optimize()
            obj = model_lp.getObjective()
            objective_value_lp = obj.getValue()
            alpha, beta, epsilon, pi, gamma, epsilon_0, epsilon_n = \
                _retrieve_dual_information_flow_formulation(model_lp)

        # Optimize the problem again and retrieve the dual information  
        else:
            alpha_in = alpha_sep
            beta_in = beta_sep
            gamma_in = gamma_sep
            epsilon_in = epsilon_sep
            epsilon_0_in = epsilon_0_sep
            epsilon_n_in = epsilon_n_sep
            objective_dual = number_vehicles * (epsilon_0_in + epsilon_n_in) + \
                np.sum(arrivals*alpha_in+gamma_in) - np.sum(big_m*beta_in)
    end_lp = time.time()
    
    # If desired, we solve the problem heuristically with all the columns generated
    if do_heuristic:
        model_heuristic = gp.Model()
        model_heuristic.setParam("LogToConsole", log_console)
        model_heuristic.setParam('MIPFocus', 2)
        model_heuristic.setParam('Symmetry', 2)
        
        if maximum_running_time:    
            model_heuristic.setParam('TimeLimit', maximum_running_time)      
        
        """ add constraints and variables """ 
        _create_variables_constraints_flow_formulation(model=model_heuristic,
            number_vehicles=number_vehicles, number_orders=number_orders, arrivals=arrivals,
            mapping_subsets_to_routing_times=mapping_subsets_to_routing_times,
            mapping_max_order_to_subsets=mapping_max_order_to_subsets,
            mapping_orders_to_subsets_they_appear=mapping_orders_to_subsets_they_appear,
            fixed_cost=fixed_cost,
            processing_times_orders=processing_times_orders,
            family_setup_times=family_setup_times,
            lp_relaxation=False
            )
        
        """ optimize master problem for the first time """
        model_heuristic.optimize()
        obj = model_heuristic.getObjective()
        objective_value_heuristic = obj.getValue()
        
        end_heuristic = time.time()
        
        runtime_lp = end_lp - start_lp
        runtime_heur = end_heuristic - end_lp
        columns_generated = len(mapping_subsets_to_routing_times)
        
        return objective_value_lp, objective_value_heuristic, \
            runtime_lp, runtime_heur, columns_generated
                
    return objective_value_lp, -1, end_lp - start_lp,\
        0, len(mapping_subsets_to_routing_times)


