# -*- coding: utf-8 -*-
"""
@author: Ignacio Erazo
"""
from __future__ import annotations
import gurobipy as gp
import numpy as np
from collections import defaultdict
import time

MAX_ITERATIONS_COLGEN = 250000
TOLERANCE_DUAL_PRIMAL_COLGEN = 0.0001

from src.family_setups.tool_functions import powerset, _initialize_all_subsets_families
from src.family_setups.regular_formulations import naive_formulation_families
from src.family_setups.lower_bounds import lower_bound_improved_families


def _initialize_subsets_set_cover_formulation(
        number_orders: int,
        number_vehicles: int,
        number_families: int,
        arrivals: np.ndarray, 
        processing_times_orders: np.ndarray, 
        orders_to_families_assignment: np.ndarray,
        family_setup_times: np.ndarray,
        fixed_cost: int=0,
        do_only_singletons: bool=False
        ):
    # This is useful for column generation
    if do_only_singletons:
        all_subsets = [(i, ) for i in range(number_orders)]
        all_subsets.append(tuple(i for i in range(number_orders)))
    # In contrast with the other formulations, we need to consider batches
    # that have orders from multiple families
    else:
        all_subsets = powerset([i for i in range(number_orders)])
    
    # Obtaining makespan information for vehicles based on a batch
    mapping_orders_to_subsets_they_appear = defaultdict(lambda: set())
    mapping_max_order_to_subsets = defaultdict(lambda: set())
    mapping_subsets_to_routing_times = dict()
    for subset_tuple in all_subsets:
        if len(subset_tuple) > 0:
            subset_list = list(subset_tuple)
            arrivals_subproblem = arrivals[subset_list]
            processing_times_orders_subproblem = \
                processing_times_orders[subset_list]
            orders_to_families_assignment_subproblem = \
                orders_to_families_assignment[subset_list]
            
            # Need to solve a single vehicle subproblem
            makespan = naive_formulation_families(
                number_orders=len(subset_list),
                number_vehicles=1,
                number_families=number_families,
                arrivals=arrivals_subproblem,
                processing_times_orders=processing_times_orders_subproblem,
                orders_to_families_assignment=orders_to_families_assignment_subproblem,
                family_setup_times=family_setup_times,
                fixed_cost=fixed_cost,
                lp_relaxation=False,
                log_console=0,
                maximum_running_time=0
                )
            tupled = tuple(sorted(subset_list))
            mapping_subsets_to_routing_times[tupled] = makespan
            mapping_max_order_to_subsets[tupled[-1]].add(tupled)
            for order in tupled:
                mapping_orders_to_subsets_they_appear[order].add(tupled)
    return mapping_orders_to_subsets_they_appear, mapping_max_order_to_subsets, \
        mapping_subsets_to_routing_times

def _create_variables_constraints_set_cover_formulation(
        model: gp.Model,
        number_vehicles: int,
        number_orders: int,
        arrivals: np.ndarray,
        mapping_subsets_to_routing_times: dict[tuple[int], float],
        mapping_max_order_to_subsets: dict[int, set[tuple[int]]],
        mapping_orders_to_subsets_they_appear: dict[int, set[tuple[int]]],
        lp_relaxation: bool
        ):
    indices_x = [(subset, vehicle) for vehicle in range(number_vehicles) \
        for subset in mapping_subsets_to_routing_times]
    index_makespan = [0]
    # Add the "full day batches" variables x
    if lp_relaxation:
        x = model.addVars(indices_x, name='x')
    else:
        x = model.addVars(indices_x, name='x', vtype=gp.GRB.BINARY)  
    
    # Set objective value
    makespan = model.addVars(index_makespan, name='makespan')
    model.setObjective(makespan[0], gp.GRB.MINIMIZE)
    
    # Add makespan constraint
    model.addConstrs((makespan[0] >= gp.quicksum([
        x[(subset, vehicle)] * mapping_subsets_to_routing_times[subset] \
        for subset in mapping_subsets_to_routing_times])
        for vehicle in range(number_vehicles)), name=f'const4a')
    
    # At least number_vehicle dispatches
    model.addConstr(gp.quicksum([x[tupled] for tupled in x]) >= \
        number_vehicles, name=f'const4b')
    
    # Coverage constraint for each order
    model.addConstrs((gp.quicksum([x[(subset, vehicle)] for subset in \
        mapping_orders_to_subsets_they_appear[order] for vehicle in\
        range(number_vehicles)]) >= 1 for order in range(number_orders)), name='const4c')
    return

def set_cover_formulation_families(
        number_orders: int,
        number_vehicles: int,
        number_families: int,
        arrivals: np.ndarray, 
        processing_times_orders: np.ndarray, 
        orders_to_families_assignment: np.ndarray,
        family_setup_times: np.ndarray,
        fixed_cost: int=0,
        lp_relaxation: bool=False,
        log_console: int=0,
        maximum_running_time: float|bool=False
        ):
    # Get all the info we need in dictionaries
    mapping_orders_to_subsets_they_appear, mapping_max_order_to_subsets, \
        mapping_subsets_to_routing_times = \
        _initialize_subsets_set_cover_formulation(
        number_orders=number_orders,
        number_vehicles=number_vehicles,
        number_families=number_families,
        arrivals=arrivals, 
        processing_times_orders=processing_times_orders, 
        orders_to_families_assignment=orders_to_families_assignment,
        family_setup_times=family_setup_times,
        fixed_cost=fixed_cost
        )
    
    # Create the model
    model_primal = gp.Model()
    model_primal.setParam("LogToConsole", log_console)
    model_primal.setParam('MIPFocus', 2)
    model_primal.setParam('Symmetry', 2)
    
    if maximum_running_time:    
        model_primal.setParam('TimeLimit', maximum_running_time) 
    
    # Add the constraints for set cover formulation
    _create_variables_constraints_set_cover_formulation(model=model_primal,
        number_vehicles=number_vehicles, number_orders=number_orders, arrivals=arrivals,
        mapping_subsets_to_routing_times=mapping_subsets_to_routing_times,
        mapping_max_order_to_subsets=mapping_max_order_to_subsets,
        mapping_orders_to_subsets_they_appear=mapping_orders_to_subsets_they_appear,
        lp_relaxation=lp_relaxation
        )
    
    # Optimize and retrieve solution
    model_primal.optimize()
    solution_list=[]
    for i in model_primal.getVars():
        if i.x > 0:
            solution_list.append((i.varname, i.x))
    obj = model_primal.getObjective()
    objective_value = obj.getValue()
    return objective_value

def _retrieve_dual_information_set_cover_formulation(
        model_lp: gp.Model
        ):
    """
    Retrieve dual information for column generation purposes
    """
    alpha = []
    beta = 0
    gamma = []
    all_names = []
    for c in model_lp.getConstrs():
        if c.constrName[6] == 'a':
            alpha.append(c.pi)
        if c.constrName[6] == 'b':
            # They come out ordered by order -> all vehicles, order+1 -> ...
            beta = c.pi
        if c.constrName[6] == 'c':
            gamma.append(c.pi)
        all_names.append(c.constrName)
    return alpha, beta, gamma

def _ip_pricing_set_cover_formulation_families_with_no_fixed(
        number_orders: int,
        number_families: int,
        arrivals: np.ndarray,
        processing_times_orders: np.ndarray,
        orders_to_families_assignment: np.ndarray,
        family_setup_times: np.ndarray,
        alpha: float,
        gamma: np.ndarray,
        fixed_cost=0, # otherwise the logic behind the model does not work
        log_console: int=0,
        maximum_running_time: int=0
        ):
    # Obtain information required in maps
    mapping_orders_to_subsets_they_appear, mapping_max_order_to_subsets, \
        mapping_subsets_to_routing_times = _initialize_all_subsets_families(
        number_orders=number_orders,
        number_families=number_families,
        orders_to_families_assignment=orders_to_families_assignment,
        family_setup_times=family_setup_times,
        processing_times_orders=processing_times_orders,
        fixed_cost=fixed_cost
        )
    # dual map
    mapping_subsets_to_summed_gammas = dict()
    for subset in mapping_subsets_to_routing_times:
        subset_list = list(subset)
        summed_gamma = np.sum(gamma[subset_list])
        mapping_subsets_to_summed_gammas[subset] = summed_gamma
        
    """ creating the IP and setting parameters """   
    model_pricing = gp.Model()
    model_pricing.setParam("LogToConsole", log_console)
    model_pricing.setParam('MIPFocus', 2) 
    model_pricing.setParam('Symmetry', 2)
    if maximum_running_time:    
        model_pricing.setParam('TimeLimit', maximum_running_time) 
    
    # Adding the indices to create the variables
    index_makespan = [0]
    indices_x = [subset for subset in mapping_subsets_to_routing_times]
    indices_t = [order for order in range(number_orders)]
    
    # Creating the variables
    makespan = model_pricing.addVars(index_makespan, name='makespan')
    x = model_pricing.addVars(indices_x, name='x', vtype=gp.GRB.BINARY)
    t = model_pricing.addVars(indices_t, name='t')
    
    # Set objective value
    model_pricing.setObjective(-alpha * makespan[0] + gp.quicksum([ 
        x[subset]*mapping_subsets_to_summed_gammas[subset] for subset in \
        mapping_subsets_to_routing_times]), gp.GRB.MAXIMIZE)
    
    # This will help to make sure the opt. sol cannot be the empty set
    minimum_makespan = min([processing_times_orders[order] + \
        family_setup_times[orders_to_families_assignment[order]] +\
        arrivals[order] for order in range(number_orders) if gamma[order]> 0]) # revisit
    
    # add single vehicle smd constraints
    model_pricing.addConstr(makespan[0] >= minimum_makespan, name='c1')
    
    # Adding other pricing constraints
    model_pricing.addConstrs((t[order] >= arrivals[order] * gp.quicksum([
        x[subset] for subset in mapping_max_order_to_subsets[order]]) \
        for order in range(number_orders)), name='c2')
    
    model_pricing.addConstrs((t[order+1] >= t[order] + gp.quicksum([
        x[subset]*mapping_subsets_to_routing_times[subset] \
        for subset in mapping_max_order_to_subsets[order]]) \
        for order in range(number_orders - 1)), name='c3')
    
    model_pricing.addConstr(makespan[0] >= t[number_orders - 1] + \
        gp.quicksum([x[subset]*mapping_subsets_to_routing_times[subset] \
        for subset in mapping_max_order_to_subsets[number_orders-1]]), \
        name='c4')
    
    model_pricing.addConstrs((gp.quicksum([
        x[subset] for subset in mapping_orders_to_subsets_they_appear[order]])
        <= 1 for order in range(number_orders)), name='c5' )
    
    # Optimize the pricing problem and retrieve solution
    model_pricing.optimize()
    obj = model_pricing.getObjective()
    objective_pricing = obj.getValue()
    subset = set()
    for i in model_pricing.getVars():
        if i.x > 0.5 and i.varname[0:1]=='x':
            all_numbers = i.varname[2:-1]
            individual_numbers = set(all_numbers.split(sep=','))
            subset.update(individual_numbers)
        elif i.x > 0 and i.varname[0:3]=='mak':
            routing_time = i.x
    return subset, routing_time, objective_pricing

def colgen_set_cover_families(
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
    # Get all the info we need in dictionaries
    mapping_orders_to_subsets_they_appear, mapping_max_order_to_subsets, \
        mapping_subsets_to_routing_times = \
        _initialize_subsets_set_cover_formulation(
        number_orders=number_orders,
        number_vehicles=number_vehicles,
        number_families=number_families,
        arrivals=arrivals, 
        processing_times_orders=processing_times_orders, 
        orders_to_families_assignment=orders_to_families_assignment,
        family_setup_times=family_setup_times,
        fixed_cost=fixed_cost,
        do_only_singletons=True
        )
    
    """ creating the LP and setting parameters """   
    model_lp = gp.Model()
    model_lp.setParam("LogToConsole", log_console)
    model_lp.setParam('MIPFocus', 2)
    model_lp.setParam('Symmetry', 2)
    
    if maximum_running_time:    
        model_lp.setParam('TimeLimit', maximum_running_time)      
    
    """ add constraints and variables """ 
    _create_variables_constraints_set_cover_formulation(model=model_lp,
        number_vehicles=number_vehicles, number_orders=number_orders, arrivals=arrivals,
        mapping_subsets_to_routing_times=mapping_subsets_to_routing_times,
        mapping_max_order_to_subsets=mapping_max_order_to_subsets,
        mapping_orders_to_subsets_they_appear=mapping_orders_to_subsets_they_appear,
        lp_relaxation=True
        )
    
    # Optimize the initial LP
    model_lp.optimize()
    obj = model_lp.getObjective()
    objective_value_lp = obj.getValue()
    
    # Retrieve dual information and start column generation
    alpha, beta, gamma = _retrieve_dual_information_set_cover_formulation(
        model_lp=model_lp)
    alpha_in = np.zeros(number_vehicles) 
    beta_in = 0  #i,k
    gamma_in = np.zeros(number_orders) 
    objective_dual = beta_in*number_vehicles + np.sum(gamma_in)
    tolerance_eps = TOLERANCE_DUAL_PRIMAL_COLGEN
    max_iterations = MAX_ITERATIONS_COLGEN
    iterations = 0
    columns_added_in_pricing = set()
    
    # Our convergence criteria requires dual and primal bound to be equal
    while(abs(objective_value_lp-objective_dual) > tolerance_eps and\
        iterations <= max_iterations):
        iterations += 1
        if iterations % 2500 == 0:
            print(f'{iterations = } and {objective_value_lp = } and {objective_dual = }')
            
        alpha_prime = np.asarray(alpha)
        beta_prime = beta
        gamma_prime = np.asarray(gamma)
        # Set our variables for the separation problem
        alpha_sep = p_feas*alpha_in + (1-p_feas)*alpha_prime
        beta_sep = p_feas*beta_in + (1-p_feas)*beta_prime
        gamma_sep = p_feas*gamma_in + (1-p_feas)*gamma_prime
        
        # Solving the pricing problem
        index_min_alpha = np.argmin(alpha_sep)
        subset_opt, routing_time, opt_pricing = \
            _ip_pricing_set_cover_formulation_families_with_no_fixed(
                number_orders=number_orders,
                number_families=number_families,
                arrivals=arrivals,
                processing_times_orders=processing_times_orders,
                orders_to_families_assignment=orders_to_families_assignment,
                family_setup_times=family_setup_times,
                alpha=alpha_sep[index_min_alpha],
                gamma=gamma_sep,
                fixed_cost=0
                )
        
        # Add variable to master problem 
        if opt_pricing + beta_sep > 0.0001 and \
                (tuple(sorted(subset_opt)), index_min_alpha) not in columns_added_in_pricing:
            subset_to_enter = sorted(subset_opt)
            tupled = tuple(subset_to_enter)
            columns_added_in_pricing.add((tupled, index_min_alpha))
        
            # Add into cosntraints
            constraints_where_enters = [
                model_lp.getConstrByName(f'const4c[{order}]') for order in \
                subset_to_enter]
            coefficients_constraints = [1 for order in subset_to_enter]
        
            constraints_where_enters.append(
                model_lp.getConstrByName(f'const4b'))
            coefficients_constraints.append(1)
            
            constraints_where_enters.append(
                model_lp.getConstrByName(f'const4a[{index_min_alpha}]'))
            coefficients_constraints.append(-routing_time)

            mapping_subsets_to_routing_times[tupled] = routing_time
            mapping_max_order_to_subsets[tupled[-1]].add(tupled)
            for order in tupled:
                mapping_orders_to_subsets_they_appear[order].add(tupled)
            
            model_lp.addVar(obj=0, name=f'x_{iterations}', 
                column=gp.Column(coefficients_constraints, constraints_where_enters))
            
            model_lp.optimize()
            obj = model_lp.getObjective()
            objective_value_lp = obj.getValue()
            alpha, beta, gamma = _retrieve_dual_information_set_cover_formulation(
                model_lp=model_lp)
                
        # The dual variable was feasible so we update our dual variables
        else:
            alpha_in = alpha_sep
            beta_in = beta_sep
            gamma_in = gamma_sep
            objective_dual = beta_in*number_vehicles + np.sum(gamma_in)
    end_lp = time.time()

    if do_heuristic:
        model_heuristic = gp.Model()
        model_heuristic.setParam("LogToConsole", log_console)
        model_heuristic.setParam('MIPFocus', 2)
        model_heuristic.setParam('Symmetry', 2)

        if maximum_running_time:    
            model_heuristic.setParam('TimeLimit', maximum_running_time)      
        
        """ add constraints and variables """ 
        _create_variables_constraints_set_cover_formulation(model=model_heuristic,
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
    return objective_value_lp, -1, 0 ,0, 0


""" Strong version of set cover """
def _initialize_subsets_set_cover_formulation_strong(
        number_orders: int,
        number_vehicles: int,
        number_families: int,
        arrivals: np.ndarray, 
        processing_times_orders: np.ndarray, 
        orders_to_families_assignment: np.ndarray,
        family_setup_times: np.ndarray,
        lower_bound_on_makespan: float,
        fixed_cost: int=0,
        do_only_singletons: bool=False
        ):
    # This is useful for column generation
    if do_only_singletons:
        all_subsets = [(i, ) for i in range(number_orders)]
        all_subsets.append(tuple(i for i in range(number_orders)))
    # In contrast with the other formulations, we need to consider batches
    # that have orders from multiple families
    else:
        all_subsets = powerset([i for i in range(number_orders)])

    # Obtain relevant information in maps
    mapping_orders_to_subsets_they_appear = defaultdict(lambda: set())
    mapping_max_order_to_subsets = defaultdict(lambda: set())
    mapping_subsets_to_routing_times = dict()
    for subset_tuple in all_subsets:
        if len(subset_tuple) > 0:
            subset_list = list(subset_tuple)
            arrivals_subproblem = arrivals[subset_list]
            processing_times_orders_subproblem = \
                processing_times_orders[subset_list]
            orders_to_families_assignment_subproblem = \
                orders_to_families_assignment[subset_list]
            
            # Solve a single vehicle problem
            makespan = naive_formulation_families(
                number_orders=len(subset_list),
                number_vehicles=1,
                number_families=number_families,
                arrivals=arrivals_subproblem,
                processing_times_orders=processing_times_orders_subproblem,
                orders_to_families_assignment=orders_to_families_assignment_subproblem,
                family_setup_times=family_setup_times,
                fixed_cost=fixed_cost,
                lp_relaxation=False,
                log_console=0,
                maximum_running_time=0
                )
            # add to our maps
            tupled = tuple(sorted(subset_list))

            # This is the key step where we leverage the lower bound, compared
            # to the weak formulation
            mapping_subsets_to_routing_times[tupled] = max(
                makespan, lower_bound_on_makespan)
            
            mapping_max_order_to_subsets[tupled[-1]].add(tupled)
            for order in tupled:
                mapping_orders_to_subsets_they_appear[order].add(tupled)
    return mapping_orders_to_subsets_they_appear, mapping_max_order_to_subsets, \
        mapping_subsets_to_routing_times

def set_cover_formulation_families_strong(
        number_orders: int,
        number_vehicles: int,
        number_families: int,
        arrivals: np.ndarray, 
        processing_times_orders: np.ndarray, 
        orders_to_families_assignment: np.ndarray,
        family_setup_times: np.ndarray,
        fixed_cost: int=0,
        lp_relaxation: bool=False,
        log_console: int=0,
        maximum_running_time: float|bool=False
        ):
    # Lower bound that is leveraged by the strong set cover formulation
    lower_bound_on_makespan = lower_bound_improved_families(
                number_orders=number_orders,
                number_vehicles=number_vehicles,
                arrivals=arrivals, 
                processing_times_orders=processing_times_orders, 
                orders_to_families_assignment=orders_to_families_assignment,
                family_setup_times=family_setup_times,
                fixed_cost=fixed_cost
                )
    
    # Get all the info we need in dictionaries
    mapping_orders_to_subsets_they_appear, mapping_max_order_to_subsets, \
        mapping_subsets_to_routing_times = \
        _initialize_subsets_set_cover_formulation_strong(
        number_orders=number_orders,
        number_vehicles=number_vehicles,
        number_families=number_families,
        arrivals=arrivals, 
        processing_times_orders=processing_times_orders, 
        orders_to_families_assignment=orders_to_families_assignment,
        family_setup_times=family_setup_times,
        lower_bound_on_makespan=lower_bound_on_makespan,
        fixed_cost=fixed_cost
        )
    
    # Create model   
    model_primal = gp.Model()
    model_primal.setParam("LogToConsole", log_console)
    model_primal.setParam('MIPFocus', 2)
    model_primal.setParam('Symmetry', 2)
    
    if maximum_running_time:    
        model_primal.setParam('TimeLimit', maximum_running_time) 
    
    # Create variables for the formulation
    _create_variables_constraints_set_cover_formulation(model=model_primal,
        number_vehicles=number_vehicles, number_orders=number_orders, arrivals=arrivals,
        mapping_subsets_to_routing_times=mapping_subsets_to_routing_times,
        mapping_max_order_to_subsets=mapping_max_order_to_subsets,
        mapping_orders_to_subsets_they_appear=mapping_orders_to_subsets_they_appear,
        lp_relaxation=lp_relaxation
        )
    
    # Optimize and retrieve the optimal solution
    model_primal.optimize()
    solution_list = []
    for i in model_primal.getVars():
        if i.x > 0:
            solution_list.append((i.varname, i.x))
    obj = model_primal.getObjective()
    objective_value = obj.getValue()
    return objective_value

def colgen_set_cover_families_strong(
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
    # Get all the info we need in dictionaries
    lower_bound_on_makespan = lower_bound_improved_families(
                number_orders=number_orders,
                number_vehicles=number_vehicles,
                arrivals=arrivals, 
                processing_times_orders=processing_times_orders, 
                orders_to_families_assignment=orders_to_families_assignment,
                family_setup_times=family_setup_times,
                fixed_cost=fixed_cost
                )
    
    # Initialize the strong set cover formulation
    mapping_orders_to_subsets_they_appear, mapping_max_order_to_subsets, \
        mapping_subsets_to_routing_times = \
        _initialize_subsets_set_cover_formulation_strong(
        number_orders=number_orders,
        number_vehicles=number_vehicles,
        number_families=number_families,
        arrivals=arrivals, 
        processing_times_orders=processing_times_orders, 
        orders_to_families_assignment=orders_to_families_assignment,
        family_setup_times=family_setup_times,
        fixed_cost=fixed_cost,
        lower_bound_on_makespan=lower_bound_on_makespan,
        do_only_singletons=True
        )
    
    """ creating the LP and setting parameters """   
    model_lp = gp.Model()
    model_lp.setParam("LogToConsole", log_console)
    model_lp.setParam('MIPFocus', 2)
    model_lp.setParam('Symmetry', 2)
    
    if maximum_running_time:    
        model_lp.setParam('TimeLimit', maximum_running_time)      
    
    """ add constraints and variables """ 
    _create_variables_constraints_set_cover_formulation(model=model_lp,
        number_vehicles=number_vehicles, number_orders=number_orders, arrivals=arrivals,
        mapping_subsets_to_routing_times=mapping_subsets_to_routing_times,
        mapping_max_order_to_subsets=mapping_max_order_to_subsets,
        mapping_orders_to_subsets_they_appear=mapping_orders_to_subsets_they_appear,
        lp_relaxation=True
        )
    
    # Optimize the LP from master problem
    model_lp.optimize()
    obj = model_lp.getObjective()
    objective_value_lp = obj.getValue()
    
    # Retrieve initial dual information
    alpha, beta, gamma = _retrieve_dual_information_set_cover_formulation(
        model_lp=model_lp)
    alpha_in = np.zeros(number_vehicles) 
    beta_in = 0  #i,k
    gamma_in = np.zeros(number_orders) 
    objective_dual = beta_in*number_vehicles + np.sum(gamma_in)
    tolerance_eps = TOLERANCE_DUAL_PRIMAL_COLGEN
    max_iterations = MAX_ITERATIONS_COLGEN
    iterations = 0
    
    # Perform column generation
    columns_added_in_pricing = set()
    while(abs(objective_value_lp - objective_dual) > tolerance_eps and\
        iterations <= max_iterations):
        iterations += 1
        if iterations % 2500 == 0:
            print(f'{iterations = } and {objective_value_lp = } and {objective_dual = }')
            
        alpha_prime = np.asarray(alpha)
        beta_prime = beta
        gamma_prime = np.asarray(gamma)
        # Set our variables for the separation problem
        alpha_sep = p_feas*alpha_in + (1-p_feas)*alpha_prime
        beta_sep = p_feas*beta_in + (1-p_feas)*beta_prime
        gamma_sep = p_feas*gamma_in + (1-p_feas)*gamma_prime
        
        
        index_min_alpha = np.argmin(alpha_sep)
        subset_opt, routing_time, opt_pricing = \
            _ip_pricing_set_cover_formulation_families_with_no_fixed_strong(
                number_orders=number_orders,
                number_families=number_families,
                arrivals=arrivals,
                processing_times_orders=processing_times_orders,
                orders_to_families_assignment=orders_to_families_assignment,
                family_setup_times=family_setup_times,
                alpha=alpha_sep[index_min_alpha],
                gamma=gamma_sep,
                lower_bound_on_makespan=lower_bound_on_makespan,
                fixed_cost=0
                )
        
        # Variable enters the Master problem
        if opt_pricing + beta_sep > 0.0001 and \
                (tuple(sorted(subset_opt)), index_min_alpha) not in columns_added_in_pricing:
            subset_to_enter = sorted(subset_opt)
            tupled = tuple(subset_to_enter)
            columns_added_in_pricing.add((tupled, index_min_alpha))
            
            # Add in constraints
            constraints_where_enters = [
                model_lp.getConstrByName(f'const4c[{order}]') for order in \
                subset_to_enter]
            coefficients_constraints = [1 for order in subset_to_enter]
        
            constraints_where_enters.append(
                model_lp.getConstrByName(f'const4b'))
            coefficients_constraints.append(1)
            
            constraints_where_enters.append(
                model_lp.getConstrByName(f'const4a[{index_min_alpha}]'))
            coefficients_constraints.append(-routing_time)
            
            # add to our maps
            mapping_subsets_to_routing_times[tupled] = routing_time
            mapping_max_order_to_subsets[tupled[-1]].add(tupled)
            for order in tupled:
                mapping_orders_to_subsets_they_appear[order].add(tupled)
            
            model_lp.addVar(obj=0, name=f'x_{iterations}', 
                column=gp.Column(coefficients_constraints, constraints_where_enters))
            
            # Reoptimize master problem
            model_lp.optimize()
            obj = model_lp.getObjective()
            objective_value_lp = obj.getValue()
            alpha, beta, gamma = _retrieve_dual_information_set_cover_formulation(
                model_lp=model_lp)
                
        # In this case the dual variables were feasible so we update them
        else:
            # try removing copy object
            alpha_in = alpha_sep
            beta_in = beta_sep
            gamma_in = gamma_sep
            objective_dual = beta_in*number_vehicles + np.sum(gamma_in)
    end_lp = time.time()

    if do_heuristic:
        model_heuristic = gp.Model()
        model_heuristic.setParam("LogToConsole", log_console)
        model_heuristic.setParam('MIPFocus', 2)
        model_heuristic.setParam('Symmetry', 2)
        
        if maximum_running_time:    
            model_heuristic.setParam('TimeLimit', maximum_running_time)      
        
        """ add constraints and variables """ 
        _create_variables_constraints_set_cover_formulation(model=model_heuristic,
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
    return objective_value_lp, -1, 0

def _ip_pricing_set_cover_formulation_families_with_no_fixed_strong(
        number_orders: int,
        number_families: int,
        arrivals: np.ndarray,
        processing_times_orders: np.ndarray,
        orders_to_families_assignment: np.ndarray,
        family_setup_times: np.ndarray,
        alpha: float,
        gamma: np.ndarray,
        lower_bound_on_makespan: float,
        fixed_cost=0, # otherwise the logic behind the model does not work
        log_console: int=0,
        maximum_running_time: int=0
        ):
    """
    Solving the pricing problem from the strong set cover formulation
    """
    
    mapping_orders_to_subsets_they_appear, mapping_max_order_to_subsets, \
        mapping_subsets_to_routing_times = _initialize_all_subsets_families(
        number_orders=number_orders,
        number_families=number_families,
        orders_to_families_assignment=orders_to_families_assignment,
        family_setup_times=family_setup_times,
        processing_times_orders=processing_times_orders,
        fixed_cost=fixed_cost
        )
    mapping_subsets_to_summed_gammas = dict()
    for subset in mapping_subsets_to_routing_times:
        subset_list = list(subset)
        summed_gamma = np.sum(gamma[subset_list])
        mapping_subsets_to_summed_gammas[subset] = summed_gamma
        
    """ creating the IP and setting parameters """   
    model_pricing = gp.Model()
    model_pricing.setParam("LogToConsole", log_console)
    model_pricing.setParam('MIPFocus', 2)
    model_pricing.setParam('Symmetry', 2)

    if maximum_running_time:    
        model_pricing.setParam('TimeLimit', maximum_running_time) 
    
    # Adding the indices to create the variables
    index_makespan = [0]
    indices_x = [subset for subset in mapping_subsets_to_routing_times]
    indices_t = [order for order in range(number_orders)]

    # Creating the variables
    makespan = model_pricing.addVars(index_makespan, name='makespan')
    x = model_pricing.addVars(indices_x, name='x', vtype=gp.GRB.BINARY)
    t = model_pricing.addVars(indices_t, name='t')
    
    
    # Set objective value
    model_pricing.setObjective(-alpha * makespan[0] + gp.quicksum([ 
        x[subset]*mapping_subsets_to_summed_gammas[subset] for subset in \
        mapping_subsets_to_routing_times]), gp.GRB.MAXIMIZE)
    
    # add single vehicle SMD constraints
    model_pricing.addConstr(makespan[0] >= lower_bound_on_makespan, name='c1')
    
    # Add other pricing constraints
    model_pricing.addConstrs((t[order] >= arrivals[order] * gp.quicksum([
        x[subset] for subset in mapping_max_order_to_subsets[order]]) \
        for order in range(number_orders)), name='c2')
    
    model_pricing.addConstrs((t[order+1] >= t[order] + gp.quicksum([
        x[subset]*mapping_subsets_to_routing_times[subset] \
        for subset in mapping_max_order_to_subsets[order]]) \
        for order in range(number_orders - 1)), name='c3')
    
    model_pricing.addConstr(makespan[0] >= t[number_orders - 1] + \
        gp.quicksum([x[subset]*mapping_subsets_to_routing_times[subset] \
        for subset in mapping_max_order_to_subsets[number_orders-1]]), \
        name='c4')
    
    model_pricing.addConstrs((gp.quicksum([
        x[subset] for subset in mapping_orders_to_subsets_they_appear[order]])
        <= 1 for order in range(number_orders)), name='c5' )
    
    # Optimize pricing problem
    model_pricing.optimize()
    obj = model_pricing.getObjective()
    objective_pricing = obj.getValue()
    subset = set()
    for i in model_pricing.getVars():
        if i.x > 0.5 and i.varname[0:1]=='x':
            all_numbers = i.varname[2:-1]
            individual_numbers = set(all_numbers.split(sep=','))
            subset.update(individual_numbers)
        elif i.x > 0 and i.varname[0:3]=='mak':
            routing_time = i.x
    return subset, routing_time, objective_pricing


