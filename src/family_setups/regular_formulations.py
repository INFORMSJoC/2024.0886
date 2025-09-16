# -*- coding: utf-8 -*-
"""
@author: Ignacio Erazo
"""
from __future__ import annotations
import gurobipy as gp
import numpy as np

from src.family_setups.tool_functions import _initialize_all_subsets_families, \
    _initialize_interval_variables_families

from src.family_setups.create_variables_and_constraints import _create_variables_constraints_naive_formulation, \
    _create_variables_constraints_large_scale_naive_formulation, \
    _create_variables_constraints_low_symmetry_formulation, \
     _create_variables_constraints_flow_formulation


def naive_formulation_families(
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
        mapping_subsets_to_routing_times = _initialize_all_subsets_families(
        number_orders=number_orders,
        number_families=number_families,
        orders_to_families_assignment=orders_to_families_assignment,
        family_setup_times=family_setup_times,
        processing_times_orders=processing_times_orders,
        fixed_cost=fixed_cost
        )
    
    # Create the model and add configurations
    model_primal = gp.Model()
    model_primal.setParam("LogToConsole", log_console)
    model_primal.setParam('MIPFocus', 2)
    model_primal.setParam('Symmetry', 2)
    if maximum_running_time:    
        model_primal.setParam('TimeLimit', maximum_running_time) 
    
    # Add the variables and constraints
    _create_variables_constraints_naive_formulation(model=model_primal,
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

def low_symmetry_formulation_families(
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
        mapping_subsets_to_routing_times = _initialize_all_subsets_families(
        number_orders=number_orders,
        number_families=number_families,
        orders_to_families_assignment=orders_to_families_assignment,
        family_setup_times=family_setup_times,
        processing_times_orders=processing_times_orders,
        fixed_cost=fixed_cost
        )     
    
    # Create the model and add configurations
    model_primal = gp.Model()
    model_primal.setParam("LogToConsole", log_console)
    model_primal.setParam('MIPFocus', 2)
    model_primal.setParam('Symmetry', 2)
    
    if maximum_running_time:    
        model_primal.setParam('TimeLimit', maximum_running_time) 
    
    # Add the variables and constraints
    _create_variables_constraints_low_symmetry_formulation(
        model=model_primal,
        number_vehicles=number_vehicles,
        number_orders=number_orders,
        arrivals=arrivals,
        mapping_subsets_to_routing_times=mapping_subsets_to_routing_times,
        mapping_max_order_to_subsets=mapping_max_order_to_subsets,
        mapping_orders_to_subsets_they_appear=mapping_orders_to_subsets_they_appear,
        fixed_cost=fixed_cost,
        processing_times_orders=processing_times_orders,
        family_setup_times=family_setup_times,
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

def flow_formulation_families(
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
        mapping_subsets_to_routing_times = _initialize_all_subsets_families(
        number_orders=number_orders,
        number_families=number_families,
        orders_to_families_assignment=orders_to_families_assignment,
        family_setup_times=family_setup_times,
        processing_times_orders=processing_times_orders,
        fixed_cost=fixed_cost
        )
    
    # Create the model and add configurations
    model_primal = gp.Model()
    model_primal.setParam("LogToConsole", log_console)
    model_primal.setParam('MIPFocus', 2)
    model_primal.setParam('Symmetry', 2)
    
    if maximum_running_time:    
        model_primal.setParam('TimeLimit', maximum_running_time) 

    # Add the variables and constraints
    _create_variables_constraints_flow_formulation(
        model=model_primal,
        number_vehicles=number_vehicles,
        number_orders=number_orders,
        arrivals=arrivals,
        mapping_subsets_to_routing_times=mapping_subsets_to_routing_times,
        mapping_max_order_to_subsets=mapping_max_order_to_subsets,
        mapping_orders_to_subsets_they_appear=mapping_orders_to_subsets_they_appear,
        fixed_cost=fixed_cost,
        processing_times_orders=processing_times_orders,
        family_setup_times=family_setup_times,
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

def naive_quadratic_interval_formulation_families(
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
        mapping_subsets_to_routing_times = _initialize_interval_variables_families(
        number_orders=number_orders,
        orders_to_families_assignment=orders_to_families_assignment,
        family_setup_times=family_setup_times,
        processing_times_orders=processing_times_orders,
        fixed_cost=fixed_cost
        ) 
    
    # Create the model and add configurations
    model_primal = gp.Model()
    model_primal.setParam("LogToConsole", log_console)
    model_primal.setParam('MIPFocus', 2)
    model_primal.setParam('Symmetry', 2)

    if maximum_running_time:    
        model_primal.setParam('TimeLimit', maximum_running_time) 
    
    # Add the variables and constraints
    _create_variables_constraints_large_scale_naive_formulation(model=model_primal,
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