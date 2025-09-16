# -*- coding: utf-8 -*-
"""
@author: Ignacio Erazo
"""
from __future__ import annotations
import gurobipy as gp
import numpy as np

def _create_variables_constraints_naive_formulation(
        model: gp.Model,
        number_vehicles: int,
        number_orders: int,
        arrivals: np.ndarray,
        mapping_subsets_to_routing_times: dict[tuple[int], float],
        mapping_max_order_to_subsets: dict[int, set[tuple[int]]],
        mapping_orders_to_subsets_they_appear: dict[int, set[tuple[int]]],
        lp_relaxation: bool
        ):
    indices_t = [(order, vehicle) for order in range(number_orders) \
        for vehicle in range(number_vehicles) ]
    indices_x = [(subset, vehicle) for vehicle in range(number_vehicles) \
        for subset in mapping_subsets_to_routing_times]
    index_makespan = [0]
    # Add the variables
    t = model.addVars(indices_t, name='time')
    if lp_relaxation:
        x = model.addVars(indices_x, name='x')
    else:
        x = model.addVars(indices_x, name='x', vtype=gp.GRB.BINARY)    
    makespan = model.addVars(index_makespan, name='makespan')
    # Set objective value
    model.setObjective(makespan[0], gp.GRB.MINIMIZE)

    # Add departure constraints
    model.addConstrs( 
        (t[tupled] >= arrivals[tupled[0]] for tupled in t), name=f'const1a')
    
    # Add vehicle previous dispatch constraints
    for order in range(1, number_orders):
        model.addConstrs(
            (t[(order, vehicle)] >= t[(order-1, vehicle)] + gp.quicksum([ \
            x[(subset, vehicle)] * mapping_subsets_to_routing_times[subset]\
            for subset in mapping_max_order_to_subsets[order-1]]) \
            for vehicle in range(number_vehicles)), name=f'const1b_order{order-1}')
        
    # Add makespan constraints
    model.addConstrs(
            (makespan[0] >= t[(number_orders-1, vehicle)] + gp.quicksum([ \
            x[(subset, vehicle)] * mapping_subsets_to_routing_times[subset]\
            for subset in mapping_max_order_to_subsets[number_orders-1]]) \
            for vehicle in range(number_vehicles)), name='const1c')
    
    # Add coverage constraint for all orders
    model.addConstrs(
        (gp.quicksum([x[tupled] for tupled in x if tupled[0] in 
        mapping_orders_to_subsets_they_appear[order]]) >= 1 \
        for order in range(number_orders)), name='const1d')
    return

def _create_variables_constraints_large_scale_naive_formulation(
        model: gp.Model,
        number_vehicles: int,
        number_orders: int,
        arrivals: np.ndarray,
        mapping_subsets_to_routing_times: dict[tuple[int], float],
        mapping_max_order_to_subsets: dict[int, set[tuple[int]]],
        mapping_orders_to_subsets_they_appear: dict[int, set[tuple[int]]],
        lp_relaxation: bool
        ):
    mapping_subsets_to_ids = dict()
    for new_id, subset in enumerate(mapping_subsets_to_routing_times):
        mapping_subsets_to_ids[subset] = new_id
    indices_t = [(order, vehicle) for order in range(number_orders) \
        for vehicle in range(number_vehicles) ]
    indices_x = [(mapping_subsets_to_ids[subset], vehicle) \
        for vehicle in range(number_vehicles) \
        for subset in mapping_subsets_to_routing_times]
    index_makespan = [0]
    # Add the variables
    t = model.addVars(indices_t, name='time')
    if lp_relaxation:
        x = model.addVars(indices_x, name='x')
    else:
        x = model.addVars(indices_x, name='x', vtype=gp.GRB.BINARY)    
    makespan = model.addVars(index_makespan, name='makespan')
    # Set objective value
    model.setObjective(makespan[0], gp.GRB.MINIMIZE)

    # Add departure constraints
    model.addConstrs( 
        (t[tupled] >= arrivals[tupled[0]] for tupled in t), name=f'const1a')
    
    # Add vehicle previous dispatch constraints
    for order in range(1, number_orders):
        model.addConstrs(
            (t[(order, vehicle)] >= t[(order-1, vehicle)] + gp.quicksum([ \
            x[(mapping_subsets_to_ids[subset], vehicle)] * \
            mapping_subsets_to_routing_times[subset]\
            for subset in mapping_max_order_to_subsets[order-1]]) \
            for vehicle in range(number_vehicles)), name=f'const1b_order{order-1}')
    
    # Add makespan constraints
    model.addConstrs(
            (makespan[0] >= t[(number_orders-1, vehicle)] + gp.quicksum([ \
            x[(mapping_subsets_to_ids[subset], vehicle)] * mapping_subsets_to_routing_times[subset]\
            for subset in mapping_max_order_to_subsets[number_orders-1]]) \
            for vehicle in range(number_vehicles)), name='const1c')
    
    # Add coverage constraint for all orders
    model.addConstrs(
        (gp.quicksum([x[(mapping_subsets_to_ids[subset], vehicle)] for\
            subset in mapping_orders_to_subsets_they_appear[order] for\
            vehicle in range(number_vehicles)]) >= 1 \
        for order in range(number_orders)), name='const1d')
    return

def _create_variables_constraints_low_symmetry_formulation(
        model: gp.Model,
        number_vehicles: int,
        number_orders: int,
        arrivals: np.ndarray,
        mapping_subsets_to_routing_times: dict[tuple[int], float],
        mapping_max_order_to_subsets: dict[int, set[tuple[int]]],
        mapping_orders_to_subsets_they_appear: dict[int, set[tuple[int]]],
        fixed_cost: float,
        processing_times_orders: np.ndarray,
        family_setup_times: np.ndarray,
        lp_relaxation: bool
        ):
    big_m = np.sum(processing_times_orders) + np.sum(family_setup_times) + fixed_cost
    indices_t = [(order, vehicle) for order in range(number_orders)\
        for vehicle in range(number_vehicles) ]
    indices_w = [(order, vehicle) for vehicle in range(number_vehicles) \
        for order in range(number_orders)]
    indices_y = [(order, vehicle) for vehicle in range(number_vehicles) \
        for order in range(number_orders)]
    indices_x = [subset for subset in mapping_subsets_to_routing_times]
    index_makespan = [0]
    # Add the variables
    t = model.addVars(indices_t, name='time')
    w = model.addVars(indices_w, name='time')
    if lp_relaxation:
        x = model.addVars(indices_x, name='x')
        y = model.addVars(indices_y, name='y')
    else:
        x = model.addVars(indices_x, name='x', vtype=gp.GRB.BINARY)
        y = model.addVars(indices_y, name='y', vtype=gp.GRB.BINARY)    
    makespan = model.addVars(index_makespan, name='makespan')
    # Set objective value
    model.setObjective(makespan[0], gp.GRB.MINIMIZE)

    """ Add constraints (number is matched with paper) """
    # Add departure constraints
    model.addConstrs( 
        (t[tupled] >= arrivals[tupled[0]] for tupled in t), name='const2a')
    
    # Add vehicle previous dispatch constraints
    for order in range(1, number_orders):
        model.addConstrs(
            (t[(order, vehicle)] >= t[(order-1, vehicle)] + w[(order-1, vehicle)] \
            for vehicle in range(number_vehicles)), name=f'const2b_order{order}')
        
    # Add makespan constraints
    model.addConstrs(
            (makespan[0] >= t[(number_orders-1, vehicle)] + \
            w[(number_orders-1, vehicle)] for vehicle in range(number_vehicles)),
            name='const2c')
    
    # Add workload division constraints
    model.addConstrs(
        (gp.quicksum([w[(order, vehicle)] for vehicle in range(number_vehicles)])\
        == gp.quicksum([x[subset]*mapping_subsets_to_routing_times[subset] \
        for subset in mapping_max_order_to_subsets[order]]) 
        for order in range(number_orders)), name='const2d')
    model.addConstrs(
        (w[(order, vehicle)] <= y[(order, vehicle)]*big_m for order in \
        range(number_orders) for vehicle in range(number_vehicles)), name='const2e')
    model.addConstrs(
        (gp.quicksum([y[(order, vehicle)] for vehicle in range(number_vehicles)])\
        <= 1 for order in range(number_orders)), name='const2f')
    
    # Add coverage constraint for each order
    model.addConstrs(
        (gp.quicksum([x[s] for s in mapping_orders_to_subsets_they_appear[order]])\
        >= 1 for order in range(number_orders)), name='const2g')
    return

def _create_variables_constraints_flow_formulation(
        model: gp.Model,
        number_vehicles: int,
        number_orders: int,
        arrivals: np.ndarray,
        mapping_subsets_to_routing_times: dict[tuple[int], float],
        mapping_max_order_to_subsets: dict[int, set[tuple[int]]],
        mapping_orders_to_subsets_they_appear: dict[int, set[tuple[int]]],
        fixed_cost: float,
        processing_times_orders: np.ndarray,
        family_setup_times: np.ndarray,
        lp_relaxation: bool
        ):
    big_m = np.sum(processing_times_orders) + np.sum(family_setup_times) + fixed_cost
    indices_t = [order for order in range(number_orders + 1)]
    indices_y = [(order_i, order_j) for order_i in range(-1, number_orders)\
        for order_j in range(order_i+1, number_orders+1) \
        if order_j-order_i <= number_orders]
    indices_x = [subset for subset in mapping_subsets_to_routing_times]

    # Add the variables
    t = model.addVars(indices_t, name='time')
    if lp_relaxation:
        x = model.addVars(indices_x, name='x')
        y = model.addVars(indices_y, name='y')
    else:
        x = model.addVars(indices_x, name='x', vtype=gp.GRB.BINARY)
        y = model.addVars(indices_y, name='y', vtype=gp.GRB.BINARY)  

    # Set objective value
    model.setObjective(t[number_orders], gp.GRB.MINIMIZE)

    """ Add constraints """
    # Subsets depart after all orders have arrived
    model.addConstrs( 
        (t[order] >= arrivals[order] for order in range(number_orders)), name='const3a')
    
    # Flow constraint at origin
    model.addConstr(
        gp.quicksum([y[(-1, order)] for order in range(number_orders)]) == \
        number_vehicles, name='const3b')
    
    # Flow constraint at sink
    model.addConstr(
        gp.quicksum([y[(order, number_orders)] \
        for order in range(number_orders)]) == number_vehicles, name='const3c')
    
    # Vector x is a partition
    model.addConstrs(
        (gp.quicksum([x[s] for s in mapping_orders_to_subsets_they_appear[order]])\
        == 1 for order in range(number_orders)), name='const3d')
    
    # Flow is linked to vector x
    model.addConstrs(
        (gp.quicksum([x[s] for s in mapping_orders_to_subsets_they_appear[order]])\
        == gp.quicksum([y[(order_j, order)] for order_j in range(-1, order)])\
        for order in range(number_orders)), name='const3e')
    
    # Flow conservation for each order
    model.addConstrs(
        (gp.quicksum([y[(order_j, order)] for order_j in range(-1, order)])\
        == gp.quicksum([y[(order, order_j)] for order_j in range(order+1, number_orders+1)])\
        for order in range(number_orders)), name='const3f')
    
    # Time variables
    model.addConstrs(
        (t[order_j] - t[order] + (1-y[(order, order_j)])*big_m >=\
        gp.quicksum([mapping_subsets_to_routing_times[s]*x[s] for s in \
        mapping_max_order_to_subsets[order]]) for order in range(number_orders)\
        for order_j in range(order+1, number_orders+1)), name='const3g')
    return