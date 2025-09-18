# -*- coding: utf-8 -*-
"""
@author: Ignacio Erazo
"""
from __future__ import annotations
import numpy as np

from src.family_setups.tool_functions import _compute_routing_time_multiple_families_one_subset


def lower_bound_general_in_paper_families(
        number_orders: int,
        number_vehicles: int,
        arrivals: np.ndarray, 
        processing_times_orders: np.ndarray, 
        orders_to_families_assignment: np.ndarray,
        family_setup_times: np.ndarray,
        fixed_cost: int=0
        ):
    """
    Computes a general lower bound for the problem.
    This corresponds to the lower bound of Proposition 5 of the paper.
    """
    list_orders = [order for order in range(number_orders)]
    best_lower_bound = 0
    for order in range(number_orders):
        lower_bound = arrivals[order] + \
            _compute_routing_time_multiple_families_one_subset(
                order_list=list_orders[order:],
                processing_times_orders=processing_times_orders, 
                orders_to_families_assignment=orders_to_families_assignment,
                family_setup_times=family_setup_times,
                fixed_cost=fixed_cost
                ) / min(number_vehicles, number_orders - order)
        best_lower_bound = max(best_lower_bound, lower_bound)
    return best_lower_bound


def lower_bound_improved_families(
        number_orders: int,
        number_vehicles: int,
        arrivals: np.ndarray, 
        processing_times_orders: np.ndarray, 
        orders_to_families_assignment: np.ndarray,
        family_setup_times: np.ndarray,
        fixed_cost: int=0
        ):
    """
    Computes a stronger lower bound that leverages the structure of the 
    family setups problem.
    This corresponds to the lower bound used for the experiments, explained in
    Appendix E.3
    """
    list_orders = [order for order in range(number_orders)]
    best_lower_bound = 0
    for order in range(number_orders):
        families_spawned = len({orders_to_families_assignment[o] \
            for o in range(order, number_orders)})
        maximum_vehicles_feasible_to_use = min(number_vehicles, number_orders - order)
        if families_spawned < maximum_vehicles_feasible_to_use:
            routing_component = _compute_extra_workload_because_of_divisions_family_setups(
                order_list=list_orders[order:],
                number_families_spawned=families_spawned,
                vehicles_to_divide_workload_into=maximum_vehicles_feasible_to_use,
                orders_to_families_assignment=orders_to_families_assignment,
                family_setup_times=family_setup_times,
                )
        else:
            routing_component = 0
        lower_bound = arrivals[order] + \
            _compute_routing_time_multiple_families_one_subset(
                order_list=list_orders[order:],
                processing_times_orders=processing_times_orders, 
                orders_to_families_assignment=orders_to_families_assignment,
                family_setup_times=family_setup_times,
                fixed_cost=fixed_cost
                ) / maximum_vehicles_feasible_to_use +\
                routing_component / maximum_vehicles_feasible_to_use
        best_lower_bound = max(best_lower_bound, lower_bound)
    return best_lower_bound

def _compute_extra_workload_because_of_divisions_family_setups(
        order_list: list[int],
        number_families_spawned: int,
        vehicles_to_divide_workload_into: int,
        orders_to_families_assignment: np.ndarray,
        family_setup_times: np.ndarray
        ):
    """
    Subroutine to compute the lower bound from appendix E.3
    """
    actual_assignments = orders_to_families_assignment[order_list]
    families_found, count_families = np.unique(actual_assignments, return_counts=True) 
    eligible_families = [families_found[index] for index in \
        range(len(families_found)) if count_families[index] >= 2]
    eligible_counts = [value for value in count_families if value >= 2]
    number_of_adjustments = vehicles_to_divide_workload_into - number_families_spawned
    extra_workload = 0
    while number_of_adjustments > 0:
        # Find family with minimum setup time
        eligible_family_setups = family_setup_times[eligible_families]
        family_of_min = np.argmin(eligible_family_setups)
        extra_workload += np.min(eligible_family_setups)
        number_of_adjustments -= 1
        # Update counts of families
        eligible_counts[family_of_min] -= 1
        eligible_families = [eligible_families[index] for index in \
            range(len(eligible_families)) if eligible_counts[index] >= 2]
        eligible_counts = [value for value in eligible_counts if value >= 2]
    return extra_workload
