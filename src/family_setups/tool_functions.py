# -*- coding: utf-8 -*-
"""
@author: Ignacio Erazo
"""
from __future__ import annotations
from itertools import chain, combinations
import numpy as np
from collections import defaultdict

""" Tool functions """
def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(1, len(s)+1))

def create_input_family_setups(
        number_orders: int,
        number_families: int,
        distribution_orders: callable=np.random.randint,
        params_distribution_orders: dict[int]={'low': 1, 'high': 50},
        distribution_families: callable=np.random.randint,
        params_distribution_families: dict[int]={'low': 1, 'high': 50},
        distribution_interarrivals: callable=np.random.randint,
        params_distribution_interarrivals: dict[int]={'low': 1, 'high': 50}
        ):
    """This function creates the complete input for family setup instances

    Args:
        number_orders (int): number of orders for the instance.
        number_families (int): number of families in the instance
        distribution_orders (callable, optional): distribution used to
            create the processing time function of each individual order. 
            Defaults to np.random.randint.
        params_distribution_orders (dict[int], optional): dictionary that
            carries over the parameters to be used with the distribution_orders
            function. Defaults to {'low': 1, 'high': 50}.
        distribution_families (callable, optional): distribution used to
            create the setup time for each of the families. 
            Defaults to np.random.randint.
        params_distribution_orders (dict[int], optional): dictionary that
            carries over the parameters to be used with the distribution_families
            function. Defaults to {'low': 1, 'high': 50}.
        distribution_interarrivals (callable, optional): distribution used to
            create the arrival process of the orders. 
            Defaults to np.random.randint.
        params_distribution_interarrivals (dict[int], optional): dictionary that
            carries over the parameters to be used with the distribution_interarrivals
            function. Defaults to {'low': 1, 'high': 50}.

    Returns the inputs for our family setup problems
    """
    if number_orders < number_families:
        print('Number of orders cannot be smaller than number of families')
        number_orders = number_families
        print(f'We set {number_orders = } and continue the algorithm')
    # Compute arrivals vector
    arrivals_n = distribution_interarrivals(
        **params_distribution_interarrivals, size=number_orders-1)
    arrivals = np.zeros(number_orders)
    arrivals[1:]= np.cumsum(arrivals_n)
    # Compute families setups
    family_setup_times = distribution_families(
        **params_distribution_families, size=number_families)
    # Compute processing times for orders
    processing_times_orders = distribution_orders(
        **params_distribution_orders, size=number_orders)
    # Assigns a family to each order
    orders_to_families_assignment = np.random.choice(number_families, size=number_orders,
        replace=True)
    # Makes sure each family has at least one order
    fixed_assignments = np.random.choice(number_orders, size=number_families,
        replace=False)
    for family, order in enumerate(fixed_assignments):
        orders_to_families_assignment[order]=family
    return arrivals, family_setup_times, processing_times_orders, \
        orders_to_families_assignment

def _compute_routing_time_families(
        order_list: list[int],
        processing_times_orders: np.ndarray,
        family_setup: float,
        fixed_cost: float
        ) -> float:
    """
    Computes the dispatching time for a batch = order_list of orders in the 
    same family.
    """
    total_time = np.sum(processing_times_orders[order_list])
    return total_time + family_setup + fixed_cost

""" This function can also be used to compute a lower bound """
def _compute_routing_time_multiple_families_one_subset(
        order_list,
        processing_times_orders: np.ndarray, 
        orders_to_families_assignment: np.ndarray,
        family_setup_times: np.ndarray,
        fixed_cost: float
        ):
    families_spawned = set(np.unique(orders_to_families_assignment[order_list]))
    total_routing_time = np.sum(processing_times_orders[order_list]) + \
        np.sum(family_setup_times[sorted(families_spawned)]) + fixed_cost
    return total_routing_time

def _add_order_subset_to_problem(
        order: int,
        mini_list: list[int],
        processing_times_orders: np.ndarray,
        family_setup: float,
        fixed_cost: float,
        mapping_subsets_to_routing_times: dict[tuple[int], float],
        mapping_max_order_to_subsets: dict[int, list[tuple[int]]],
        mapping_orders_to_subsets_they_appear: dict[int, list[tuple[int]]]
        ) -> None:
    """
    Adds a given batch of orders subset from the same family setup, into the 
    data structures leveraged by our models:
        - mapping_subsets_to_routing_times: dict[tuple[int], float],
        - mapping_max_order_to_subsets: dict[int, list[tuple[int]]],
        - mapping_orders_to_subsets_they_appear: dict[int, list[tuple[int]]]
    """
    routing_time = _compute_routing_time_families(
        order_list=mini_list, 
        processing_times_orders=processing_times_orders,
        family_setup=family_setup,
        fixed_cost=fixed_cost)
    tupled_list = tuple(mini_list)
    mapping_subsets_to_routing_times[tupled_list] = routing_time
    mapping_max_order_to_subsets[order].add(tupled_list)
    for value in tupled_list:
        mapping_orders_to_subsets_they_appear[
            value].add(tupled_list)
    return

def _add_multiple_family_subset_to_problem(
        order: int,
        mini_list: list[int],
        processing_times_orders: np.ndarray,
        orders_to_families_assignment,
        family_setup_times,
        fixed_cost: float,
        mapping_subsets_to_routing_times: dict[tuple[int], float],
        mapping_max_order_to_subsets: dict[int, list[tuple[int]]],
        mapping_orders_to_subsets_they_appear: dict[int, list[tuple[int]]]
        ):
    """
    Adds a given batch of orders subset from different families, into the 
    data structures leveraged by our models:
        - mapping_subsets_to_routing_times: dict[tuple[int], float],
        - mapping_max_order_to_subsets: dict[int, list[tuple[int]]],
        - mapping_orders_to_subsets_they_appear: dict[int, list[tuple[int]]]
    """
    routing_time = _compute_routing_time_multiple_families_one_subset(
        order_list=mini_list,
        processing_times_orders=processing_times_orders, 
        orders_to_families_assignment=orders_to_families_assignment,
        family_setup_times=family_setup_times,
        fixed_cost=fixed_cost
        )
    tupled_list = tuple(mini_list)
    mapping_subsets_to_routing_times[tupled_list] = routing_time
    mapping_max_order_to_subsets[order].add(tupled_list)
    for value in tupled_list:
        mapping_orders_to_subsets_they_appear[
            value].add(tupled_list)
    return





""" Functions to initialize subsets for full/colgen formulations """
def _initialize_all_subsets_families(
        number_orders: int,
        number_families: int,
        orders_to_families_assignment: dict[int, list[int]],
        # The two below are basically dictionaries since the lists are indexed by index
        family_setup_times: list[int, float],
        processing_times_orders: list[int],
        fixed_cost: float
        ):
    """
    Initializes all the subsets for each of the given families
    """
    dictionary_families_to_orders = {
        family: [index for index in range(number_orders) if \
        orders_to_families_assignment[index] == family] \
        for family in range(number_families)}
    mapping_orders_to_subsets_they_appear = defaultdict(lambda: set())
    mapping_max_order_to_subsets = defaultdict(lambda: set())
    mapping_subsets_to_routing_times = dict()
    for family in range(number_families):
        family_setup = family_setup_times[family]
        for index, order in enumerate(dictionary_families_to_orders[family]):
            # Initialize the singleton set
            _add_order_subset_to_problem(
                order=order,
                mini_list=[order],
                processing_times_orders=processing_times_orders,
                family_setup=family_setup,
                fixed_cost=fixed_cost,
                mapping_subsets_to_routing_times=mapping_subsets_to_routing_times,
                mapping_max_order_to_subsets=mapping_max_order_to_subsets,
                mapping_orders_to_subsets_they_appear=mapping_orders_to_subsets_they_appear
                )
            # Initialize all other sets
            new_sets = list(powerset(dictionary_families_to_orders[family][:index]))
            for mini_set in new_sets:
                mini_list = list(mini_set)
                mini_list.append(order)
                _add_order_subset_to_problem(
                    order=order,
                    mini_list=mini_list,
                    processing_times_orders=processing_times_orders,
                    family_setup=family_setup,
                    fixed_cost=fixed_cost,
                    mapping_subsets_to_routing_times=mapping_subsets_to_routing_times,
                    mapping_max_order_to_subsets=mapping_max_order_to_subsets,
                    mapping_orders_to_subsets_they_appear=mapping_orders_to_subsets_they_appear
                    )
    return mapping_orders_to_subsets_they_appear, mapping_max_order_to_subsets, \
        mapping_subsets_to_routing_times

def _initialize_singletons_families(
        number_orders: int,
        orders_to_families_assignment: dict[int, list[int]],
        family_setup_times: list[int, float],
        processing_times_orders: list[int],
        fixed_cost: float
        ):
    """
    Initializes only the singletons subsets
    """
    mapping_orders_to_subsets_they_appear = defaultdict(lambda: set())
    mapping_max_order_to_subsets = defaultdict(lambda: set())
    mapping_subsets_to_routing_times = dict()
    for order in range(number_orders):
        family_of_order = orders_to_families_assignment[order]
        family_setup = family_setup_times[family_of_order]
        _add_order_subset_to_problem(
            order=order,
            mini_list=[order],
            processing_times_orders=processing_times_orders,
            family_setup=family_setup,
            fixed_cost=fixed_cost,
            mapping_subsets_to_routing_times=mapping_subsets_to_routing_times,
            mapping_max_order_to_subsets=mapping_max_order_to_subsets,
            mapping_orders_to_subsets_they_appear=mapping_orders_to_subsets_they_appear
            )
    return mapping_orders_to_subsets_they_appear, mapping_max_order_to_subsets, \
        mapping_subsets_to_routing_times

def _initialize_interval_variables_families(
        number_orders: int,
        orders_to_families_assignment: dict[int, list[int]],
        family_setup_times: list[int, float],
        processing_times_orders: list[int],
        fixed_cost: float
        ):
    """
    Initializes all the interval batches
    """
    mapping_orders_to_subsets_they_appear = defaultdict(lambda: set())
    mapping_max_order_to_subsets = defaultdict(lambda: set())
    mapping_subsets_to_routing_times = dict()
    all_orders = [ord for ord in range(number_orders)]
    for order in range(number_orders):
        for order_j in range(order+1, number_orders+1):
            mini_list = all_orders[order: order_j]
            _add_multiple_family_subset_to_problem(
                order=order_j-1,
                mini_list=mini_list,
                processing_times_orders=processing_times_orders,
                orders_to_families_assignment=orders_to_families_assignment,
                family_setup_times=family_setup_times,
                fixed_cost=fixed_cost,
                mapping_subsets_to_routing_times=mapping_subsets_to_routing_times,
                mapping_max_order_to_subsets=mapping_max_order_to_subsets,
                mapping_orders_to_subsets_they_appear=mapping_orders_to_subsets_they_appear
                )
    return mapping_orders_to_subsets_they_appear, mapping_max_order_to_subsets, \
        mapping_subsets_to_routing_times
        