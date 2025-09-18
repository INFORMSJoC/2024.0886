# -*- coding: utf-8 -*-
"""
@author: Ignacio Erazo
"""
from __future__ import annotations

def arrivals_constant_rate(
        num_orders: int
        ) -> list[int]:
    """
    Function that returns all orders arriving every 1 unit of time.
    """
    arrivals=[i for i in range(1, num_orders + 1)]
    return arrivals


def arrivals_inhomogeneous_early(
        num_orders: int,
        num_orders_early: int,
        expected_time_between_orders_early: float,
        expected_time_between_orders_late: float
        ) -> list[int]:
    arrivals = []
    start_time = 0
    # Initialize the arrival for early orders
    for _ in range(0, num_orders_early):
        start_time += expected_time_between_orders_early
        arrivals.append(start_time)
    # Initialize the arrival for all other orders
    for _ in range(num_orders_early, num_orders):
        start_time += expected_time_between_orders_late
        arrivals.append(start_time)
    return arrivals


def arrivals_inhomogeneous_u_shaped(
        num_orders: int,
        num_orders_early: int,
        expected_time_between_orders_early: float,
        num_orders_mid_day: int,
        expected_time_between_orders_mid_day: float,
        expected_time_between_orders_late: float
        ) -> list[int]:
    arrivals = []
    start_time = 0
    # Initialize the arrival for early orders
    for _ in range(0, num_orders_early):
        start_time += expected_time_between_orders_early
        arrivals.append(start_time)
    # Initialize the arrival for mid day orders
    for _ in range(num_orders_early, num_orders_early + num_orders_mid_day):
        start_time += expected_time_between_orders_mid_day
        arrivals.append(start_time)
    # Initialize the arrival for all the other orders
    for _ in range(num_orders_early + num_orders_mid_day, num_orders):
        start_time += expected_time_between_orders_late
        arrivals.append(start_time)
    return arrivals