# -*- coding: utf-8 -*-
"""
@author: Ignacio Erazo
"""
from __future__ import annotations
import gurobipy as gp

def _retrieve_dual_information_naive_formulation(
        model_lp: gp.Model
        ):
    """
    This function retrieves the dual data needed to perform column generation
    for the naive formulation.
    """
    alpha = []
    beta = []
    gamma = []
    all_names = []
    for c in model_lp.getConstrs():
        if c.constrName[6] == 'a':
            alpha.append(c.pi)
        if c.constrName[6] in {'b', 'c'}:
            # They come out ordered by order-> all vehicles, order+1 -> ...
            beta.append(c.pi)
        if c.constrName[6] == 'd':
            gamma.append(c.pi)
        all_names.append(c.constrName)
    return alpha, beta, gamma

def _retrieve_dual_information_low_symmetry_formulation(
        model_lp: gp.Model
        ):
    """
    This function retrieves the dual data needed to perform column generation
    for the low symmetry.
    """
    alpha = []
    indices_alpha = []
    beta = []
    indices_beta = []
    gamma = []
    indices_gamma = []
    epsilon = []
    pi = []
    all_names = []
    for c in model_lp.getConstrs():
        if c.constrName[6] == 'a':
            alpha.append(c.pi)
            indices_alpha.append(c.constrName)
        if c.constrName[6] in {'b', 'c'}:
            # They come out ordered by order-> all vehicles, order+1 -> ...
            beta.append(c.pi)
            indices_beta.append(c.constrName)
        if c.constrName[6] == 'd':
            epsilon.append(c.pi)
        if c.constrName[6] == 'f':
            pi.append(c.pi)
        if c.constrName[6] == 'g':
            gamma.append(c.pi)
            indices_gamma.append(c.constrName)
        all_names.append(c.constrName)
    return alpha, beta, epsilon, pi, gamma
    
def _retrieve_dual_information_flow_formulation(
        model_lp: gp.Model
        ):
    """
    This function retrieves the dual data needed to perform column generation
    for the flow-based formulation.
    """
    epsilon = []
    alpha = []
    beta = []
    gamma = []
    pi = []
    all_names = []
    for c in model_lp.getConstrs():
        if c.constrName[6] == 'a':
            alpha.append(c.pi)
        if c.constrName[6] == 'b':
            epsilon_minus = c.pi
        if c.constrName[6] == 'c':
            epsilon_plus = c.pi
        if c.constrName[6] == 'd':
            gamma.append(c.pi)
        if c.constrName[6] == 'e':
            epsilon.append(c.pi)
        if c.constrName[6] == 'f':
            pi.append(c.pi)
        if c.constrName[6] == 'g':
            beta.append(c.pi)
        all_names.append(c.constrName)
    return alpha, beta, epsilon, pi, gamma, epsilon_minus, epsilon_plus