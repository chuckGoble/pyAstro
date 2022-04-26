# -*- coding: utf-8 -*-
"""This python module is a custom module that has functions to perform fast
math operations that are just-in-time compiled.

Function List
-------------
dot: Dot product or inner product operation on set of 1x3 array

cross: Cross product or outer product operation on set of 1x3 array

norm: Compute the magnitude of a 1x3 array

sign: Determine and return the sign of a number
"""


import math as m
from numba import jit


# COMMON OPERATIONS
# ************************************************************************** #
@jit(nopython=True)
def dot(vec_01, vec_02):
    """My dot product function.

    Parameters
    ----------
    vec_01: 1x3 array
        3 element array for dot product calculation
    vec_02: 1x3 array
        3 element array for dot product calculation

    Returns
    -------
    dot_product: float
        Result scalar of the dot product calculation
    """
    # Calculate the dot product
    dot_product = (vec_01[0]*vec_02[0]) + (vec_01[1]*vec_02[1]) + (vec_01[2]*vec_02[2])
    return dot_product


@jit(nopython=True)
def cross(vec_01, vec_02):
    """My cross product function.

    Parameters
    ----------
    vec_01: 1x3 array
        3 element array for cross product calculation
    vec_02: 1x3 array
        3 element array for cross product calculation

    Returns
    -------
    new_vec = 1x3 array
        3 element array solution for cross product calculation
    """
    # Calculate each element independently
    i_comp = (vec_01[1]*vec_02[2]) - (vec_01[2]*vec_02[1])
    j_comp = (vec_01[2]*vec_02[0]) - (vec_01[0]*vec_02[2])
    k_comp = (vec_01[0]*vec_02[1]) - (vec_01[1]*vec_02[0])

    # Build the resulting vector
    new_vec = [i_comp, j_comp, k_comp]
    return new_vec


@jit(nopython=True)
def norm(vector):
    """My Euclidean norm function.

    Parameters
    ----------
    vector: 1x3 array
        3 element array to find the magnitude

    Returns
    -------
    mag: float
        Magnitude of the input vector
    """
    mag = m.sqrt((vector[0]**2) + (vector[1]**2) + (vector[2]**2))
    return mag


@jit(nopython=True)
def sign(var):
    """Returns the sign of an input variable.

    Parameters
    ----------
    var: float or signed int
        variable to check the positive/negative sign

    Returns
    -------
    var: signed int
        signed integer of input variable
    """
    if var < 0:
        var = -1
    else:
        var = 1
    return var
# ************************************************************************** #
