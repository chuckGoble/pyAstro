# -*- coding: utf-8 -*-
"""Module computes a generic Hohmann Transfer."""
import math as m
from numba import jit


@jit(nopython=True)
def norm(x):
    """Compute the magnitude of the position or velocity vector."""
    return m.sqrt((x[0]**2) + (x[1]**2) + (x[2]**2))


@jit(nopython=True)
def patchedConics(stateDep, stateArr, GMsun, GMdep, GMarr, rPark, rScience):
    """
    Calcuate a Patched Conics Transfer using Hohmann Transfer as the Heliocentric Transfer.

    Parameters
    ----------
    stateDep : array
         stateDep is the state vector of the departure planet as observed by
         the Sun

    stateArr : array
         stateArr is the state vector of the arrival planet as observed by
         the Sun

    GMsun : float
            GMsun is the gravitational parameter for the sun to be used during
            the heliocentric phase of the Hohmann transfer

    GMdep : float
            GMdep is the gravitational parameter for the departure planet to be
            used during the departure phase of the problem

    GMarr : float
            GMarr is the gravitational parameter for the arrival planet to be
            used during the arrival/capture phase of the problem

    rPark : float
    rScience : float

    Returns
    -------
    dV_dep : float
    dV_arr : float
    dV_tot : float
    """
    # Solve the Transfer Ellipse Trajectory
    r1 = norm(stateDep[:3])  # magnitude of the departure planet position
    v1 = norm(stateDep[3:])  # magnitude of the departure planet velocity
    r2 = norm(stateArr[:3])  # magnitude of the arrival planet position
    v2 = norm(stateArr[3:])  # magnitude of the arrival planet velocity
    # Semi-major axis of the transfer ellipse in km
    a_trans = 0.5*(r1 + r2)
    # Transfer time: 0.5 the total period in seconds
    tau_trans = m.pi*m.sqrt((a_trans**3)/GMsun)
    # Velocity at periapsis for the transfer
    v_trans_p = m.sqrt(((2*GMsun)/(r1)) - (GMsun/a_trans))
    # Hyperbolic Excess Velocity at Departure Planet's Sphere of Influence
    vInf_depSoi = abs(v_trans_p - v1)
    # Velocity at apoapsis for the transfer
    v_trans_a = m.sqrt(((2*GMsun)/r2) - ((GMsun)/(a_trans)))
    # Hyperbolic Excess Velocity at Arrival Planet's Sphere of Influence
    vInf_arrSoi = abs(v2 - v_trans_a)

    # Solve the Departure Hyperbolic Trajectory
    # Sepcific Mechanical Energy of the Departure Parking Orbit
    sme_dep_park = (-1*GMdep)/(rPark[0] + rPark[1])
    # Velocity on the Parking Orbit
    v_park = m.sqrt(2*(((GMdep)/(rPark[0])) + sme_dep_park))
    # Specific Mechanical Energy of the Hyperbolic Orbit
    sme_hypb_dep = 0.5*m.pow(vInf_depSoi, 2)
    # Velocity at Periapsis of the Departure Hyperbola
    v_atPer_hyp_dep = m.sqrt(2*(((GMdep)/(rPark[0])) + (sme_hypb_dep)))
    dV_dep = abs(v_atPer_hyp_dep - v_park)

    # Solve the Arrival Hyperbolic Trajectory
    # Specific Mechanical Energy of the Arrival Science Orbit
    sme_arr_science = (-1*GMarr)/(2*rScience)
    # Velocity of the Parking Orbit
    v_science = m.sqrt(2*(((GMarr)/(rScience)) + sme_arr_science))
    # Specific Mechanical Energy of the Hyperbolic Orbit
    sme_hypb_arr = 0.5*m.pow(vInf_arrSoi, 2)
    # Velocity at Periapsis of the Arrival Hyperbola
    v_atPer_hyp_arr = m.sqrt(2*(((GMarr)/(rScience)) + (sme_hypb_arr)))
    dV_arr = abs(v_atPer_hyp_arr - v_science)

    # Total Mission Delta Velocity
    dV_tot = dV_dep + dV_arr

    return dV_dep, dV_arr, dV_tot, tau_trans
