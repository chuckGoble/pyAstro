# -*- coding: utf-8 -*-
import numpy as np
from numba import *


@jit(nopython=True)
def distance(r):
    """
    just the standard euclidean distance fromula, returns the scalar
    """
    return np.sqrt(r[0]**2 + r[1]**2 + r[2]**2)

@jit(nopython=True)
def cos_law(s1, s2, angle):
    """
    cosine law returns the final side
    """
    return np.sqrt(s1**2.0 + s2**2.0 - 2.0*s1*s2*np.cos(angle))


@jit(nopython=True)
def battin(mu, r1, r2, dt, tm=1, orbit_type=1):
    """
    Lambert solver using Battin's method from fundamentals of
    astrodynamics and applications
    ---
    mu -- central body parameter
    r1 -- position vector to point one in space
    r2 -- position vector to point two in space
    dt -- transfer time
    tm -- +1 or -1 depending on whther its a short or long way transfer
    ---
    returns the two delta v changes necessary to achieve orbit
    """
    r1m = distance(r1)
    r2m = distance(r2)
    # angles and convenience parameters
    c_nu = np.dot(r1, r2)/(r1m*r2m)
    s_nu = tm*np.sqrt(1 - c_nu**2.0)
    nu = np.arctan2(s_nu, c_nu)
    if nu < 0:
        nu = nu + (2*np.pi)
    s_nu2 = np.sin(nu/4)**2.0
    c_nu2 = np.cos(nu/4)**2.0
    c = cos_law(r1m, r2m, nu)
    s = (r1m + r2m + c)/2
    eps = (r2m - r1m)/r1m
    t2w = (eps**2.0/4.0)/(np.sqrt(r2m/r1m) + r2m/r1m*(2.0 + np.sqrt(r2m/r1m)))
    rop = np.sqrt(r1m*r2m)*(c_nu2 + t2w)

    if nu < np.pi:
        tp = s_nu2 + t2w
        btm = s_nu2 + t2w + np.cos(nu/2.0)
        l = tp/btm
    else:
        tp = c_nu2 + t2w - np.cos(nu/2.0)
        btm = c_nu2 + t2w
        l = tp/btm

    m = mu*dt**2.0/(8*rop**3)

    if orbit_type == 1:
        x = l
    else:
        x = 0

    err = 10  # error
    tol = 1e-6  # tolerance
    iters = 0
    iters_max = 100
    # coefficients
    c1 = 8.0
    c2 = 1.0
    c3 = 9.0/7.0
    c4 = 16.0/63.0
    c5 = 25.0/99.0
    while err > tol:
        eta = x/(np.sqrt(1.0 + x) + 1.0)**2.0
        xi = c1*(np.sqrt(1.0 + x) + 1.0)/(3.0 + c2/(5.0 + eta + c3*eta/(1.0 + c4*eta/(1.0 + c5*eta))))
        bt1 = (1.0 + 2*x + l)
        bt2 = (4.0*x + xi*(3.0 + x))
        h1 = (l + x)**2.0*(1.0 + 3.0*x + xi)/(bt1*bt2)
        h2 = (m*(x - l + xi))/(bt1*bt2)
        B = 27.0*h2/(4.0*(1.0 + h1)**3.0)
        U = B/(2.0*(np.sqrt(1.0 + B) + 1.0))
        K = (1/3)/(1 + (4/27)*U/(1 + (8/27)*U/(1 + (700/2907)*U)))
        y = (1+h1)/3*(2 + np.sqrt(1 + B)/(1 + 2*U*K**2.0))
        xn = np.sqrt(((1-l)/2)**2.0 + m/y**2) - (1 + l)/2
        err = np.abs(xn - x)
        x = xn
        iters += 1
        if iters > iters_max:
            break
    # semi major axis
    a = mu*(dt**2.0)/(16*rop**2*x*y**2.0)
    if a > 0:
        s_beta = np.sqrt((s - c)/(2*a))
        beta = 2.0*np.arcsin(s_beta)
        if nu > np.pi:
            beta = -beta
        a_min = s/2
        tmin = np.sqrt(a_min**3.0/mu)*(np.pi - beta + np.sin(beta))
        alpha = 2.0*np.arcsin(np.sqrt(s/(2*a)))
        if dt > tmin:
            alpha = 2*np.pi - alpha
        dE = alpha - beta
        f = 1 - a/r1m*(1 - np.cos(dE))
        g = dt - np.sqrt(a**3.0/mu)*(dE - np.sin(dE))
        gdot = 1 - a/r2m*(1 - np.cos(dE))
    else:
        alpha = 2.0*np.arcsinh(np.sqrt(s/(-2*a)))
        beta = 2.0*np.arcsinh(np.sqrt((s-c)/(-2*a)))
        dH = alpha - beta
        f = 1 - a/r1m*(1 - np.cosh(dH))
        g = dt - np.sqrt(-a**3.0/mu)*(np.sinh(dH) - dH)
        gdot = 1 - a/r2m*(1 - np.cosh(dH))

    v1 = (r2 - f*r1)/g
    v2 = (gdot*r2 - r1)/g
    return v1, v2
