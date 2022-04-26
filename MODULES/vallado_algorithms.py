# -*- coding: utf-8 -*-

"""Module Doc String"""

import math as m
import numpy as np
from numba.typed import Dict
from numba import jit, types, njit

##############################################################################
IVEC = np.array([1, 0, 0])
JVEC = np.array([0, 1, 0])
KVEC = np.array([0, 0, 1])
FLOAT_ARRAY = types.float64[:]
##############################################################################


# ************************************************************************** #
@jit(nopython=True)
def dot(a, b):
    """My dot product function."""
    dp = (a[0]*b[0]) + (a[1]*b[1]) + (a[2]*b[2])
    return dp


@jit(nopython=True)
def cross(a, b):
    """My cross product function."""
    ii = (a[1]*b[2]) - (a[2]*b[1])
    jj = (a[2]*b[0]) - (a[0]*b[2])
    kk = (a[0]*b[1]) - (a[1]*b[0])
    return [ii, jj, kk]


@jit(nopython=True)
def norm(a):
    """My Euclidean norm function."""
    return m.sqrt((a[0]**2) + (a[1]**2) + (a[2]**2))


@jit(nopython=True)
def sign(x):
    """Returns the sign of an input"""
    if x < 0:
        return -1
    else:
        return 1
# ************************************************************************** #


# ALGORITHM 01
# TODO: numba vectorize or numpy vectorize this function -- figure this out
@jit(nopython=True)
def find_c2c3(psi):
    """
    Given psi calculate c2 and c3.

    Parameters
    ----------
    psi : float
    what is psi?

    Returns
    -------
    c2, c3 : tuple
    explain what c2 and c3 are

    """
    if psi > 1E-6:
        c2 = (1 - m.cos(m.sqrt(psi)))/(psi)
        c3 = ((m.sqrt(psi)) - (m.sin(psi)))/(m.sqrt((psi**3)))
    elif psi < -1E-6:
        c2 = (1 - m.cosh(m.sqrt(-1*psi)))/(psi)
        c3 = ((m.sinh(-1*psi)) - (m.sqrt(-1*psi)))/((-1*psi)**3)
    else:
        c2 = 1/2
        c3 = 1/6
    return (c2, c3)


# ALGORITHM 02
@jit(nopython=True)
def kep_eqn_e(M, ecc):
    """
    Solve Kepler's problem for an elyptical orbit.

    Parameters
    ----------
    M : float
    M is the mean anamoly. The mean anamoly represents ...

    ecc : float
    ecc is the magnitude of the eccentricity vector and is bounded between
    zero and less than 1

    Returns
    -------
    E : float
    E is the eccentric anamoly
    """
    if -1*m.pi < M < 0 or M > m.pi:
        E = M - ecc
    else:
        E = M + ecc
    err = 1
    while err > 1E-10:
        num = M - E + (ecc*m.sin(E))
        dom = 1 - (ecc*m.cos(E))
        Enp1 = E + num/dom

        err = abs(Enp1 - E)
        E = Enp1
    return m.degrees(E)


# ALGORITHM 03
# TODO: solve the cubic with Cardan's solution on page 1027-1029
@jit(nopython=True)
def kep_eqn_p(dt, p, GM):
    """
    Solve Kepler's equation for a parabolic orbit.

    Parameters
    ----------
    dt : float
    Time since periapsis which can be written (t-T) in seconds

    p : float
    The orbit's semi parameter or semi latus rectum in kilometers

    GM : float
    GM is the gravitational parameter mu for the primary central gravitational
    field.

    Returns
    -------
    B : float
    B is the parabolic orbit's parabolic anomaly.
    """
    mean_motion = 2*m.sqrt((GM)/(p**3))
    # Try Cardan's solution
    # b = -3*np*dt
    # delta = 1 + ((b**2)/4)
    # arg1 = (-1*(b/2)) + m.sqrt(delta)
    # arg2 = (-1*(b/2)) - m.sqrt(delta)
    # B = m.pow(arg1, (1/3)) + m.pow(arg2, (1/3))
    # ----
    def f(B, mean_motion, dt): return ((B**3)/3) + B - (mean_motion*dt)
    def df(B): return (B**2) + 1
    B = m.pi
    err = 1
    while err > 1E-10:
        Bnp1 = B - f(B, mean_motion, dt)/df(B)
        err = abs(Bnp1 - B)
        B = Bnp1
    return m.degrees(B)


# ALGORITHM 04
@jit(nopython=True)
def kep_eqn_h(M, ecc):
    """
    Solve Kepler's equation for a hyperbolic orbit.

    Parameters
    ----------
    M :  float
    M is the mean anamoly of the orbit

    ecc : float
    ecc is the eccentricity of the orbit and should be greater than 1

    Returns
    -------
    H : float
    H is the hyperbolic anomaly for the hyperbolic orbit
    """
    M = m.radians(M)
    if (ecc < 1.6) and (((-1*m.pi) < M < 0) or (M > m.pi)):
        H = M - ecc
    elif (ecc < 1.6) and not (((-1*m.pi) < M < 0) or (M > m.pi)):
        H = M + ecc
    elif (ecc < 3.6) and (abs(M) > m.pi):
        H = M - m.copysign(ecc, M)
    else:
        H = M/(ecc-1)
    err = 1
    while err > 1E-10:
        Hnp1 = H + (M - (ecc*m.sinh(H)) + H)/((ecc*m.cosh(H)) - 1)
        err = abs(Hnp1 - H)
        H = Hnp1
    return m.degrees(H)


# ALGORITHM 05
@jit(nopython=True)
def nu2anomaly(ecc, nu):
    """
    Transform true anomaly into eccentric/parabolic/hyperbolic anomaly.

    Parameters
    ----------
    ecc : float
    ecc is the orbital element eccentricity.

    nu : float
    nu is the orbital element true anomaly.

    Returns
    -------
    return : float
    The function either returns the eccentric, parabolic or hyperbolic
    anamoly depending on the type of orbit that is input into the function.
    """
    nu = m.radians(nu)
    if ecc < 1:
        denom = (1 + (ecc*m.cos(nu)))
        sine_E = (m.sin(nu)*m.sqrt(1-(ecc**2)))/denom
        cosine_E = (ecc + m.cos(nu))/denom
        return m.degrees(m.atan2(sine_E, cosine_E))
    elif ecc == 1:
        return m.tan(nu/2)
    else:
        denom = (1 + (ecc*m.cos(nu)))
        sinh_E = (m.sin(nu)*m.sqrt((ecc**2) - 1))/denom
        cosh_E = (ecc + m.cos(nu))/denom
        return m.degrees(m.atanh(sinh_E/cosh_E))


# ALGORITHM 06
@jit(nopython=True)
def anomaly2nu(ecc, anomaly):
    """
    Transform the eccentric or hyperbolic anomaly to the true anomaly.

    Parameters
    ----------
    ecc : float
    ecc is the orbital element eccentricity

    anomaly : tuple
    anomaly is either the eccentric, parabolic or hyperbolic anomaly

    Returns
    -------
    return : float
    The function returns the true anomaly at that point in the orbit.
    """
    anomaly = m.radians(anomaly)
    if ecc < 1:
        denom = 1 - (ecc*m.cos(anomaly))
        sine_nu = (m.sin(anomaly)*m.sqrt(1 - (ecc**2)))/denom
        cosine_nu = (m.cos(anomaly) - ecc)/denom
    elif ecc > 1:
        denom = 1 - (ecc*m.cosh(anomaly))
        sine_nu = ((-1*m.sin(anomaly))*m.sqrt((ecc**2)-1))/denom
        cosine_nu = (m.cosh(anomaly) - ecc)/denom
    return m.degrees(m.atan2(sine_nu, cosine_nu))


@jit(nopython=True)
def parabolicAnomaly2nu(B, p, r):
    """
    Transform the parabolic anomaly to the true anomaly.

    Parameters
    ----------
    ecc : float
    ecc is the orbital element eccentricity

    anomaly : tuple
    anomaly is either the eccentric, parabolic or hyperbolic anomaly

    p : float
    p is the semiparameter or semilatus rectum of the conic section

    r : float
    r is the current radius

    Returns
    -------
    return : float
    The function returns the true anomaly at that point in the orbit.
    """
    sine_nu = (p*B)/r
    cosine_nu = (p-r)/r
    return m.degrees(m.atan2(sine_nu, cosine_nu))


# ALGORITHM 07
def kepler_coe():

    return


# ALGORITHM 08
def kepler(state_vec, dt, mu, tol=1E-8):
    """
    Propagates a cartesian state vector under two body dynamics for time dt.

    Parameters
    ----------
    state_vec : numpy array of floats
    dt : float
    mu : float
    tol : float

    Returns
    ndarray : numpy array of floats
    """
    sqrt_mu = m.sqrt(mu)
    r0_vec = state_vec[:3]
    v0_vec = state_vec[3:]
    v0_mag = norm(v0_vec)
    r0_mag = norm(r0_vec)
    sme = ((v0_mag**2)/2) - (mu/r0_mag)
    alpha = (-1*sme*2)/mu 

    if alpha > 1E-6:
        chi_n = sqrt_mu*dt*alpha

    elif abs(1.0 - alpha) < tol:
        ang_mom = cross(r0_vec, v0_vec)
        semiparameter = (norm(ang_mom)**2)/mu
        cot_2s = 3*dt*m.sqrt((mu)/(semiparameter**3))
        two_s = m.atan(1/cot_2s)
        s = 0.5 * two_s
        cube_tan_w = m.tan(s)
        tan_w = cube_tan_w**(1/3)
        w = m.atan(tan_w) 
        chi_n = 2*m.sqrt(semiparameter)*m.tan(1/(2*w))
        alpha = 0
    elif alpha < -1*1E-6:
        sma = 1/alpha
        num = -2*mu*alpha*dt
        denom = dot(r0_vec, v0_vec) + sign(dt)*m.sqrt(-1*mu*sma)*(1-(r0_mag*alpha))
        chi_n = sign(dt) * m.sqrt(-1*sma)*m.log(num/denom) 
    err = 1

    while err > 1E-6:
        chi_n_sqr = chi_n**2
        psi = alpha * (chi_n_sqr)
        c2c3 = find_c2c3(psi)
        c2 = c2c3[0]
        c3 = c2c3[1]

        temp1 = chi_n_sqr * c2
        temp2 = (dot(r0_vec, v0_vec))/(sqrt_mu)
        temp3 = temp2 * chi_n
        temp4 = temp3 * (1 - (psi * c3))
        temp5 = r0_mag * (1 - (psi*c2))
        rad = temp1 + temp4 + temp5

        temp6 = dt * sqrt_mu
        temp7 = c3 * (chi_n**3)
        temp8 = (chi_n_sqr) * (c2) * ((dot(r0_vec, v0_vec))/(sqrt_mu))
        temp9 = r0_mag * chi_n * (1 - (psi * c3))
        temp10 = temp6 - temp7 - temp8 - temp9
        chi_np1 = chi_n + (temp10/rad)
        breakpoint()
        err = abs(chi_n - chi_np1)
        chi_n = chi_np1

    f = 1 - (((chi_n**2)/(r0_mag))*c2)
    df = ((sqrt_mu)/(rad*r0_mag)) * chi_n * ((psi * c3) - 1)

    g = dt - (((chi_n**3)/(m.sqrt(mu))) * c3)
    dg = 1 - (((chi_n**2)/(rad)) * c2)

    r_vec = f*r0_vec + g*v0_vec
    v_vec = df*r0_vec + dg*v0_vec
    breakpoint()
    return np.array([r_vec, v_vec])


# ALGORITHM 09
@jit(nopython=True)
def rv2coe(state, GM, tol_ecc=1E-6, tol_inc=1):
    """
    Convert the inertial state to Keplerian Classic Orbital Elements.

    This function implements Algorithm 9 found in the 4th edition of
    Fundamentals of Astrodynamics and Applications by David Vallado.

    This function finds the classical orbital elements given the geocentric
    equatorial position and velocity vectors

    Parameters
    ----------
    state : numpy array
    State is the 1x6 array that represents the inertial state of a spacecraft
    around a central grativational field.

    GM : float
    GM is the gravitational parameter of the central body.
    GM is an assumption that the mass of the spacecraft can be neglected
    GM approximates the following:
    mu = GM ~= G(M + m)

    tol : float
    tol is the tolerance for the determination of a circular orbit and an
    inclined versus equatorial orbit

    Returns
    -------
    coe : tuple
    """
    posVec = state[:3]
    posMag = norm(posVec)
    velVec = state[3:]
    velMag = norm(velVec)
    spcAngMomVec = cross(posVec, velVec)
    spcAngMom = norm(spcAngMomVec)
    nodeVec = cross(KVEC, spcAngMomVec)

    a = ((velMag**2) - (GM/posMag))*posVec
    b = dot(posVec, velVec)*velVec
    eccVec = (a - b)/GM
    ecc = norm(eccVec)

    spcMechEnergy = ((velMag**2)/2) - (GM/posMag)

    if ecc < 1:
        sma = (-1*GM)/(2*spcMechEnergy)
        semiparameter = sma*(1 - (ecc**2))
        meanMotion = m.sqrt(GM/(sma**3))
    elif ecc == 1:
        sma = np.inf
        semiparameter = (spcAngMom**2)/GM
        meanMotion = 2*m.sqrt(GM/(semiparameter**3))
    else:  # ecc > 1
        sma = -1E20

    denom = norm(KVEC) * spcAngMom
    cosine_inc = dot(KVEC, spcAngMomVec)/denom
    sine_inc = norm(cross(KVEC, spcAngMomVec))/denom
    inc = m.degrees(m.atan2(sine_inc, cosine_inc))
    if inc < 0 and inc > -360:
        inc += 360

    denom = norm(IVEC) * norm(nodeVec)
    cosine_raan = dot(IVEC, nodeVec)/denom
    sine_raan = norm(cross(IVEC, nodeVec)/denom)
    raan = m.degrees(m.atan2(sine_raan, cosine_raan))
    if nodeVec[1] < 0:
        raan = 360 - raan

    return


@njit
def rv2coeVerbose(state, GM, tol_ecc=1E-6, tol_inc=1):
    """
    Convert the inertial state to Keplerian Classic Orbital Elements.

    This function implements Algorithm 9 found in the 4th edition of
    Fundamentals of Astrodynamics and Applications by David Vallado.

    This function finds the classical orbital elements given the geocentric
    equatorial position and velocity vectors

    Parameters
    ----------
    state : numpy array
    State is the 1x6 array that represents the inertial state of a spacecraft
    around a central grativational field.

    GM : float
    GM is the gravitational parameter of the central body.
    GM is an assumption that the mass of the spacecraft can be neglected
    GM approximates the following:
    mu = GM ~= G(M + m)

    tol : float
    tol is the tolerance for the determination of a circular orbit and an
    inclined versus equatorial orbit

    Returns
    -------
    coe : tuple
    coe[0] is the 1x6 array of the Keplerian Classic Orbital Elements
    coe[1] is a dictionary of all the parameters output from the algorithm
    coe[2] is a dictionary of other orbital parameters that are calculated

    """
    posVec = state[:3]
    posMag = norm(posVec)
    velVec = state[3:]
    velMag = norm(velVec)
    spcAngMomVec = cross(posVec, velVec)
    spcAngMom = norm(spcAngMomVec)
    nodeVec = cross(KVEC, spcAngMomVec)

    a = ((velMag**2) - (GM/posMag))*posVec
    b = dot(posVec, velVec)*velVec
    eccVec = (a - b)/GM
    ecc = norm(eccVec)

    spcMechEnergy = ((velMag**2)/2) - (GM/posMag)

    if ecc < 1:
        sma = (-1*GM)/(2*spcMechEnergy)
        semiparameter = sma*(1 - (ecc**2))
        meanMotion = m.sqrt(GM/(sma**3))
    elif ecc == 1:
        sma = np.inf
        semiparameter = (spcAngMom**2)/GM
        meanMotion = 2*m.sqrt(GM/(semiparameter**3))
    else:  # ecc > 1
        sma = -1E20

    denom = norm(KVEC) * spcAngMom
    cosine_inc = dot(KVEC, spcAngMomVec)/denom
    sine_inc = norm(cross(KVEC, spcAngMomVec))/denom
    inc = m.degrees(m.atan2(sine_inc, cosine_inc))
    if inc < 0 and inc > -360:
        inc += 360

    denom = norm(IVEC) * norm(nodeVec)
    cosine_raan = dot(IVEC, nodeVec)/denom
    sine_raan = norm(cross(IVEC, nodeVec)/denom)
    raan = m.degrees(m.atan2(sine_raan, cosine_raan))
    if nodeVec[1] < 0:
        raan = 360 - raan

    inc_mod = inc % 90
    if ecc > tol_ecc and ecc < 1 and inc_mod > tol_inc:
        """ Ellipse and Inclined Orbit """
        # compute the arguement of periapsis
        denom = norm(nodeVec)*ecc
        cosine_aop = dot(nodeVec, eccVec)/denom
        sine_aop = norm(cross(nodeVec, eccVec))/denom
        aop = m.degrees(m.atan2(sine_aop, cosine_aop))
        if eccVec[2] < 0:
            aop = 360 - aop
        # compute true anomaly
        denom = ecc * posMag
        cosine_nu = dot(eccVec, posVec)/denom
        sine_nu = norm(cross(eccVec, posVec))/denom
        nu = m.degrees(m.atan2(sine_nu, cosine_nu))
        if dot(posVec, velVec) < 0:
            nu = 360 - nu
        # build COE state vector
        coe_vec = np.array([sma, ecc, inc, raan, aop, nu])
        # compute true longitude of periapsis
        denom = norm(IVEC) * ecc
        cosine_true_lop = dot(IVEC, eccVec)/denom
        sine_true_lop = norm(cross(IVEC, eccVec))/denom
        true_lop = m.degrees(m.atan2(sine_true_lop, cosine_true_lop))
        if eccVec[1] < 0:
            true_lop = 360 - true_lop
        # compute argument of latitude
        denom = norm(nodeVec) * posMag
        cosine_aol = dot(nodeVec, posVec)/denom
        sine_aol = norm(cross(nodeVec, posVec))/denom
        aol = m.degrees(m.atan2(sine_aol, cosine_aol))
        # compute mean argument of latitude
        mean_aol = -1E20
        # compute the longitude of periapsis
        lop = raan + aop
        # compute true longitude
        denom = norm(IVEC) * posMag
        cosine_true_long = dot(IVEC, posVec)/denom
        sine_true_long = norm(cross(IVEC, posVec))/denom
        true_long = m.degrees(m.atan2(sine_true_long, cosine_true_long))

    elif ecc > tol_ecc and ecc < 1 and inc_mod < tol_inc:
        """ Ellipse and Equatorial Orbit """
        # compute true longitude of periapsis
        denom = norm(IVEC) * ecc
        cosine_true_lop = dot(IVEC, eccVec)/denom
        sine_true_lop = norm(cross(IVEC, eccVec))/denom
        true_lop = m.degrees(m.atan2(sine_true_lop, cosine_true_lop))
        if eccVec[1] < 0:
            true_lop = 360 - true_lop

    elif ecc < tol_ecc and inc_mod > tol_inc:
        """ Circular Inclined Orbit """
        # compute argument of latitude
        denom = norm(eccVec) * posMag
        cosine_aol = dot(eccVec, posVec)/denom
        sine_aol = norm(cross(eccVec, posVec))/denom
        aol = m.degrees(m.atan2(sine_aol, cosine_aol))
        # compute mean anomaly
        # TODO: code and call algorithm 2 from page 103 of vallado
        # compute mean argument of latitude

    else:
        """ Circular and Equatorial Orbit """
        # compute true longitude
        denom = norm(IVEC) * posMag
        cosine_true_long = dot(IVEC, posVec)/denom
        sine_true_long = norm(cross(IVEC, posVec))/denom
        true_long = m.degrees(m.atan2(sine_true_long, cosine_true_long))

    coe_dict = Dict.empty(key_type=types.unicode_type,
                          value_type=FLOAT_ARRAY)
    coe_dict = {'semiparameter': semiparameter,
                'semimajor_axis': sma,
                'eccentricity': ecc,
                'inclination': inc,
                'right_ascension_of_the_ascending_node': raan,
                'arguement_of_perigee': aop,
                'true_anomaly': nu,
                'longitude_of_periapsis': lop,
                'true_longitude_of_periapsis': true_lop,
                'argument_of_latitude': aol,
                'mean_argument_of_latitude': mean_aol,
                'true_longitude': true_long,
                'mean_motion': meanMotion,
                'specific_angular_momentum_magnitude': spcAngMom}

    orbit_facts = Dict.empty(key_type=types.unicode_type,
                             value_type=FLOAT_ARRAY)
    """orbit_facts = {'position vector': posVec, 'velocity vector': velVec,
                  'specific angular momentum vector': spcAngMomVec,
                  'line of nodes': nodeVec,
                  'eccentricity vector': eccVec}"""
    return coe_vec, coe_dict, orbit_facts


# ALGORITHM 10
def coe_2_rv():
    return


# Algorithm 36: Hohmann Transfer
def hohmann(r1, r2, mu):
    """
    Compute a Hohmann transfer between two circular orbits

    Parameters:
    -----------
    r1 : float
        r1 is the radius of the circular departure orbit from the central
        body and r1 is the magnitude of the position vector and is also
        equivalent to the orbit's semimajor axis.

    r2 : float
        r2 is the radius of the circular arrival orbit from the central body
        r2 is the magnitude of the position vector and is also equivalent to
        the orbit's semimajor axis.

    mu : float
        mu is the gravitational parameter equivalent to GM of the central body

    Returns:
    --------
    sma_trans : float
        sma_trans is the semimajor axis of the transfer ellipse

    tau_trans : float
        tau_trans is the total transfer time in seconds. It is equivalent to
        half of the period of the transfer ellipse.
    """
    sma_trans = 0.5*(r1 + r2)
    v_int = m.sqrt(mu/r1)
    v_trans_a = m.sqrt(((2*mu)/r1) - (mu/sma_trans))
    v_trans_b = m.sqrt(((2*mu)/r2) - (mu/sma_trans))
    v_final = m.sqrt(mu/r2)
    delta_vel_1 = abs(v_trans_a - v_int)
    delta_vel_2 = abs(v_final - v_trans_b)
    delta_vel_tot = delta_vel_1 + delta_vel_2
    tau_trans = m.pi * m.sqrt((sma_trans**3)/mu)
    return [sma_trans, delta_vel_1, delta_vel_2, delta_vel_tot, tau_trans]


# Algorithm 37: Bi-elliptic Transfer
def bielliptic(r1, r2, r3, mu):
    """
    Compute a Bi-Elliptic Transfer between two circular orbits

    Parameters:
    -----------
    r1 : float
        r1 is the radius of the circular departure orbit from the central
        body r1 is the magnitude of the position vector and is also equivalent
        to the orbit's semimajor axis

    r2 : float
        r2 is the radius of the coupling point for the two Hohmann Transfers
        and the location where the second maneuver will occur

    r3 : float
        r3 is the radius of the circular arrival orbit from the central body
        r3 is the magnitude of the position vector and is also equivalent to
        the orbit's semimajor axis

    Returns:
    -------
    sma_trans_1 : float
        sma_trans_1 is the semimajor axis of the first Hohmann transfer

    sma_trans_2 : float
        sma_trans_2 is the semimajor axis of the second Hohmann transfer

    dV_1 : float
        dV_1 is the magnitude of the velocity change for the first maneuver

    dV_2 : float
        dV_2 is the magnitude of the velocity change for the second maneuver

    dV_3 : float
        dV_3 is the magnitude of the velocity change for the third maneuver

    dV_tot : float
        dV_tot is the magnitude of the total velocity change for all maneuvers

    tau_trans
        tau_trans is the total transfer time of the transfer in seconds
    """
    a_trans_1 = 0.5*(r1 + r2)
    a_trans_2 = 0.5*(r2 + r3)
    v_int = m.sqrt(mu/r1)
    v_trans_1_a = m.sqrt(((2*mu)/r1) - (mu/a_trans_1))
    v_trans_1_b = m.sqrt(((2*mu)/r2) - (mu/a_trans_1))
    v_trans_2_b = m.sqrt(((2*mu)/r2) - (mu/a_trans_2))
    v_trans_2_c = m.sqrt(((2*mu)/r3) - (mu/a_trans_2))
    v_final = m.sqrt(mu/r3)
    delta_vel_1 = abs(v_trans_1_a - v_int)
    delta_vel_2 = abs(v_trans_2_b - v_trans_1_b)
    delta_vel_3 = abs(v_final - v_trans_2_c)
    delta_vel_tot = delta_vel_1 + delta_vel_2 + delta_vel_3
    tau_trans = m.pi*(m.sqrt((a_trans_1**3)/mu) + m.sqrt((a_trans_2**3)/mu))
    return [a_trans_1, a_trans_2, delta_vel_1, delta_vel_2, delta_vel_3,
            delta_vel_tot, tau_trans]


# Algorithm 36
def hohmann(rad1, rad2, mu):
    """
    Computes a Hohmann Transfer.
    
    Parameters
    ----------
    rad1 : float
        rad1 is the radius of the inital circular orbit
    rad2 : float
        rad2 is the radius of the final circular orbit
    mu : float
        mu is the gravitational parameter of the central body
    
    Returns
    -------
    sma_trans : float
        sma_trans is a semimajor axis of the transfer orbit
    dv1 : float
        dv1 is the magnitude of the first velocity change
    dv2 : float
        dv2 is the magnitude of the second velocity change 
    dv_tot : float
        dv_tot is the magnitude of the total velocity change
    tau_trans : float
        tau_trans is the time of the transfer
    """
    sma_trans = 0.5*(rad1 + rad2)
    vi = m.sqrt(mu/rad1)
    v1 = m.sqrt(((2*mu)/rad1) - (mu/sma_trans))
    v2 = m.sqrt(((2*mu)/rad2) - (mu/sma_trans))
    vf = m.sqrt(mu/rad2)
    dv1 = abs(v1 - vi)
    dv2 = abs(vf - v2)
    dv_tot = dv1 +  dv2
    tau_trans = m.pi * m.sqrt((sma_trans**3)/mu)
    return (sma_trans, dv1, dv2, dv_tot, tau_trans)


# Algorithm 37
def bi_elliptic(rad_i, rad_b, rad_f, mu):
    """
    Computes a Bi-Elliptic Transfer.

    Paramters
    ---------
    rad_i : float
        rad_i is the radius of the inital circular orbit
    rad_b : float
        rad_b is the radius of the intermediate orbit
    rad_f : float
        rad_f is the radius of the final circular orbit
    mu : float
        mu is the gravitational parameter of the central body

    Returns
    -------
    sma_trans_1 : float
        sma_trans_1 is the semimajor axis of the 1st part of the transfer
    sma_trans_2 : float
        sma_trans_2 is the semimajor axis of the 2nd part of the transfer
    dv1 : float
        dv1 is the magnitude of the first velocity change
    dv2 : float
        dv2 is the magnitude of the second velocity change
    dv3 : float
        dv3 is the magnitude of the third velocity change
    dv_tot : float
        dv_tot is the magnitude of the total velocity change
    tau_trans
        tau_trans is the total transfer time
    """
    sma_trans_1 = 0.5*(rad_i + rad_b)
    sma_trans_2 = 0.5*(rad_b + rad_f)
    vi = m.sqrt(mu/rad_i)
    v_trans_1_a = m.sqrt(((2*mu)/rad_i) - (mu/sma_trans_1))
    v_trans_1_b = m.sqrt(((2*mu)/rad_b) - (mu/sma_trans_1))
    v_trans_2_b = m.sqrt(((2*mu)/rad_b) - (mu/sma_trans_2))
    v_trans_2_c = m.sqrt(((2*mu)/rad_f) - (mu/sma_trans_2))
    vf = m.sqrt(mu/rad_f)
    dv1 = abs(v_trans_1_a - vi)
    dv2 = abs(v_trans_2_b - v_trans_1_b)
    dv3 = abs(vf - v_trans_2_c)
    dv_tot = dv1 + dv2 + dv3
    tau_trans = m.pi*m.sqrt((sma_trans_1**3)/mu) + m.pi*m.sqrt((sma_trans_2**3)/mu)
    return (sma_trans_1, sma_trans_2, dv1, dv2, dv3, dv_tot, tau_trans) 


# Algorithm 38
def one_tangent(rad_i, rad_f, nu_trans, mu, periapsis=True):
    """
    Computes a One-Tangent Burn.

    Parameters
    ----------
    rad_i : float
        radius of the initial orbit
    rad_f : float
        radius of the final orbit
    nu_trans : float
        true anomaly angle at final orbit insertion 
    mu : float
        Gravitational pramater of the central body
    periapsis : Bool
        Signal a departur at either periapsis or apoapsis

    Returns
    -------
    sma_trans : float
        semimajor axis of the transfer orbit
    dv1 : float
        magnitude of the first velocity change
    dv2 : float
        magnitude of the second velocity change
    dv_tot : float
        magnitude of the total velocity change
    """
    R = rad_i/rad_f
    if periapsis:
        ecc_trans = (R - 1)/(m.cos(nu_trans) - R)
        sma_trans = rad_i/(1 - ecc_trans)
    else:
        ecc_trans = (R - 1)/(m.cos(nu_trans) + R)
        sma_trans = rad_i/(1 + ecc_trans)
    v_i = m.sqrt(mu/rad_i)
    v_trans_a = m.sqrt(((2*mu)/rad_i) - (mu/sma_trans))
    v_trans_b = m.sqrt(((2*mu)/rad_f) - (mu/sma_trans))
    v_f = m.sqrt(mu/rad_f)
    dv1 = abs(v_trans_a - v_i)
    tan_fpa = (ecc_trans*m.sin(nu_trans))/(1 + (ecc_trans*m.cos(nu_trans)))
    fpa = m.atan(tan_fpa)
    dv2 = m.sqrt((v_trans_b**2) + (v_f**2) - (2*v_trans_b*v_f*m.cos(fpa)))
    dv_tot = dv1 + dv2
    cos_E = (ecc_trans + m.cos(nu_trans))/(1 + ecc_trans*m.cos(nu_trans))
    # TODO Add Transfer Time
    return (sma_trans, dv1, dv2, dv_tot)


# Algorithm 39
def inclination_only(delta_inc, gamma, vel):
    """
    Computes a change Inclination Only

    Parameters
    ----------
    delta_inc : float
        the required change in inclination
    gamma : float
        the flight path angle
    vel : float
        the velocity at the burn location - typically at a node

    Returns
    -------
    dv : float
        the magnitude of the required velocity change
    """
    return 2*vel*m.cos(m.radians(gamma))*m.sin((m.radians(delta_inc/2)))


# Algorithm 40
def raan_only(delta_raan, inc, vel):
    """
    Change in the Ascending Node -- Circular

    Parameters
    ----------
    delta_raan : float
        the required change in the right ascension of the ascending node
    inc : float
        the inclination of the orbit
    vel : float
        the velocity at the burn location

    Returns
    -------
    dv : float
        the magnitude of the required velocity change    
    """
    delta_raan = m.radians(delta_raan)
    inc = m.radians(inc)
    cosine_ang = ((m.cos(inc))**2) + (((m.sin(inc))**2)*m.cos(delta_raan))
    ang = m.acos(cosine_ang)
    return 2*vel*(m.sin(ang/2))


# Algorithm 41
def inc_and_raan(inc_i, inc_f, delta_raan, vel_0):
    """
    Change in both the Ascending Node and Inclination -- Circular
    
    Parameters
    ----------
    inc_i : float
        initial orbit inclination 
    inf_f : float
        final orbit inclination
    delta_raan : float
        the required change in the right ascension of the ascending node
    vel_0 : float
        the velocity at the burn location

    Returns
    -------
    dv : float
        the magnitude of the required velocity change
    """
    inc_i = m.radians(inc_i)
    inc_f = m.radians(inc_f)
    aa = m.cos(inc_i)*m.cos(inc_f)
    bb = m.sin(inc_i)*m.sin(inc_f)*m.cos(delta_raan)
    cosine_ang = aa + bb
    ang = m.acos(cosine_ang)
    return 2*vel_0*m.sin(ang/2)

# Algorithm 48
def hill_eqn(state, w, t):
    """
    Hill Reletative Equations of Motion for Two Bodies in Circular Orbits
    
    Parameters
    ----------
    state : numpy array of floats
        the initial relative state vector before propagation
    w : float
        the angular velocity of the system
    t : float
        the time of the propagation

    Returns
    -------
    vec : numpy array of floats
        the relative state vector after the propagation
    """
    rx0 = state[0]
    ry0 = state[1]
    rz0 = state[2]
    vx0 = state[3]
    vy0 = state[4]
    vz0 = state[5]

    x = (vx0/w)*m.sin(w*t) - (((3*rx0) + ((2*vy0)/w))*m.cos(w*t)) + ((4*rx0) + ((2*vy0)/w))
    y = ((6*rx0 + (4*vy0/w))*m.sin(w*t)) + ((2*vx0/w)*m.cos(w*t))
    y = y - ((6*w*rx0 + 3*vy0)*t) + (ry0 - (2*vx0/w)) 
    z = rz0*m.cos(w*t) + ((vz0/w)*m.sin(w*t))

    dx = (vx0*m.cos(w*t)) + (((3*w*rx0) + (2*vy0))*m.sin(w*t))
    dy = (((6*w*rx0) + (4*vy0))*m.cos(w*t)) - ((2*rx0)*m.sin(w*t)) - ((6*w*rx0) + (3*vy0))
    dz = (-1*rz0*w*m.sin(w*t)) + (vz0*m.cos(w*t))
    return np.array([x, y, z, dx, dy, dz])
