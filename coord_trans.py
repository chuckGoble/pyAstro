# -*- coding: utf-8 -*-
""" This python module is a custom module that has functions to perform coordinate
transformations.

Function List
-------------
rot_01: Single axis rotation about the 'X' axis

rot_02: Single axis rotation about the 'Y' axis

rot_03: Single axis rotation about the 'Z' axis

ecef2enu: Rotation from Earth-Centered Earth-Fixed to East-North-Up
"""


import math as m
import numpy as np


# COORDINATE TRANSFORMATIONS
# ************************************************************************** #
def rot_01(ang, deg=False):
    """Function performs a single rotation about the 'X' body fixed axis.

    Parameters
    ----------
    ang: float
        Angle of rotation measured in radians

    Returns
    -------
    rot_mat: 3x3 numpy array
        Rotation matrix
    """

    # Check if angle is in degrees and change to radians if true
    if deg:
        ang = m.radians(ang)

    # Build the rotation matrix
    rot_mat = np.matrix([[1, 0, 0],
                        [0, m.cos(ang), m.sin(ang)],
                        [0, -1*m.sin(ang), m.cos(ang)]])
    return rot_mat


def rot_02(ang, deg=False):
    """Function performs a single rotation about the 'Y' body fixed axis.

    Parameters
    ----------
    ang: float
        Angle of rotation measured in radians

    Returns
    -------
    rot_mat: 3x3 numpy array
        Rotation matrix
    """

    # Check if angle is in degrees and change to radians if true
    if deg:
        ang = m.radians(ang)

    # Build the rotation matrix
    rot_mat = np.matrix([[m.cos(ang), 0, -1*m.sin(ang)],
                        [0, 1, 0],
                        [m.sin(ang), 0, m.cos(ang)]])
    return rot_mat


def rot_03(ang, deg=False):
    """Function performs a single rotation about the 'Z' body fixed axis.

    Parameters
    ----------
    ang: float
        Angle of rotation measured in radians

    Returns
    -------
    rot_mat: 3x3 numpy array
        Rotation matrix
    """

    # Check if angle is in degrees and change to radians if true
    if deg:
        ang = m.radians(ang)

    # Build the rotation matrix
    rot_mat = np.matrix([[m.cos(ang), m.sin(ang), 0],
                        [-1*m.sin(ang), m.cos(ang), 0],
                        [0, 0, 1]])
    return rot_mat


def ecef2enu(lat, long, ecef_vec, lat_deg=True, long_deg=True):
    """Implemention of appendix 4.A.2 ECEF to ENU from GPS textbook found on page 137.

    Parameters
    ----------
    lat: float
        Geodetic latitude as the angle measured in the meridian plane through the
        point between the equatorial plane of the ellipsoid and the line perpendicular
        to the surface of the ellipsoid. Angle is measured positive north of the equator.

    long: float
        Geodetic longitude as the angle measured in the equatorial plane between the
        reference meridian and the meridian plane through the point on the ellipsoid.
        The angle is measured positive to the east from the zero meridian.

    ecef_vec: 1x3 numpy array
        3 element vector of the position on the reference ellipsoid as expressed in the
        Earth-Centered Earth-Fixed Coordinate System
    Returns
    -------
    enu_vec: 1x3 numpy array
        3 element vector of the position on the reference ellipsoid as expressed in the local
        body fixed rotating East-North-Up Coordinate System.

    """

    # Check if input angles are in degrees or radians
    if lat_deg:
        ang1 = m.radians(90 - lat)
    else:
        ang1 = m.pi - lat
    if long_deg:
        ang2 = m.radians(long + 90)
    else:
        ang2 = long + m.pi

    # Calculate the Rotation Matrix Q for ECEF2ENU
    q_rot_mat = rot_01(ang1) * rot_02(ang2)

    # Perform the coordinate transformation via rotation
    enu_vec = q_rot_mat * ecef_vec

    return enu_vec
# ************************************************************************** #
