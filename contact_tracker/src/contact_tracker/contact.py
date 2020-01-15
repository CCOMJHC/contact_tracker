#!/usr/bin/env python
# Class to create a Contact object and initialize its
# associated Kalman filters.

# Author: Rachel White
# University of New Hampshire
# Last Modified: 01/08/2020


import rospy
import numpy as np

from filterpy.kalman import KalmanFilter
from filterpy.kalman import update
from filterpy.kalman import predict
from filterpy.common import Q_discrete_white_noise 

dt = 1
V = 0.1
class Contact:
    """
    Class to create contact object with its own KalmanFilter.
    """

    def __init__(self, detect_info, kf, timestamp, contact_id):
        """
        Define the constructor.
        """

        self.info = detect_info
        self.kf = kf
        self.last_accessed = timestamp
        self.id = contact_id
        self.xs = []
        self.zs = []


    def init_kf(self, dt):
        """
        Initialize the kalman filter for this contact without values for velocity.
        """

        # Define the state variable vector
        self.kf.x = np.array([self.info['x_pos'], self.info['y_pos']]).T

        # Define the state covariance matrix
        self.kf.P = np.array([[self.info['pos_covar'][0], 0],
                              [0, self.info['pos_covar'][7]]])

        # Define the noise covariance (TBD)
        self.kf.Q = Q_discrete_white_noise(dim=2, dt=dt, var=0.04**2)

        # Define the process model matrix
        self.kf.F = np.array([[1, dt],
                              [0, 1]])

        # Define the measurement function
        self.kf.H = np.array([1, 0],
                             [0, 1])

        # Define the measurement covariance
        self.kf.R = np.array([V, 0],
                             [0, V])


    def init_kf_with_velocity(self, dt):
        """
        Initialize the kalman filter including velocity values for this contact.
        """

        # Define the state variable vector
        self.kf.x = np.array([self.info['x_pos'], self.info['y_pos'], self.info['x_vel'], self.info['y_vel']]).T

        # Define the state covariance matrix
        self.kf.P = np.array([[self.info['pos_covar'][0], 0, 0, 0],
                             [0, self.info['pos_covar'][7], 0 , 0],
                             [0, 0, self.info['twist_covar'][0], 0],
                             [0, 0, 0, self.info['twist_covar'][7]]])

        # Define the noise covariance (TBD)
        self.kf.Q = Q_discrete_white_noise(dim=4, dt=dt, var=0.04**2) 

        # Define the process model matrix
        self.kf.F = np.array([[1, 0, dt, 0],
                              [0, 1, 0, dt],
                              [0, 0, 1, 0],
                              [0, 0, 0, 1]])

        # Define the measurement function
        self.kf.H = np.array([[1, 0, 0, 0],
                              [0, 1, 0, 0]])

        # Define the measurement covariance
        # Initially we estimate the value of V, i.e., the variance in our measurement I think?
        self.kf.R = np.array([[V, 0],
                              [0, V]])
