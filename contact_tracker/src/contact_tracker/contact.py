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

V = 50 
class Contact:
    """
    Class to create contact object with its own KalmanFilter.
    """

    def __init__(self, detect_info, kf, timestamp, contact_id):
        """
        Define the constructor.
        
        detect_info -- dictionary containing data from the Detect message being used to create this Contact
        kf -- unique KalmanFilter object for this specific Contact object
        timestamp -- indication of when this Contact was last accessed
        contact_id -- unique id for this Contact object for easy lookup in KalmanTracker's all_contacts dictionary
        xs -- list of tuples containing values for the predictions
        zs -- list of tuples containing values for the measurements
        """

        self.info = detect_info
        self.kf = kf
        self.first_accessed = timestamp
        self.last_accessed = timestamp
        self.id = contact_id
        self.xs = []
        self.zs = []
        self.times = []
        self.dt = 1 


    def init_kf_with_position_only(self):
        """
        Initialize the kalman filter with only position values for this contact.
        
        Keyword arguments:
        dt -- time step for the KalmanFilter
        """

        # Define the state variable vector
        self.kf.x = np.array([self.info['x_pos'], self.info['y_pos'], 0, 0]).T

        # Define the state covariance matrix
        self.kf.P = np.array([[self.info['pos_covar'][0], 0, 0, 0],
                              [0, self.info['pos_covar'][7], 0 , 0],
                              [0, 0, self.info['twist_covar'][0], 0],
                              [0, 0, 0, self.info['twist_covar'][7]]])

        # Define the noise covariance (TBD)
        self.kf.Q = Q_discrete_white_noise(dim=4, dt=self.dt, var=0.04**2) 

        # Define the process model matrix
        self.kf.F = np.array([[1, 0, self.dt, 0],
                              [0, 1, 0, self.dt],
                              [0, 0, 1, 0],
                              [0, 0, 0, 1]])

        # Define the measurement function
        self.kf.H = np.array([[1, 0, 0, 0],
                              [0, 1, 0, 0]])

        # Define the measurement covariance
        # Initially we estimate the value of V, i.e., the variance in our measurement I think?
        self.kf.R = np.array([[V, 0],
                              [0, V]])


    def init_kf_with_velocity_only(self):
        """
        Initialize the kalman filter with only velocity values for this contact.
        
        Keyword arguments:
        dt -- time step for the KalmanFilter
        """

        # Define the state variable vector
        self.kf.x = np.array([0, 0, self.info['x_vel'], self.info['y_vel']]).T

        # Define the state covariance matrix
        self.kf.P = np.array([[self.info['pos_covar'][0], 0, 0, 0],
                              [0, self.info['pos_covar'][7], 0 , 0],
                              [0, 0, self.info['twist_covar'][0], 0],
                              [0, 0, 0, self.info['twist_covar'][7]]])

        # Define the noise covariance (TBD)
        self.kf.Q = Q_discrete_white_noise(dim=4, dt=self.dt, var=0.04**2) 

        # Define the process model matrix
        self.kf.F = np.array([[1, 0, self.dt, 0],
                              [0, 1, 0, self.dt],
                              [0, 0, 1, 0],
                              [0, 0, 0, 1]])

        # Define the measurement function
        self.kf.H = np.array([[0, 0, 1, 0],
                              [0, 0, 0, 1]])

        # Define the measurement covariance
        # Initially we estimate the value of V, i.e., the variance in our measurement I think?
        self.kf.R = np.array([[V, 0],
                              [0, V]])



