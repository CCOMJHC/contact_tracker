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



class Contact:
    """
    Class to create contact object with its own KalmanFilter.
    """

    def __init__(self, detect_info, kf, variance, timestamp, contact_id):
        """
        Define the constructor.
        
        detect_info -- dictionary containing data from the Detect message being used to create this Contact
        kf -- unique KalmanFilter object for this specific Contact object
        variance -- variance for the measurement
        timestamp -- indication of when this Contact was last accessed
        contact_id -- unique id for this Contact object for easy lookup in KalmanTracker's all_contacts dictionary
        xs -- list of tuples containing values for the predictions
        zs -- list of tuples containing values for the measurements
        """

        self.info = detect_info
        self.kf = kf
        self.variance = variance 
        self.first_accessed = timestamp
        self.last_accessed = timestamp
        self.id = contact_id
        self.xs = []
        self.zs = []
        self.times = []
        self.dt = 1 


    def init_kf_with_position_only(self):
        """
        Initialize a first-order KalmanFilter given only values for position.
        """

        # Define the state variable vector
        self.kf.x = np.array([self.info['x_pos'], self.info['y_pos'], 0, 0]).T

        # Define the state covariance matrix
        self.kf.P = np.array([[self.info['pos_covar'][0], 0, 0, 0],
                              [0, self.info['pos_covar'][7], 0 , 0],
                              [0, 0, 0, 0],
                              [0, 0, 0, 0]])

        # Define the noise covariance (TBD)
        self.kf.Q = Q_discrete_white_noise(dim=4, dt=self.dt, var=self.variance) 

        # Define the process model matrix
        self.kf.F = np.array([[1, 0, self.dt, 0],
                              [0, 1, 0, self.dt],
                              [0, 0, 1, 0],
                              [0, 0, 0, 1]])

        # Define the measurement function
        self.kf.H = np.array([[1, 0, 0, 0],
                              [0, 1, 0, 0]])

        # Define the measurement covariance
        self.kf.R = np.array([[self.variance, 0],
                              [0, self.variance]])


    def init_kf_with_velocity_only(self):
        """
        Initialize a first-oder KalmanFilter given only values for velocity.
        """

        # Define the state variable vector
        self.kf.x = np.array([0, 0, self.info['x_vel'], self.info['y_vel']]).T

        # Define the state covariance matrix
        self.kf.P = np.array([[0, 0, 0, 0],
                              [0, 0, 0 , 0],
                              [0, 0, self.info['twist_covar'][0], 0],
                              [0, 0, 0, self.info['twist_covar'][7]]])

        # Define the noise covariance (TBD)
        self.kf.Q = Q_discrete_white_noise(dim=4, dt=self.dt, var=self.variance) 

        # Define the process model matrix 
        self.kf.F = np.array([[1, 0, self.dt, 0],
                              [0, 1, 0, self.dt],
                              [0, 0, 1, 0],
                              [0, 0, 0, 1]])

        # Define the measurement function
        self.kf.H = np.array([[0, 0, 1, 0],
                              [0, 0, 0, 1]])

        # Define the measurement covariance
        self.kf.R = np.array([[self.variance, 0],
                              [0, self.variance]])


    def init_kf_with_position_and_velocity(self):
        """
        Initialize a first-order KalmanFilter given values for position and velocity.
        """

        # Define the state variable vector
        self.kf.x = np.array([self.info['x_pos'], self.info['y_pos'], self.info['x_vel'], self.info['y_vel']]).T

        # Define the state covariance matrix
        self.kf.P = np.array([[self.info['pos_covar'][0], 0, 0, 0],
                              [0, self.info['pos_covar'][7], 0 , 0],
                              [0, 0, self.info['twist_covar'][0], 0],
                              [0, 0, 0, self.info['twist_covar'][7]]])

        # Define the noise covariance (TBD)
        self.kf.Q = Q_discrete_white_noise(dim=4, dt=self.dt, var=self.variance) 

        # Define the process model matrix
        self.kf.F = np.array([[1, 0, self.dt, 0],
                              [0, 1, 0, self.dt],
                              [0, 0, 1, 0],
                              [0, 0, 0, 1]])

        # Define the measurement function
        self.kf.H = np.array([[1, 0, 0, 0],
                              [0, 1, 0, 0],
                              [0, 0, 1, 0],
                              [0, 0, 0, 1]])

        # Define the measurement covariance
        self.kf.R = np.array([[self.variance, 0, 0, 0],
                              [0, self.variance, 0, 0],
                              [0, 0, self.variance, 0],
                              [0, 0, 0, self.variance]])


