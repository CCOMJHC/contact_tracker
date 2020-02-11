#!/usr/bin/env python
# Class to create a Contact object and initialize its
# associated Kalman filters.

# Author: Rachel White
# University of New Hampshire
# Last Modified: 02/04/2020


import rospy
import numpy as np
import math

from filterpy.kalman import KalmanFilter
from filterpy.kalman import update
from filterpy.kalman import predict
from filterpy.kalman import IMMEstimator
from filterpy.common import Q_discrete_white_noise



class Contact:
    """
    Class to create contact object with its own Kalman filter bank.
    """

    def __init__(self, detect_info, all_filters, variance, timestamp):
        """
        Define the constructor.
        
        detect_info -- dictionary containing data from the Detect message being used to create this Contact
        all_filters -- list containing unique KalmanFilter objects for this specific Contact object
        variance -- variance for the measurement
        first_measured -- indication of when this Contact was first measured
        last_measured -- indication of when this Contact was last measured 
        contact_id -- unique id for this Contact object for easy lookup in KalmanTracker's all_contacts dictionary
        xs -- list of tuples containing values for the predictions
        zs -- list of tuples containing values for the measurements
        times -- list representing times that correspond to when each measurement was taken
        dt -- initial timestep
        """

        self.info = detect_info
        self.mu = np.array([0.3, 0.7])
        self.M = np.array([[0.97, 0.03],
                           [0.05, 0.95]])
        self.all_filters = all_filters # This is purely for initializing the filters
        self.filter_bank = None
        self.variance = variance 
        self.first_measured = timestamp
        self.last_measured = timestamp
        self.id = timestamp 
        self.xs = []
        self.zs = []
        self.ps = []
        self.times = []
        self.dt = 1 
        self.last_xpos = 0
        self.last_ypos = 0
        self.last_xvel = 0
        self.last_yvel = 0


    def init_kf_with_position_only(self):
        """
        Initialize each filter in this Contact's filter bank given only values for position.
        """

        for i in range(0, len(self.filter_bank.filters)):
            # Define the state variable vector
            self.filter_bank.filters[i].x = np.array([self.info['x_pos'], self.info['y_pos'], 0, 0, 0, 0]).T
    
            # Define the state covariance matrix
            self.filter_bank.filters[i].P = np.array([[self.info['pos_covar'][0], 0, 0, 0, 0, 0],
                             [0, self.info['pos_covar'][7], 0 , 0, 0, 0],
                             [0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0]])
    
            # Define the noise covariance (TBD)
            self.filter_bank.filters[i].Q = Q_discrete_white_noise(dim=2, dt=self.dt, var=self.variance, block_size=3, order_by_dim=False)

            # Define the process model matrix
            self.filter_bank.filters[i].F = np.array([[1, 0, self.dt, 0, 0, 0],
                             [0, 1, 0, self.dt, 0, 0],
                             [0, 0, 1, 0, 0, 0],
                             [0, 0, 0, 1, 0, 0],
                             [0, 0, 0, 0, 1, 0],
                             [0, 0, 0, 0, 0, 1]])
    
            # Define the measurement function
            self.filter_bank.filters[i].H = np.array([[1, 0, 0, 0, 0, 0],
                             [0, 1, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0]])

            # Define the measurement covariance
            self.filter_bank.filters[i].R = np.array([[self.variance, 0, 0, 0, 0, 0],
                             [0, self.variance, 0, 0, 0, 0],
                             [0, 0, self.variance, 0, 0, 0],
                             [0, 0, 0, self.variance, 0, 0],
                             [0, 0, 0, 0, self.variance, 0],
                             [0, 0, 0, 0, 0, self.variance]])



    def init_kf_with_velocity_only(self):
        """
        Initialize a first-oder KalmanFilter given only values for velocity.
        """

        for i in range(0, len(self.filter_bank.filters)):
            # Define the state variable vector
            '''self.filter_bank.filters[i].x = np.array([0, 0, self.info['x_vel'], self.info['y_vel']]).T

            # Define the state covariance matrix
            self.filter_bank.filters[i].P = np.array([[0, 0, 0, 0],
                              [0, 0, 0, 0],
                              [0, 0, self.info['twist_covar'][0], 0],
                              [0, 0, 0, self.info['twist_covar'][7]]])

            # Define the noise covariance (TBD)
            self.filter_bank.filters[i].Q = Q_discrete_white_noise(dim=4, dt=self.dt, var=self.variance) 

            # Define the process model matrix 
            self.filter_bank.filters[i].F = np.array([[1, 0, self.dt, 0],
                              [0, 1, 0, self.dt],
                              [0, 0, 1, 0],
                              [0, 0, 0, 1]])

            # Define the measurement function
            self.filter_bank.filters[i].H = np.array([[0, 0, 0, 0],
                              [0, 0, 0, 0],
                              [0, 0, 1, 0],
                              [0, 0, 0, 1]])

            # Define the measurement covariance
            self.filter_bank.filters[i].R = np.array([[self.variance, 0, 0, 0],
                              [0, self.variance, 0, 0],
                              [0, 0, self.variance, 0],
                              [0, 0, 0, self.variance]])'''
            self.filter_bank.filters[i].x = np.array([[self.info['x_vel'], self.info['y_vel']], [0, 0]]).T

            # Define the state covariance matrix
            self.filter_bank.filters[i].P = np.array([
                              [self.info['twist_covar'][0], 0],
                              [0, self.info['twist_covar'][7]]])

            # Define the noise covariance (TBD)
            self.filter_bank.filters[i].Q = Q_discrete_white_noise(dim=4, dt=self.dt, var=self.variance) 

            # Define the process model matrix 
            self.filter_bank.filters[i].F = np.array([[1, 0],
                              [0, self.dt]])

            # Define the measurement function
            self.filter_bank.filters[i].H = np.array([
                              [1, 0],
                              [0, 1]])

            # Define the measurement covariance
            self.filter_bank.filters[i].R = np.array([[self.variance, 0],
                              [0, self.variance]])




    def init_kf_with_position_and_velocity(self):
        """
        Initialize a first-order KalmanFilter given values for position and velocity.
        """
        
        for kf in self.all_filters:
            # Define the state variable vector
            kf.x = np.array([self.info['x_pos'], self.info['y_pos'], self.info['x_vel'], self.info['y_vel'], 0, 0]).T

            # Define the state covariance matrix
            kf.P = np.array([[self.info['pos_covar'][0], 0, 0, 0, 0, 0],
                             [0, self.info['pos_covar'][7], 0, 0, 0, 0],
                             [0, 0, self.info['twist_covar'][0], 0, 0, 0],
                             [0, 0, 0, self.info['twist_covar'][7], 0, 0],
                             [0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0]])

            # Define the noise covariance (TBD)
            kf.Q = Q_discrete_white_noise(dim=2, dt=self.dt, var=self.variance, block_size=3, order_by_dim=False) 

            # Define the process model matrix
            kf.F = np.array([[1, 0, self.dt, 0, 0, 0],
                              [0, 1, 0, self.dt, 0, 0],
                              [0, 0, 1, 0, 0, 0],
                              [0, 0, 0, 1, 0, 0],
                              [0, 0, 0, 0, 1, 0],
                              [0, 0, 0, 0, 0, 1]])

            # Define the measurement function
            kf.H = np.array([[1, 0, 0, 0, 0, 0],
                             [0, 1, 0, 0, 0, 0],
                             [0, 0, 1, 0, 0, 0],
                             [0, 0, 0, 1, 0, 0],
                             [0, 0, 0, 0, 1, 0],
                             [0, 0, 0, 0, 0, 1]])

            # Define the measurement covariance
            kf.R = np.array([[self.variance, 0, 0, 0, 0, 0],
                             [0, self.variance, 0, 0, 0, 0],
                             [0, 0, self.variance, 0, 0, 0],
                             [0, 0, 0, self.variance, 0, 0],
                             [0, 0, 0, 0, self.variance, 0],
                             [0, 0, 0, 0, 0, self.variance]])


    def recompute_q(self, epoch):
        """
        Recompute the values of Q for the Kalman filters in this Contact.

        Keyword arguments:
        epoch -- time since this contact was last incorporated into the filter
        """
        
        for i in range(0, len(self.filter_bank.filters)):
            self.filter_bank.filters[i].Q = Q_discrete_white_noise(dim=2, dt=epoch*self.dt, var=self.variance, block_size=3, order_by_dim=False)


    def get_z(self, detect_info):
        """
        Get the measurement vector based on information sent in the detect message.

        Keyword arguments:
        detect_info -- the dictionary containing the detect info being checked

        Returns:
        Z -- the measurement vector for the detect message being examined
        """
        
        Z = np.zeros(self.filter_bank.x.shape)
        
        if math.isnan(detect_info['x_pos']):
            Z = [self.last_xpos, self.last_ypos, self.info['x_vel'], self.info['y_vel'], 0, 0]
        elif math.isnan(detect_info['x_vel']):
            Z = [self.info['x_pos'], self.info['y_pos'], self.last_xvel, self.last_yvel, 0, 0]
        else:
            Z = [self.info['x_pos'], self.info['y_pos'], self.info['x_vel'], self.info['y_vel'], 0, 0]
        
        return Z 
