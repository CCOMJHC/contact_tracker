#!/usr/bin/env python
# Class to create a Contact object and initialize its
# associated Kalman filters.

# Author: Rachel White
# University of New Hampshire
# Last Modified: 02/20/2020


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

    def __init__(self, detect_info, all_filters, timestamp):
        """
        Define the constructor.
        
        detect_info -- dictionary containing data from the Detect message being used to create this Contact
        all_filters -- list containing unique KalmanFilter objects for this specific Contact object
        timestamp -- header from the detect message 
        """

        # Variables for the IMM Estimator
        self.mu = np.array([0.3, 0.7])
        self.M = np.array([[0.97, 0.03],
                           [0.05, 0.95]])
        # all_filters is purely for initializing the filters
        self.all_filters = all_filters 
        self.filter_bank = None
       
        # Variables that keep track of time
        self.dt = 1.0 
        self.first_measured = timestamp
        self.last_measured = timestamp
        self.last_xpos = .0
        self.last_ypos = .0
        self.last_xvel = .0
        self.last_yvel = .0

        # Variables that keep track of data for plotting purposes
        self.xs = []
        self.zs = []
        self.ps = []
        self.times = []
        
        # Other important variables
        self.info = detect_info
        self.id = timestamp 
        

    def init_filters(self):
        """
        Initialize each filter in this Contact's filter bank.
        """
        
        for i in range(0, len(self.all_filters)):
            if not math.isnan(self.info['x_pos']) and math.isnan(self.info['x_vel']):
                rospy.loginfo('Instantiating Kalman filters with position but without velocity')
                self.all_filters[i].x = np.array([self.info['x_pos'], self.info['y_pos'], .0, .0, .0, .0]).T
                self.all_filters[i].F = np.array([
                    [.1, .0, self.dt, .0, ((.1/.2)*self.dt)**2, .0],
                    [.0, .1, .0, self.dt, .0, ((.1/.2)*self.dt)**2],
                    [.0, .0, .1, .0, self.dt, .0],
                    [.0, .0, .0, .1, .0, self.dt],
                    [.0, .0, .0, .0, .0, .0],
                    [.0, .0, .0, .0, .0, .0]])

                # All Q matrices have to have matching dimensions (6x6 in our case).
                # So we have to zero-pad the Q matrix of the constant velocity filter
                # as it is naturally a 4x4.
                empty_array = np.zeros([6, 6])
                noise = Q_discrete_white_nosie(dim=2, var=self.dt, block_size=2, order_by_dim=False) 
                empty_array[:noise.shape[0],:noise.shape[1]] = noise  
                self.all_filters[i].Q = empty_array
            
            elif not math.isnan(self.info['x_pos']) and not math.isnan(self.info['x_vel']):
                rospy.loginfo('Instantiating Kalman filters with velocity and position')
                self.all_filters[i].x = np.array([self.info['x_pos'], self.info['y_pos'], self.info['x_vel'], self.info['y_vel'], .0, .0]).T
                self.all_filters[i].F = np.array([
                    [.1, .0, self.dt, .0, ((.1/.2)*self.dt)**2, .0],
                    [.0, .1, .0, self.dt, .0, ((.1/.2)*self.dt)**2],
                    [.0, .0, .1, .0, self.dt, .0],
                    [.0, .0, .0, .1, .0, self.dt],
                    [.0, .0, .0, .0, .1, .0],
                    [.0, .0, .0, .0, .0, .1]])

                self.all_filters[i].Q = Q_discrete_white_noise(dim=3, var=self.dt, block_size=2, order_by_dim=False) 

            # Define the state covariance matrix
            self.all_filters[i].P = np.array([
                [25.0*(self.info['pos_covar'][0]**2), .0, .0, .0, .0, .0],
                [.0, 25.0*(self.info['pos_covar'][7]**2), .0, .0, .0, .0],
                [.0, .0, 1.0**2, .0, .0, .0],
                [.0, .0, .0, 1.0**2, .0, .0],
                [.0, .0, .0, .0, 0.5**2, .0],
                [.0, .0, .0, .0, .0, 0.5**2]])







