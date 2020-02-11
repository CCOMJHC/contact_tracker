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
        self.Z = None
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
        Initialize each filter in this Contact's filter bank given only values for velocity.
        """

        for i in range(0, len(self.filter_bank.filters)):
            # Define the state variable vector
            self.filter_bank.filters[i].x = np.array([0, 0, self.info['x_vel'], self.info['y_vel']]).T

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
                              [0, 0, 0, self.variance]])



    def init_kf_with_position_and_velocity(self):
        """
        Initialize each filter in this Contact's filter bank given only values for velocity.
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


    def set_Q(self, epoch):
        """
        Recompute the values of Q for the Kalman filters in this Contact.

        Keyword arguments:
        epoch -- time since this contact was last incorporated into the filter
        """
        
        for i in range(0, len(self.filter_bank.filters)):
            self.filter_bank.filters[i].Q = Q_discrete_white_noise(dim=2, dt=epoch*self.dt, var=self.variance, block_size=3, order_by_dim=False)

    
    def set_R(self, pc):
        """
        Set each filter's R value for this contact besed on the pos_covar field from  
        a Detect message.

        Keyword arguments:
        pc -- the pose covariance field from the Detect message being eexamined
        """
        
        for i in range(0, len(self.filter_bank.filters)):
            self.filter_bank.filters[i].R = np.array([[pc[0], pc[1], pc[2], pc[3], pc[4], pc[5]],
                                                      [pc[6], pc[7], pc[8], pc[9], pc[10], pc[11]], 
                                                      [pc[12], pc[13], pc[14], pc[15], pc[16], pc[17]],
                                                      [pc[18], pc[19], pc[20], pc[21], pc[22], pc[23]],
                                                      [pc[24], pc[25], pc[26], pc[27], pc[28], pc[29]], 
                                                      [pc[30], pc[31], pc[32], pc[33], pc[34], pc[35]]])  


    def set_Z(self, detect_info):
        """
        Set the measurement vector based on information sent in the detect message.

        Keyword arguments:
        detect_info -- the dictionary containing the detect info being checked
        """
        
        if math.isnan(detect_info['x_pos']):
            self.Z = [self.last_xpos, self.last_ypos, self.info['x_vel'], self.info['y_vel'], 0, 0]
        elif math.isnan(detect_info['x_vel']):
            self.Z = [self.info['x_pos'], self.info['y_pos'], self.last_xvel, self.last_yvel, 0, 0]
        else:
            self.Z = [self.info['x_pos'], self.info['y_pos'], self.info['x_vel'], self.info['y_vel'], 0, 0]
        

    def calculate_bayes_factor(self, K, testfactor):
        """ 
        Calculates Bayes Factor to test how consistant
        the measurement is with the model.
        
        Keyword arguments:
        K -- filterpy.kalman.KalmanFilter object
        testfactor -- Scalar factor indiating how many standard deviations
             outside which the the measurement will be deemed inconsistent
                    
        Returns: Log Bayes Factor
                    
        The Bayes Factor provides a method to discern which measurements
        are likely to be of the modeled system or not.  
        
        The Bayes Factor tests the relative odds of two models. 
        In this context it is used to test the odds that a
        new measurement is statistially consistant with the 
        current model, or a hypothetical model displaced from it.
        The displacement is the 'testfactor' x sigma where sigma is
        the uncertainty of the model. 
        
        Baysian Forcasting and Dynmaic Models, Second Edition,
        West and Harrison, 1997, page 394.
        
        (6) Following Jeffreys (1961), a log Bayes' factor of 
        1(-1) indicates evidence in favour of model 0 (1), a
        value of 2(-2) indicating the evidence to be strong.
        
        EXAMPLE:
        
        Given a Kalman filter object K, and a new measurement vector, Z,
        having uncertainty matrix, R, one can calculate the log Bayes Factor 
        against a hypothetical model displaced two standard deviations 
        away with the following:
        
        K.R = newmeasurementR()
        logBF = calculateBayesFactor(K,Z,2)
        
        If the result is greater than 2, than the hypothesis that the 
        measurement is more consistant with the alternate model can be rejected
        and the measurement can be integrated into the model under test.
        
        if logBF > 2:
            K.update(Z)
        
        See:
        Baysian Forcasting and Dynmaic Models, Second Editio,n
        West and Harrison, 1997. for more details.
        """ 
        
        # First we calculate the system uncertainty, S, which includes
        # the uncertainty of the model, P, propagated to the measurement 
        # time and the uncertaint of the measurement, R.
        # Note, if K.predict() has already been called, the result of this 
        # calculation would be available in K.S. Calculating it explicitly
        # here allows us to delay propagating the model in the event that 
        # we decide not the include the measurement. 
        S = np.dot(K.H,np.dot(K.P, K.H.T)) + K.R
        invS = np.linalg.inv(S)
        
        # h will be an offset from the current model providing an alternative 
        # hypothesis. It is calculated as testfactor * model's uncertainty
        # prior to incorporating the measurement.
        h = np.sqrt(np.diag(K.P)) * testfactor

        '''print('S: ', S)
        print('invS: ', invS)
        print('h: ', h)'''
        
        # The "likelihood" of the measurement under the existing 
        # model, and that of an alternative model are calculated as 
        #
        #     L = (z - HX).T Hinv(S)H.T (Z - HX)
        # and 
        #     L2 = (z - X + h).T Hinv(S)H.T (Z - X + h)
        #
        # However care must be taken in the sign of the estimate offset, h to
        # ensure the model shifts away from the measurement values relative to 
        # the estimate. This calcualtion is done in piece-meal steps to 
        # make it more clear and easier to debug. 
        ZHX0 = self.Z - np.dot(K.H,K.x)
        
        # Here we need to apply a different alternate hypothesis for each
        # state variable depending on where the measurement falls (< or >)
        # relative to it. 
        multiplier = [1 if x < 0 else -1 for x in (self.Z - np.dot(K.H,K.x))]
        ZHX1 = np.abs(self.Z - np.dot(K.H,K.x)) + multiplier * np.dot(K.H,h)

        '''print('ZHX0: ', ZHX0)
        print('ZHX1: ', ZHX1)
        print('multiplier: ', multiplier)'''
    
        log_likelihoodM0 = -0.5*(np.dot(ZHX0.T, np.dot(invS, ZHX0)))
        log_likelihoodM1 = -0.5*(np.dot(ZHX1.T, np.dot(invS, ZHX1)))
        
        # Calculate te Log Bayes Factor
        log_BF = log_likelihoodM0 - log_likelihoodM1

        '''print('M0: ', log_likelihoodM0)
        print('M1: ', log_likelihoodM1)
        print('log_BF: ', log_BF)'''
    
        return log_BF
