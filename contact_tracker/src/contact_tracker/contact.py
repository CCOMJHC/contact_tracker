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
        self.dt = 1 
        self.first_measured = timestamp
        self.last_measured = timestamp
        self.last_xpos = 0
        self.last_ypos = 0
        self.last_xvel = 0
        self.last_yvel = 0

        # Variables that keep track of data for plotting purposes
        self.xs = []
        self.zs = []
        self.ps = []
        self.times = []
        
        # Other important variables
        self.info = detect_info
        self.Z = None
        self.id = timestamp 
        self.bfs = []
        

    def init_filters(self):
        """
        Initialize each filter in this Contact's filter bank.
        """
        
        for i in range(0, len(self.all_filters)):
            
            if not math.isnan(self.info['x_pos']) and math.isnan(self.info['x_vel']):
                rospy.loginfo('Instantiating Kalman filters with position but without velocity')
                self.all_filters[i].x = np.array([self.info['x_pos'], self.info['y_pos'], 0, 0, 0, 0]).T
                self.all_filters[i].F = np.array([
                    [1, 0, self.dt, 0, ((1/2)*self.dt)**2, 0],
                    [0, 1, 0, self.dt, 0, ((1/2)*self.dt)**2],
                    [0, 0, 1, 0, self.dt, 0],
                    [0, 0, 0, 1, 0, self.dt],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0]])
            
            elif not math.isnan(self.info['x_pos']) and not math.isnan(self.info['x_vel']):
                rospy.loginfo('Instantiating Kalman filters with velocity and position')
                self.all_filters[i].x = np.array([self.info['x_pos'], self.info['y_pos'], self.info['x_vel'], self.info['y_vel'], 0, 0]).T
                self.all_filters[i].F = np.array([
                    [1, 0, self.dt, 0, ((1/2)*self.dt)**2, 0],
                    [0, 1, 0, self.dt, 0, ((1/2)*self.dt)**2],
                    [0, 0, 1, 0, self.dt, 0],
                    [0, 0, 0, 1, 0, self.dt],
                    [0, 0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 0, 1]])

    
            # Define the state covariance matrix
            self.all_filters[i].P = np.array([
                [25*(self.info['pos_covar'][0]**2), 0, 0, 0, 0, 0],
                [0, 25*(self.info['pos_covar'][7]**2), 0, 0, 0, 0],
                [0, 0, 1**2, 0, 0, 0],
                [0, 0, 0, 1**2, 0, 0],
                [0, 0, 0, 0, 0.5**2, 0],
                [0, 0, 0, 0, 0, 0.5**2]])

            # Define the noise covariance 
            self.all_filters[i].Q = np.array([
                [((1/20)*self.dt)**5, 0, ((1/8)*self.dt)**4, 0, ((1/6)*self.dt)**3, 0],
                [0, ((1/20)*self.dt)**5, 0, ((1/8)*self.dt)**4, 0, ((1/6)*self.dt)**3],
                [((1/8)*self.dt)**4, 0, ((1/3)*self.dt)**3, 0, ((1/2)*self.dt)**2, 0],
                [0, ((1/8)*self.dt)**4, 0, ((1/3)*self.dt)**3, 0, ((1/2)*self.dt)**2],
                [((1/6)*self.dt)**3, 0, ((1/2)*self.dt)**2, 0, self.dt, 0],
                [0, ((1/6)*self.dt)**3, 0, ((1/2)*self.dt)**2, 0, self.dt]])



    def set_Q(self):
        """
        Recompute the value of Q for the Kalman filters in this Contact.
        """
        
        for i in range(0, len(self.filter_bank.filters)):
            self.filter_bank.filters[i].Q = np.array([
                [((1/20)*self.dt)**5, 0, ((1/8)*self.dt)**4, 0, ((1/6)*self.dt)**3, 0],
                [0, ((1/20)*self.dt)**5, 0, ((1/8)*self.dt)**4, 0, ((1/6)*self.dt)**3],
                [((1/8)*self.dt)**4, 0, ((1/3)*self.dt)**3, 0, ((1/2)*self.dt)**2, 0],
                [0, ((1/8)*self.dt)**4, 0, ((1/3)*self.dt)**3, 0, ((1/2)*self.dt)**2],
                [((1/6)*self.dt)**3, 0, ((1/2)*self.dt)**2, 0, self.dt, 0],
                [0, ((1/6)*self.dt)**3, 0, ((1/2)*self.dt)**2, 0, self.dt]])


    def set_F(self):
        """
        Recompute the value of F for the Kalman filters in this Contact.
        """

        for i in range(0, len(self.filter_bank.filters)):
            if i == 0:
                # Define the process model matrix for the constant velocity filter
                self.filter_bank.filters[i].F = np.array([
                    [1, 0, self.dt, 0, ((1/2)*self.dt)**2, 0],
                    [0, 1, 0, self.dt, 0, ((1/2)*self.dt)**2],
                    [0, 0, 1, 0, self.dt, 0],
                    [0, 0, 0, 1, 0, self.dt],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0]])
            else:
                # Define the process model matrix for the constant acceleration filter
                self.filter_bank.filters[i].F = np.array([
                    [1, 0, self.dt, 0, ((1/2)*self.dt)**2, 0],
                    [0, 1, 0, self.dt, 0, ((1/2)*self.dt)**2],
                    [0, 0, 1, 0, self.dt, 0],
                    [0, 0, 0, 1, 0, self.dt],
                    [0, 0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 0, 1]])


    def set_H(self, detect_info):
        """
        Recompute the values of H for the Kalman filters in this Contact.
        """

        for i in range(0, len(self.filter_bank.filters)):
            if math.isnan(detect_info['x_vel']):
                self.filter_bank.filters[i].H = np.array([
                    [1, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0]])
            else:
                self.filter_bank.filters[i].H = np.array([
                    [1, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0]])


    def set_R(self, detect_info):
        """
        Set each filter's R value for this contact besed on the pos_covar field from  
        a Detect message.

        Keyword arguments:
        detect_info -- the dictionary containing the detect info being checked
        """
        
        pc = detect_info['pos_covar']

        for i in range(0, len(self.filter_bank.filters)):
            self.filter_bank.filters[i].R = np.array([
                [pc[0], pc[1], pc[2], pc[3], pc[4], pc[5]],
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
            self.Z = [self.last_xpos, self.last_ypos, self.info['x_vel'], self.info['y_vel']]
        elif math.isnan(detect_info['x_vel']):
            self.Z = [self.info['x_pos'], self.info['y_pos'], self.last_xvel, self.last_yvel]
        else:
            self.Z = [self.info['x_pos'], self.info['y_pos'], self.info['x_vel'], self.info['y_vel']]
        

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
