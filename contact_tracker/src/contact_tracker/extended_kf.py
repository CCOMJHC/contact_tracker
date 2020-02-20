#!/usr/bin/env python

# Inherits from filterpy's KalmanFilter class 

# Author: Rachel White
# University of New Hampshire
# Date last modified: 02/04/2020

import math

from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise

import numpy as np
from numpy import zeros

DEBUG = True 

class ExtendedKalmanFilter(KalmanFilter):
    """
    Class to create custom Kalman filter.
    """


    def __init__(self, dim_x, dim_z, filter_order):
        """
        Define the constructor.

        filter_order -- order of the filter being created
        """
        
        KalmanFilter.__init__(self, dim_x, dim_z)
        self.filter_order = filter_order 
        self.bayes_factor = 0

    
    def get_bayes_factor(self):
        """
        Returns: Log Bayes Factor
        """
        return self.bayes_factor


    def set_bayes_factor(self, contact, testfactor):
        """ 
        Calculates Bayes Factor to test how consistant
        the measurement is with the model.
        
        Keyword arguments:
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
        logBF = calculateBayesFactor(2)
        
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
        S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R
        invS = np.linalg.inv(S) # Error is occurring because it's the same transposed.

        # h will be an offset from the current model providing an alternative
        # hypothesis. It is calculated as testfactor * model's uncertainty
        # prior to incorporating the measurement.
        
        # This errors out whenever there's a negative value in the diagonal
        #print(np.diag(self.P))
        h = np.sqrt(np.diag(self.P)) * testfactor

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
        ZHX0 = contact.Z - np.dot(self.H, self.x)

        # Here we need to apply a different alternate hypothesis for each
        # state variable depending on where the measurement falls (< or >)
        # relative to it.
        multiplier = [1 if x < 0 else -1 for x in (contact.Z - np.dot(self.H, self.x))]
        ZHX1 = np.abs(contact.Z - np.dot(self.H, self.x)) + multiplier * np.dot(self.H, h)

        log_likelihoodM0 = -0.5*(np.dot(ZHX0.T, np.dot(invS, ZHX0)))
        log_likelihoodM1 = -0.5*(np.dot(ZHX1.T, np.dot(invS, ZHX1)))

        # Calculate te Log Bayes Factor
        log_BF = log_likelihoodM0 - log_likelihoodM1

        self.bayes_factor = log_BF


    def set_Q(self, contact):
        """
        Recompute the value of Q for the Kalman filters in this Contact.
        """
        
        for kf in contact.filter_bank.filters:
            if kf.filter_order == 'first': 
                # All Q matrices have to have matching dimensions (6x6 in our case).
                # So we have to zero-pad the Q matrix of the constant velocity filter
                # as it is naturally a 4x4.
                empty_array = np.zeros([6, 6])
                noise = Q_discrete_white_noise(dim=2, var=contact.dt, block_size=2, order_by_dim=False) 
                empty_array[:noise.shape[0],:noise.shape[1]] = noise  
                kf.Q = empty_array
            
            elif kf.filter_order == 'second':
                kf.Q = Q_discrete_white_noise(dim=3, var=contact.dt, block_size=2, order_by_dim=False) 


    def set_F(self, contact):
        """
        Recompute the value of F (process model matrix) for the Kalman 
        filters in this Contact.
        """

        for kf in contact.filter_bank.filters:
            if kf.filter_order == 'first':
                kf.F = np.array([
                    [1., .0, contact.dt, .0, (0.5*contact.dt)**2, .0],
                    [.0, 1., .0, contact.dt, .0, (0.5*contact.dt)**2],
                    [.0, .0, 1., .0, contact.dt, .0],
                    [.0, .0, .0, 1., .0, contact.dt],
                    [.0, .0, .0, .0, .0, .0],
                    [.0, .0, .0, .0, .0, .0]])
                
            elif kf.filter_order == 'second':
                kf.F = np.array([
                    [1., .0, contact.dt, .0, (0.5*contact.dt)**2, .0],
                    [.0, 1., .0, contact.dt, .0, (0.5*contact.dt)**2],
                    [.0, .0, 1., .0, contact.dt, .0],
                    [.0, .0, .0, 1., .0, contact.dt],
                    [.0, .0, .0, .0, 1., .0],
                    [.0, .0, .0, .0, .0, 1.]])


    def set_H(self, contact, detect_info):
        """
        Recompute the values of H for the Kalman filters in this Contact.
        """

        for kf in contact.filter_bank.filters:
            if math.isnan(detect_info['x_vel']):
                kf.H = np.array([
                    [1., .0, .0, .0, .0, .0],
                    [.0, 1., .0, .0, .0, .0],
                    [.0, .0, .0, .0, .0, .0],
                    [.0, .0, .0, .0, .0, .0]])
            else:
                kf.H = np.array([
                    [1., .0, .0, .0, .0, .0],
                    [.0, 1., .0, .0, .0, .0],
                    [.0, .0, 1., .0, .0, .0],
                    [.0, .0, .0, 1., .0, .0]])


    def set_R(self, contact, detect_info):
        """
        Set each filter's R value for this contact besed on the pos_covar field from  
        a Detect message.

        Keyword arguments:
        detect_info -- the dictionary containing the detect info being checked
        """
        
        pc = detect_info['pos_covar']
        tc = detect_info['twist_covar']

        for kf in contact.filter_bank.filters:
            if math.isnan(detect_info['x_vel']):
                kf.R = np.array([[pc[0], .0],
                                 [.0, pc[7]]])
            
            else:
                kf.R = np.array([[pc[0], .0, .0, .0],
                                 [.0, pc[7], .0, .0],
                                 [.0, .0, tc[0], .0],
                                 [.0, .0, .0, tc[7]]])
                

       

