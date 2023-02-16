#!/usr/bin/env python

# Inherits from filterpy's KalmanFilter class 

# Author: Rachel White
# University of New Hampshire
# Date last modified: 03/20/2020

import math

from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise
from filterpy.common import Q_continuous_white_noise
from filterpy.stats import likelihood

import numpy as np
from numpy import zeros

DEBUG = True 

class ContactKalmanFilter(KalmanFilter):
    """
    Class to create custom ContactKalmanFilter.
    """


    def __init__(self, dim_x, dim_z, filter_type):
        """
        Define the constructor.

        filter_type -- type/order of filter being created
        """
        
        KalmanFilter.__init__(self, dim_x, dim_z)
        self.filter_type = filter_type 
        self.bayes_factor = 0.0
        self.ll = 0.0


    def predict_prior(self, u=None, B=None, F=None, Q=None):
        """
        Predict next state (prior) using the Kalman filter state propagation
        equations. This method does NOT update the state, but is used for 
        hypothesis testing. 

        Parameters
        ----------

        u : np.array
            Optional control vector. If not `None`, it is multiplied by B
            to create the control input into the system.

        B : np.array(dim_x, dim_z), or None
            Optional control transition matrix; a value of None
            will cause the filter to use `self.B`.

        F : np.array(dim_x, dim_x), or None
            Optional state transition matrix; a value of None
            will cause the filter to use `self.F`.

        Q : np.array(dim_x, dim_x), scalar, or None
            Optional process noise matrix; a value of None will cause the
            filter to use `self.Q`.
        """

        if B is None:
            B = self.B
        if F is None:
            F = self.F
        if Q is None:
            Q = self.Q
        elif np.isscalar(Q):
            Q = np.eye(self.dim_x) * Q

        # x = Fx + Bu
        if B is not None and u is not None:
            self.x_prior = np.dot(F, self.x) + np.dot(B, u)
        else:
            self.x_prior = np.dot(F, self.x)

        # P = FPF' + Q
        self.P_prior = self._alpha_sq * np.dot(np.dot(F, self.P), F.T) + Q


    def get_log_likelihood(self):
        """
        Returns: Log Likelihood of this filter.
        """
        return self.ll

     
    def get_bayes_factor(self):
        """
        Returns: Log Bayes Factor of this filter.
        """
        return self.bayes_factor

    def set_likelihood(self,contact):
        '''
        NOT USED
        Sets the likelihood of the measurements given the model prediction.
        '''

        self.L = likelihood(contact.Z, 
                                     self.x, 
                                     self.P_prior, 
                                     self.H, 
                                     self.R)   
        self.ll = np.log(self.L)

    def calculate_measurement_likelihood_from_prior(self):
        '''
        Sets the likelihood of the measurements given the model prediction.
        '''
        self.L = likelihood(self.Z,
                            self.x_prior,
                            self.P_prior,
                            self.H,
                            self.R)
        self.ll = np.log(self.L)

    def set_log_likelihood(self, contact):
        """
        NOT USED
        Calculates the log likelihood of the contact given the measurement. 

        Keyword arguments:
        contact -- the contact object for which to retrieve the likelihood given the current measurement. 
        """

        S = np.dot(self.H, np.dot(self.P_prior, self.H.T)) + self.R
        invS = np.linalg.inv(S) 

        ZHX0 = contact.Z - np.dot(self.H, self.x_prior) 

        log_likelihoodM0 = -0.5*(np.dot(ZHX0.T, np.dot(invS, ZHX0)))
        self.ll = log_likelihoodM0


    def calculate_bayes_factor(self, Nsigma = 3):
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
        S = np.dot(self.H, np.dot(self.P_prior, self.H.T)) + self.R
        invS = np.linalg.inv(S) # Error is occurring because it's the same transposed.

        # h will be an offset from the current model providing an alternative
        # hypothesis. It is calculated as testfactor * model's uncertainty
        # prior to incorporating the measurement.
        
        # This errors out whenever there's a negative value in the diagonal
        #print(np.diag(self.P))
        h = np.sqrt(np.diag(self.P_prior)) * Nsigma

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
        ZHX0 = self.Z - np.dot(self.H, self.x_prior) # shouldn't this be abs?
        '''print('Z: ', contact.Z)
        print('x: ', self.x)'''

        # Here we need to apply a different alternate hypothesis for each
        # state variable depending on where the measurement falls (< or >)
        # relative to it.
        multiplier = [1.0 if x < 0 else -1.0 for x in (self.Z - np.dot(self.H, self.x_prior))]
        ZHX1 = np.abs(self.Z - np.dot(self.H, self.x_prior)) + multiplier * np.dot(self.H, h)

        log_likelihoodM0 = -0.5*(np.dot(ZHX0.T, np.dot(invS, ZHX0)))
        log_likelihoodM1 = -0.5*(np.dot(ZHX1.T, np.dot(invS, ZHX1)))

        # Calculate the Log Bayes Factor
        '''print('_________________________________________')
        print('invS: ', invS)
        print('+++++++++++++++++++++++++++++++++++++++++')
        print('ZHX0: ', ZHX0)
        print('invS * ZHX0: ', np.dot(invS, ZHX0))
        print('np.dot(ZHX0, ans): ', np.dot(ZHX0.T, np.dot(invS, ZHX0))) 
        print('-0.5*ans: ', np.dot(ZHX0.T, np.dot(invS, ZHX0)))
        print('log_likelihoodM0: ', log_likelihoodM0)
        print('+++++++++++++++++++++++++++++++++++++++++')
        print('ZHX1: ', ZHX1)
        print('invS * ZHX1: ', np.dot(invS, ZHX1))
        print('np.dot(ZHX1, ans): ', np.dot(ZHX1.T, np.dot(invS, ZHX1)))  
        print('-0.5*ans: ', np.dot(ZHX1.T, np.dot(invS, ZHX1))) 
        print('loglikelihoodM1: ', log_likelihoodM1)
        print('+++++++++++++++++++++++++++++++++++++++++')
        '''
        log_BF = log_likelihoodM0 - log_likelihoodM1

        self.bayes_factor = log_BF

    '''
    def set_Q(self, contact):
        """
        Recompute the value of Q for the Kalman filters in this contact.

        Keyword arguments:
        contact -- contact object for which to recompute Q
        """
        
        for kf in contact.filter_bank.filters:
            if kf.filter_type == 'first': 
                # All Q matrices have to have matching dimensions (6x6 in our case).
                # So we have to zero-pad the Q matrix of the constant velocity filter
                # as it is naturally a 4x4.
                empty_array = np.zeros([6, 6])
                #noise = Q_discrete_white_noise(dim=2, var=self.vel_var, dt=contact.dt, block_size=2, order_by_dim=False) 
                noise = Q_continuous_white_noise(dim=2, 
                                                 spectral_density=self.vel_var,
                                                 dt=contact.dt,
                                                 block_size=2,
                                                 order_by_dim=False)
                empty_array[:noise.shape[0],:noise.shape[1]] = noise  
                kf.Q = empty_array
            
            elif kf.filter_type == 'second':
                #kf.Q = Q_discrete_white_noise(dim=3, var=self.acc_var, dt=contact.dt, block_size=2, order_by_dim=False) 
                kf.Q = Q_continuous_white_noise(dim=3,
                                                spectral_density=self.acc_var,
                                                dt=contact.dt,
                                                block_size=2,
                                                order_by_dim=False)    
    '''
    
    def set_F(self, dt = 1.):
        """
        Recompute the value of F (process model matrix) for the Kalman 
        filters in this Contact.

        Keyword arguments:
        contact -- contact object for which to recompute F 
        """

        if self.filter_type == 'first':
            self.F = np.array([
                [1., .0, dt, .0, 0.5*dt**2, .0],
                [.0, 1., .0, dt, .0, 0.5*dt**2],
                [.0, .0, 1., .0, dt, .0],
                [.0, .0, .0, 1., .0, dt],
                [.0, .0, .0, .0, .0, .0],
                [.0, .0, .0, .0, .0, .0]])

        elif self.filter_type == 'second':
            self.F = np.array([
                [1., .0, dt, .0, 0.5*dt**2, .0],
                [.0, 1., .0, dt, .0, 0.5*dt**2],
                [.0, .0, 1., .0, dt, .0],
                [.0, .0, .0, 1., .0, dt],
                [.0, .0, .0, .0, 1., .0],
                [.0, .0, .0, .0, .0, 1.]])


    def set_H(self, detect_type):
        """
        Recompute the values of H for the Kalman filters in this Contact.

        Keyword arguments:
        contact -- contact object for which to recompute H 
        """

        if detect_type == "POSITIONONLY":

            self.H = np.array([
                [1., .0, .0, .0, .0, .0],
                [.0, 1., .0, .0, .0, .0],
                [.0, .0, .0, .0, .0, .0],
                [.0, .0, .0, .0, .0, .0]])

        elif detect_type == "POSITIONANDVELOCITY":
            self.H = np.array([
                [1., 0., 0., 0., 0., 0.],
                [0., 1., 0., 0., 0., 0.],
                [0., 0., 1., 0., 0., 0.],
                [0., 0., 0., 1., 0., 0.]])
        elif detect_type == "VELOCITYONLY":
            self.h = np.array([
                [0., 0., 1., 0., 0., 0.],
                [0., 0., 0., 1., 0., 0.]
            ])
        else:
            print("ERROR!!! Detect type incorrect: %s" % detect_type)

    def set_Z(self, detect_info):
        ''' Set the measurement vector from detection information.'''

        if detect_info['type'] == "POSITIONONLY":
            # The filterpy filter object does not accommodate a variable
            # measurement model. One must have z=4x1 for every measurement.
            # So we fake it by populating the missing fields from the previous
            # state.
            self.Z = [detect_info['x_pos'], detect_info['y_pos'], self.x[3], self.x[4]]
        elif detect_info['type'] == "POSITIONANDVELOCITY":
            self.Z = [detect_info['x_pos'],
                                     detect_info['y_pos'],
                                     detect_info['x_vel'],
                                     detect_info['y_vel']]
        elif detect_info["type"] == "VELOCITYONLY":
            self.Z = [detect_info['x_vel'], detect_info['y_vel']]

    def set_R(self, detect_info):
        """
        Set each filter's R value for this contact besed on the pos_covar field from  
        a Detect message.

        Keyword arguments:
        contact -- contact object for which to set R 
        detect_info -- the dictionary containing the detect info being checked
        """
        
        pc = detect_info['pos_covar']
        tc = detect_info['twist_covar']

        if detect_info["type"] == "POSITIONONLY":
            '''
            self.R = np.array([[pc[0], 0.],
                               [0., pc[7]]])
                               '''

            # The filterpy filter object does not accommodate a variable
            # measurement model. One must have R=4x4 for every measurement.
            # So we fake it by populating the missing fields from the previous
            # state. Here he uncertainty is increased to ensure allow the
            # actual measurements to have sufficient influence.

            self.R = np.array([[pc[0], .0, .0, .0],
                               [.0, pc[7], .0, .0],
                               [.0, .0, self.P[2][2]*10, .0],
                               [.0, .0, .0, self.P[3][3]*10]])


        elif detect_info["type"] == "POSITIONANDVELOCITY":
            self.R = np.array([[pc[0], .0, .0, .0],
                             [.0, pc[7], .0, .0],
                             [.0, .0, tc[0], .0],
                             [.0, .0, .0, tc[7]]])
        elif detect_info["type"] == "VELOCITYONLY":
            self.R = np.array([[tc[0], .0],
                             [.0, tc[7]]])


       
    def UKF_ToMeasurementRTheta(self,Xos,Yos):
        """
        For an Unscented Kalman Filter Implmenetation - A utility method for
        implementing an Unscented Kalman Filter that converts the internal UKF
        state (in this case in Cartesian coordinates) to radial measurements
        :param Xos: Own-ship X coordinate at the time of measurement.
        :param Yos: Own-ship Y coordinate at the time of measuremnet.
        :return: [R, Theta]
        where R is the range to the contact and Theta is the azimuthal angle.
        """

        dx = self.x[0] - Xos
        dy = self.x[1] - Yos
        R = np.sqrt(dx**2 + dy**2)
        Theta = np.arctan2(dy,dx)
        return [R,Theta]

    def UDK_To_MeasurementR(self,Xos,Yos):
        """
        For an Unscented Kalman Filter Implmenetation - A utility method for
        implementing an Unscented Kalman Filter that converts the internal UKF
        state (in this case in Cartesian coordinates) to range-only measurements
        :param Xos: Own-ship X coordinate at the time of measurement.
        :param Yos: Own-ship Y coordinate at the time of measuremnet.
        :return: [R]
        where R is the range to the contact.
        """

        dx = self.x[0] - Xos
        dy = self.x[1] - Yos
        R = np.sqrt(dx**2 + dy**2)
        return [R]

    def UDK_To_MeasurementTheta(self,Xos,Yos):
        """
        For an Unscented Kalman Filter Implmenetation - A utility method for
        implementing an Unscented Kalman Filter that converts the internal UKF
        state (in this case in Cartesian coordinates) to range-only measurements
        :param Xos: Own-ship X coordinate at the time of measurement.
        :param Yos: Own-ship Y coordinate at the time of measuremnet.
        :return: [R]
        where R is the range to the contact.
        """

        dx = self.x[0] - Xos
        dy = self.x[1] - Yos
        Theta = np.arctan2(dy,dx)
        return [Theta]

    def normalizeAngle(self,theta):
        """
        For an Unscented Kalman Filter Implementation - a utility method to
        ensure angles considered fall within -pi - pi.
        :param theta: Angle to normalize.
        :return: theta: The normalized angle.
        """

        theta = theta % (2.0 * np.pi)
        if theta > np.pi:
            theta -= (2*np.pi)
        return theta

    def UKF_innovation(self,Z):
        """
        A method to calculate the Kalman Filter "innovation" for an
        Unscented Kalman Filter which measures in polar coordinates
        (range and angle) but tracks in Cartesian coordinates, handling
        the case when the difference in the angle measurement and the
        the angle estimate wraps.
        :param Z:
        :return:
        """
        y = Z - dot(self.H,self.x)
        y[2] = self.normalizeAngle(y[2])
        return y

    