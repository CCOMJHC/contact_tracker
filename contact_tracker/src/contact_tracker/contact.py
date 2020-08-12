#!/usr/bin/env python
# Class to create a Contact object and initialize its
# associated Kalman filters.

# Author: Rachel White
# University of New Hampshire
# Last Modified: 03/20/2020


import rospy
import numpy as np
import math

from filterpy.kalman import IMMEstimator
from filterpy.common import Q_continuous_white_noise
import contact_kf as CKF
from marine_msgs.msg import Detect, Contact
import time

class Contact:
    """
    Class to create contact object with its own Kalman filter bank.
    """

    def __init__(self, detect_info, contact_id):
        """
        Define the constructor.
        
        detect_info -- dictionary containing data from the detect message being used to create this contact
        all_filters -- list containing unique KalmanFilter objects for this specific contact object
        timestamp -- header from the detect message 
        """

        if detect_info is None:
            detect_info = self.fake_detectinfo() # for testing.

        # Variables for the IMM Estimator
        self.mu = np.array([0.5, 0.5])
        self.M = np.array([[0.95, 0.15],
                           [0.05, 0.85]])
        #self.M = np.array([[0.5, 0.5],
        #                   [0.5, 0.5]])

        '''self.mu = np.array([0.5, 0.5])
        self.M = np.array([[0.5, 0.5],
                           [0.5, 0.5]])'''


        
        # Variables the define the piecewise continuous white noise variance
        # model for 1st and 2nd order filters. Units are m^2/s^3 and characterize
        # both how much the uncertainty grows with each predict but also how
        # much the estimate is allowed to change with each step. 
        self.vel_var = 1
        # -noise 0.3
        self.acc_var = 0.1
        self.vel_var = 0.1
       # -noise 1
        self.acc_var = 0.1
        self.vel_var = 0.3

        # Variables that keep track of time
        self.dt = 1.0 
        self.last_measured = detect_info['header'].stamp
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
        self.last_detect_info = detect_info
        self.id = contact_id
        self.Z = None

        # Create the filter bank (uninitalized).
        first_order_kf = CKF.ContactKalmanFilter(dim_x=6, dim_z=4, filter_type='first')
        second_order_kf = CKF.ContactKalmanFilter(dim_x=6, dim_z=4, filter_type='second')
        # Note that all_filters will remain a pointer to the list of filters within the IMMEstimator object.
        self.all_filters = [first_order_kf, second_order_kf]

        # Initialize the filters and setup the IMM Estimator.
        self.init_filters()
        self.filter_bank = IMMEstimator(self.all_filters , self.mu, self.M)
        self.bayes_factor = 0

    def fake_detectinfo(self):

        detect = Detect()
        detect.header.stamp = rospy.Time.from_sec(time.time())
        detect_info = {
                'header': detect.header,
                'sensor_id': detect.sensor_id,
                'pos_covar': detect.pose.covariance,
                'twist_covar': detect.twist.covariance,
                'x_pos': float('nan'),
                'x_vel': float('nan'),
                'y_pos': float('nan'),
                'y_vel': float('nan'),
                'type': "UNKNOWN"
                }
        return detect_info

    def init_filters(self):
        """
        Initialize each filter in this contact's filter bank.
        """
        if self.last_detect_info is not None:
            self.set_X()
            self.set_P()
            self.set_F(dt=self.dt)
            self.set_H()
            self.set_Q()
            self.set_Z()
            self.set_R()

    def set_X(self,detect_info = None):
        '''Sets the state of the kalman filters for this contact.'''

        if detect_info is None:
            detect_info = self.last_detect_info

        for i in range(0,len(self.all_filters)):
            self.all_filters[i].x = np.array([detect_info['x_pos'],
                                              detect_info['y_pos'],
                                              0.,
                                              0.,
                                              0.,
                                              0.]).T

    def set_P(self, detect_info = None, posVarianceInflation = 100.0):
        ''' Sets the state covariance matrix for the kalman filters for this contact.'''

        if detect_info is None:
            detect_info = self.last_detect_info

        for i in range(0, len(self.all_filters)):
            self.all_filters[i].P = np.array([
                [posVarianceInflation * detect_info['pos_covar'][0], 0., 0., 0., 0., 0.],
                [.0, posVarianceInflation * detect_info['pos_covar'][7], 0., 0., 0., 0.],
                [0., 0., 5.0 ** 2, 0., 0., 0.],
                [0., 0., 0., 5.0 ** 2, 0., 0.],
                [0., 0., 0., 0., 1. ** 2, 0.],
                [0., 0., 0., 0., 0., 1. ** 2]])

    def set_Q(self):
        """
        Recompute the value of Q for the Kalman filters in this contact.

        Keyword arguments:
        contact -- contact object for which to recompute Q
        """
        
        for kf in self.all_filters:
            if kf.filter_type == 'first': 
                # All Q matrices have to have matching dimensions (6x6 in our case).
                # So we have to zero-pad the Q matrix of the constant velocity filter
                # as it is naturally a 4x4.
                empty_array = np.zeros([6, 6])
                #noise = Q_discrete_white_noise(dim=2, var=self.vel_var, dt=contact.dt, block_size=2, order_by_dim=False) 
                noise = Q_continuous_white_noise(dim=2, 
                                                 spectral_density=self.vel_var,
                                                 dt=self.dt,
                                                 block_size=2,
                                                 order_by_dim=False)
                empty_array[:noise.shape[0],:noise.shape[1]] = noise  
                kf.Q = empty_array
            
            elif kf.filter_type == 'second':
                #kf.Q = Q_discrete_white_noise(dim=3, var=self.acc_var, dt=contact.dt, block_size=2, order_by_dim=False) 
                kf.Q = Q_continuous_white_noise(dim=3,
                                                spectral_density=self.acc_var,
                                                dt=self.dt,
                                                block_size=2,
                                                order_by_dim=False)

    def set_F(self, dt=1.):
        ''' Set the transition matrix for each filter based on the time increment, dt since last measurement.'''
        for i in range(0,len(self.all_filters)):
            self.all_filters[i].set_F(dt=dt)

    def set_H(self):
        ''' Set the H matrix. What's that called??'''
        for i in range(0,len(self.all_filters)):
            self.all_filters[i].set_H(detect_type = self.last_detect_info['type'])

    def set_Z(self, detect_info = None):
        """
        Set the measurement vector based on information sent in the detect message.

        Keyword arguments:
        detect_info -- the dictionary containing the detect info being checked
        """
        
        '''
        if math.isnan(detect_info['x_pos']):
            self.Z = [self.last_xpos, self.last_ypos, self.last_detect_info['x_vel'], self.last_detect_info['y_vel']]
        
        elif math.isnan(detect_info['x_vel']):
            self.Z = [self.last_detect_info['x_pos'], self.last_detect_info['y_pos'], self.last_xvel, self.last_yvel]
        
        else:
            self.Z = [self.last_detect_info['x_pos'], self.last_detect_info['y_pos'], self.last_detect_info['x_vel'], self.last_detect_info['y_vel']]
        '''

        if detect_info is None:
            detect_info = self.last_detect_info
        else:
            self.last_detect_info = detect_info

        for i in range(0,len(self.all_filters)):
            self.all_filters[i].set_Z(detect_info)


        '''
        if math.isnan(detect_info['x_vel']):
            self.Z = [self.last_detect_info['x_pos'], self.last_detect_info['y_pos']]
        
        else:
            self.Z = [self.last_detect_info['x_pos'], self.last_detect_info['y_pos'], self.last_detect_info['x_vel'], self.last_detect_info['y_vel']]
        '''
 
    def set_R(self, detect_info = None):
        '''Set the measurement uncertainty covariance matrix'''

        if detect_info is None:
            detect_info = self.last_detect_info

        for i in range(0,len(self.all_filters)):
            self.all_filters[i].set_R(detect_info)

    def predict_prior(self):
        ''' Predict the state of each kalman filter at the detection time.

        This method will conduct a predict step to the measurement time, populating
        for each filter, x_prior and P_prior without updating the state variables,
        x, and P.

        x_prior and P_prior are used for statistical tests between the measurement
        and the models updated to the measurement time to determine if the measuremnt
        should be associated with the given contact or not.

        Note: for this to work properly, the following must be called first
        where c is an instance of this contact object.:
        c.last_detect_info = detect_info                                # sets detection info.
        c.dt = (detect_info['header'].stamp - c.last_measured).to_sec() # dt from current state to this detect.
        c.set_Z()    # Set measurements
        c.set_R()    # Set measurement covariance.
        c.set_H()    # Set H matrix.
        c.set_F(dt=c.dt)        # Set transition matrix for time step.
        '''


        for i in range(len(self.all_filters)):
            self.all_filters[i].predict_prior()

    def calculate_measurement_likelihood(self):
        ''' Calculates the measurement likelihood prior to incorporating measurement for each filter.'''

        for i in range(0,len(self.all_filters)):
            self.all_filters[i].calculate_measurement_likelihood_from_prior()

    def predictIMMprior(self):
        """
        Predict next state (prior) using the IMM state propagation
        equations.

        Parameters
        ----------

        u : np.array, optional
            Control vector. If not `None`, it is multiplied by B
            to create the control input into the system.
        """

        ''' This first block calculates the '''
        # compute mixed initial conditions
        xs, Ps = [], []
        for i, (f, w) in enumerate(zip(self.filter_bank.filters, self.filter_bank.omega.T)):
            x = np.zeros(self.filter_bank.x.shape)
            for kf, wj in zip(self.filter_bank.filters, w):
                x += kf.x * wj
            xs.append(x)

            P = np.zeros(self.filter_bank.P.shape)
            for kf, wj in zip(self.filter_bank.filters, w):
                y = kf.x - x
                P += wj * (np.outer(y, y) + kf.P)
            Ps.append(P)

        #  compute each filter's prior using the mixed initial conditions
        for i, f in enumerate(self.filter_bank.filters):
            # propagate using the mixed state estimate and covariance
            f.x = xs[i].copy()
            f.P = Ps[i].copy()
            f.predict_prior()

        # compute mixed IMM state and covariance and save posterior estimate
        x, P = self._compute_state_estimate()
        self.filter_bank.x_prior = x.copy()
        self.filter_bank.P_prior = P.copy()

    def _compute_state_estimate(self):
        """
        Computes the IMM's mixed state estimate from each filter using
        the the mode probability self.mu to weight the estimates.
        """
        x = np.zeros(shape=self.filter_bank.x.shape)
        for f, mu in zip(self.filter_bank.filters, self.filter_bank.mu):
            x += f.x * mu

        P = np.zeros(shape=self.filter_bank.P.shape)
        for f, mu in zip(self.filter_bank.filters, self.filter_bank.mu):
            y = f.x - self.filter_bank.x
            P += mu * (np.outer(y, y) + f.P)
        return x,P

    def calculate_IMM_bayes_factor(self,Nsigma = 3):
        ''' Calculate the Bayes Factor for the measurement and a hypothetical solution
        whose state is the predicted state of the model at the measurement time +
        a distance Nsigma away'''

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
        Baysian Forcasting and Dynmaic Models, Second Edition,
        West and Harrison, 1997. for more details.
        """

        #  A few things are needed from the filter bank.
        # and we can take them from the first filter.
        H = self.filter_bank.filters[0].H
        R = self.filter_bank.filters[0].R
        Z = self.filter_bank.filters[0].Z

        # First we calculate the system uncertainty, S, which includes
        # the uncertainty of the model, P, propagated to the measurement
        # time and the uncertaint of the measurement, R.
        # Note, if K.predict() has already been called, the result of this
        # calculation would be available in K.S. Calculating it explicitly
        # here allows us to delay propagating the model in the event that
        # we decide not the include the measurement.
        S = np.dot(H, np.dot(self.filter_bank.P_prior, H.T)) + R
        invS = np.linalg.inv(S)  # Error is occurring because it's the same transposed.

        # h will be an offset from the current model providing an alternative
        # hypothesis. It is calculated as testfactor * model's uncertainty
        # prior to incorporating the measurement.

        # This errors out whenever there's a negative value in the diagonal
        # print(np.diag(self.P))
        h = np.sqrt(np.diag(self.filter_bank.P_prior)) * Nsigma

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
        ZHX0 = Z - np.dot(H, self.filter_bank.x_prior)  # shouldn't this be abs?
        '''print('Z: ', contact.Z)
        print('x: ', self.x)'''

        # Here we need to apply a different alternate hypothesis for each
        # state variable depending on where the measurement falls (< or >)
        # relative to it.
        multiplier = [1.0 if x < 0 else -1.0 for x in (Z - np.dot(H, self.filter_bank.x_prior))]
        ZHX1 = np.abs(Z - np.dot(H, self.filter_bank.x_prior)) + multiplier * np.dot(H, h)

        log_likelihoodM0 = -0.5 * (np.dot(ZHX0.T, np.dot(invS, ZHX0)))
        log_likelihoodM1 = -0.5 * (np.dot(ZHX1.T, np.dot(invS, ZHX1)))

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
