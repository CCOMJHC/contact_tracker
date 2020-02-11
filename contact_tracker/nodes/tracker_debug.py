#!/usr/bin/env python

# A contact is identifed by position, not id, and is
# independent of the sensor that produced it.

# Author: Rachel White
# University of New Hampshire
# Date last modified: 02/04/2020

import math
import time
import rospy
import datetime
import argparse
import numpy as np
from numpy import zeros
import matplotlib.pyplot as plt

import contact_tracker.contact
from marine_msgs.msg import Detect

from filterpy.kalman import IMMEstimator
from filterpy.kalman import KalmanFilter
from filterpy.kalman import update
from filterpy.kalman import predict
from filterpy.common import Q_discrete_white_noise
from filterpy.stats.stats import plot_covariance

from dynamic_reconfigure.server import Server
from contact_tracker.cfg import contact_trackerConfig

DEBUG = True 

class ContactTracker:
    """
    Class to create custom Kalman filter.
    """


    def __init__(self):
        """
        Define the constructor.

        max_time -- amount of time that must ellapse before an item is deleted from all_contacts
        dt -- time step for the Kalman filters
        initial_velocity -- velocity at the start of the program
        """

        self.all_contacts = {}


    def plot_x_vs_y(self, output_path):
        """
        Visualize results of the Kalman filter by plotting the measurements against the 
        predictions of the Kalman filter.

        Keyword arguments:
        output_path -- The path that the plot will be saved to
        """
        
        all_mxs = []
        all_mys = []
        all_pxs = []
        all_pys = []

        for contact in self.all_contacts:
            c = self.all_contacts[contact]
        
            m_xs = []
            m_ys = []
            p_xs = []
            p_ys = []

            for i in c.zs:
                m_xs.append(i[0])
                m_ys.append(i[1])
            
            for i in c.xs:
                p_xs.append(i[0])
                p_ys.append(i[1])

            all_mxs.append(m_xs)
            all_mys.append(m_ys)
            all_pxs.append(p_xs)
            all_pys.append(p_ys)
        
        for i in range(0, len(all_mxs)):
            plt.scatter(all_mxs[i], all_mys[i], linestyle='-', label='kf' + str(i) + ' measurements', color='y')
            plt.plot(all_pxs[i], all_pys[i], label='kf' + str (i) + ' predictions')

        plt.legend()
        plt.xlabel('x position')
        plt.ylabel('y position')
        plt.xlim(0, 500)
        plt.ylim(0, 500)
        plt.savefig(output_path + '.png')


    def plot_x_vs_time(self, output_path):
        """
        Visualize results of the Kalman filter by plotting the measurements against the 
        predictions of the Kalman filter.
        
        Keyword arguments:
        output_path -- The path that the plot will be saved to
        """

        all_mxs = []
        all_pxs = []
        all_times = []

        for contact in self.all_contacts:
            c = self.all_contacts[contact]
         
            m_xs = []
            p_xs = []

            for i in c.zs:
                m_xs.append(i[0])
        
            for i in c.xs:
                p_xs.append(i[0])

            all_mxs.append(m_xs)
            all_pxs.append(p_xs)
            all_times.append(c.times)

        for i in range(0, len(all_mxs)):
            plt.scatter(all_times[i], all_mxs[i], linestyle='-', label='kf ' + str(i) + ' measurements', color='y')
            plt.plot(all_times[i], all_pxs[i], label='kf ' + str (i) + ' predictions')

        plt.legend()
        plt.xlabel('time')
        plt.ylabel('x position')
        plt.ylim(0, 300)
        plt.savefig(output_path + '.png')


    def plot_ellipses(self, output_path):
        """
        Visualize results of the Kalman filter by plotting the measurements against the 
        predictions of the Kalman filter.
        
        Keyword arguments:
        output_path -- The path that the plot will be saved to
        """

        all_pxs = []
        all_pys = []
        all_zs = []
        all_ps = []

        for contact in self.all_contacts:
            c = self.all_contacts[contact]
            p_xs = []
            p_ys = []
            cur_ps = []
            z_means = []

            for i in c.xs:
                p_xs.append(i[0])
                p_ys.append(i[1])

            all_pxs.append(p_xs)
            all_pys.append(p_ys)

            for i in range(0, len(c.xs), 4):
                z_mean = np.array([c.zs[i][0], c.zs[i][1]])
                cur_p = c.ps[i]
                plot_covariance(mean=z_mean, cov=cur_p)

            all_zs.append(z_means)
            all_ps.append(cur_ps)

            
        for i in range(0, len(all_pxs)):
            plt.plot(all_pxs[i], all_pys[i], label='predictions', color='g')

        plt.xlabel('x position')
        plt.ylabel('y position')
        plt.xlim(0, 300)
        plt.ylim(0, 300)
        plt.legend()
        plt.savefig(output_path + '.png')
        
    
    def dump_detect(self, detect_info):
        """
        Print the contents of a contact's detect_info dictionary for debugging purposes.

        Keyword arguments:
        detect_info -- the dictionary containing the detect info to be printed
        """
        
        rospy.loginfo('+++++++ DETECT +++++++')
        for k, v in detect_info.items():
            rospy.loginfo(k, ': ', v)


    def dump_contacts(self):
        """
        Print the contents of the all_contacts  dictionary for debugging purposes.
        """
        
        rospy.loginfo('+++++++ CONTACTS +++++++')
        for k, v in self.all_contacts.items():
            rospy.loginfo(k, ': ', v)


    def populate_detect_info(self, data):
        """
        Initialize the data structure for this detect message and check 
        for empty position and velocity fields. Return an empty dictionary 
        if one position or velocity field is empty and the other position
        or velocity field is not.

        Keyword arguments:
        data -- The detect message that was just transmitted
        """
        
        # Get necessary info from the Detect data
        detect_info = {
                'header': data.header,
                'sensor_id': data.sensor_id,
                'pos_covar': data.pose.covariance,
                'twist_covar': data.twist.covariance,
                'x_pos': float('nan'),
                'x_vel': float('nan'),
                'y_pos': float('nan'),
                'y_vel': float('nan'),
                'z_pos': float('nan'),
                'z_vel': float('nan')
                }
        
        # Assign values only if they are not NaNs
        if not math.isnan(data.pose.pose.position.x):
            detect_info['x_pos'] = float(data.pose.pose.position.x)

        if not math.isnan(data.pose.pose.position.y):
            detect_info['y_pos'] = float(data.pose.pose.position.y)

        if not math.isnan(data.pose.pose.position.z):
            detect_info['z_pos'] = float(data.pose.pose.position.z)

        if not math.isnan(data.twist.twist.linear.x):
            detect_info['x_vel'] = float(data.twist.twist.linear.x)

        if not math.isnan(data.twist.twist.linear.y):
            detect_info['y_vel'] = float(data.twist.twist.linear.y)

        if not math.isnan(data.twist.twist.linear.z):
            detect_info['z_vel'] = float(data.twist.twist.linear.z)


        # Check to see that if one coordinate is not NaN, neither is the other
        if ((not math.isnan(detect_info['x_pos']) and math.isnan(detect_info['y_pos'])) or (math.isnan(detect_info['x_pos']) and not math.isnan(detect_info['y_pos']))):
            rospy.loginfo('ERROR: x_pos and y_pos both were not nans...returning')
            return {} 
        if ((not math.isnan(detect_info['x_vel']) and math.isnan(detect_info['y_vel'])) or (math.isnan(detect_info['x_vel']) and not math.isnan(detect_info['y_vel']))):
            rospy.loginfo('ERROR: x_vel and y_vel both were not nans...returning')
            return {}

        return detect_info


    def calculate_bayes_factor(self, K, Z, testfactor):
        """ 
        Calculates Bayes Factor to test how consistant
        the measurement is with the model.
        
        Keyword arguments:
        K -- filterpy.kalman.KalmanFilter object
        Z -- Measurement vector
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
        ZHX0 = Z - np.dot(K.H,K.x)
        
        # Here we need to apply a different alternate hypothesis for each
        # state variable depending on where the measurement falls (< or >)
        # relative to it. 
        multiplier = [1 if x < 0 else -1 for x in (Z - np.dot(K.H,K.x))]
        ZHX1 = np.abs(Z - np.dot(K.H,K.x)) + multiplier * np.dot(K.H,h)
    
        log_likelihoodM0 = -0.5*(np.dot(ZHX0.T, np.dot(invS, ZHX0)))
        log_likelihoodM1 = -0.5*(np.dot(ZHX1.T, np.dot(invS, ZHX1)))
        
        # Calculate te Log Bayes Factor
        log_BF = log_likelihoodM0 - log_likelihoodM1
    
        return log_BF
    

    def check_all_contacts(self, detect_info, new_stamp):
        """
        Iterate over every contact in the dictionary and return contact the current Detect is 
        most likely associated with. Otherwise, return the timestamp of the current Detect message
        as the new hash_key for the new Contact that will be made.

        Keyword arguments:
        detect_info -- the dictionary containing the detect info to be checked
        new_stamp -- the timestamp from the header of the contact we're checking all others against

        Returns:
        new_stamp -- 
        """
 
        for contact in self.all_contacts:
            c = self.all_contacts[contact]

            for kf in c.filter_bank.filters:
                # TODO: Should we return after a single update is performed, or can all 
                # filters in the bank potentially be updated? I need to know this before 
                # I potentially begin refactoring. If this is a one-time thing per detect
                # message, I can potantially return an id for the particular filter that 
                # is a BayesFactor match. otherwise, we can keep the per-contact ids and  
                # call c.filter_bank.update(Z). 
                
                Z = c.get_z(detect_info) 
                
                kf.R = np.array([[100, 0, 0, 0, 0, 0],
                                 [0, 100, 0, 0, 0, 0],
                                 [0, 0, 2, 0, 0, 0],
                                 [0, 0, 0, 2, 0, 0],
                                 [0, 0, 0, 0, 2, 0],
                                 [0, 0, 0, 0, 0, 2]])
                logBF = self.calculate_bayes_factor(kf, Z, 2)
                print(logBF) 
                if logBF > 2: 
                    return c.id
                 
        # No appropriate contacts were found, so return the stamp of 
        # the Detect message being checked
        print('no matching contacts')
        return new_stamp

        
    def delete_stale_contacts(self):
        """
        Remove items from the dictionary that have not been measured in a while
        """
        
        for contact_id in self.all_contacts:
            cur_contact = self.all_contacts[contact_id]
            time_between_now_and_last_measured = (rospy.get_rostime() - cur_contact.last_measured).to_sec()

            if time_between_now_and_last_measured > self.max_stale_contact_time:
                rospy.loginfo('deleting stale Contact from dictionary')
                del self.all_contacts[cur_contact]


    def reconfigure_callback(self, config, level):
        """
        Get the parameters from the cfg file and assign them to the member variables of the 
        KalmanTracker class.
        """

        self.max_stale_contact_time = config['max_stale_contact_time']
        self.initial_velocity = config['initial_velocity']
        self.variance = config['variance']
        return config


    def callback(self, data):
        """
        Listen for detects and add to dictionary and filter if not already there.

        Keyword arguments:
        data -- the Detect message transmitted
        """

        ########################################################
        ###### VARIABLE INITIALIZATION AND ERROR HANDLING ######
        ########################################################
         
        # Initialize variables and store in a dictionary.
        detect_info = self.populate_detect_info(data)
        if len(detect_info) == 0: 
            return

        #  Compare new measurement to prediction at same time.
        #  If there are no contacts yet, no need to traverse empty dictionary
        #  Otherwise, we have to check each contact in the dictionary to see if 
        #  it is a potential match for our current Detect message. 
        if len(self.all_contacts) == 0:
            contact_id  = data.header.stamp 
        else:
            contact_id = self.check_all_contacts(detect_info, data.header.stamp)
            

        #######################################################
        ####### CREATE OR UPDATE CONTACT WITH VARIABLES #######
        #######################################################
        
        epoch = 0
        if not contact_id in self.all_contacts: 
          
            first_order_kf = KalmanFilter(dim_x=6, dim_z=6)
            second_order_kf = KalmanFilter(dim_x=6, dim_z=6)
            all_filters = [first_order_kf, second_order_kf]
            c = contact_tracker.contact.Contact(detect_info, all_filters, self.variance, data.header.stamp)
            
            if not math.isnan(detect_info['x_pos']) and math.isnan(detect_info['x_vel']):
                rospy.loginfo('Instantiating Kalman filters with position but without velocity')
                c.init_kf_with_position_only()
            
            elif math.isnan(detect_info['x_pos']) and not math.isnan(detect_info['x_vel']):
                rospy.loginfo('Instantiating Kalman filters with velocity but without position')
                c.init_kf_with_velocity_only()
            
            elif not math.isnan(detect_info['x_pos']) and not math.isnan(detect_info['x_vel']):
                rospy.loginfo('Instantiating Kalman filters with velocity and position')
                c.init_kf_with_position_and_velocity()
            
            c.filter_bank = IMMEstimator(all_filters, c.mu, c.M)
            self.all_contacts[data.header.stamp] = c

        else:
            # Recompute the value for dt, and use it to update this Contact's KalmanFilter's Q(s).
            # Then update the time stamp for when this contact was last measured so we know not
            # to remove it anytime soon. 
            c = self.all_contacts[contact_id]
            c.last_measured = data.header.stamp
            epoch = (c.last_measured - c.first_measured).to_sec()
            
            c.dt = epoch
            c.recompute_q(epoch) 
            c.info = detect_info

            if not math.isnan(detect_info['x_pos']):
                c.last_xpos = detect_info['x_pos']
                c.last_ypos = detect_info['y_pos']
           
            if not math.isnan(detect_info['x_vel']):
                c.last_xvel = detect_info['x_vel']
                c.last_yvel = detect_info['y_vel']
        
        # Incorporate with filters in the filter_bank 
        c = self.all_contacts[contact_id]
        c.filter_bank.predict()
        Z = c.get_z(detect_info) 
        c.filter_bank.update(Z)

       
        # Append appropriate prior and measurements to lists here
        
        #print('------- ONE -------') # WAY off from the measurements now. 
        #print(c.filter_bank.x)
        #print('-------- BEFORE APPEND -------')
        #print(c.xs)

        c.xs.append(np.array([c.filter_bank.x[0], c.filter_bank.x[1]]))
        
        #print('-------- AFTER APPEND -------')
        #print(c.xs)
        #print('----------------------------')

        c.zs.append(np.array([c.info['x_pos'], c.info['y_pos']]))
        c.ps.append(c.filter_bank.P)
        c.times.append(epoch)

        # Remove items from the dictionary that have not been measured in a while
        #self.delete_stale_contacts()


    def run(self, args):
        """
        Initialize the node and set it to subscribe to the detects topic.
        """

        rospy.init_node('tracker', anonymous=True)
        srv = Server(contact_trackerConfig, self.reconfigure_callback)
        rospy.Subscriber('/detects', Detect, self.callback)
        rospy.spin()
        
        if args.plot_type == 'xs_ys':
            self.plot_x_vs_y(args.o)
        elif args.plot_type =='xs_times':
            self.plot_x_vs_time(args.o)
        elif args.plot_type == 'ellipses':
            self.plot_ellipses(args.o)


def main():
    
    arg_parser = argparse.ArgumentParser(description='Track contacts by applying Kalman filters to incoming detect messages. Optionally plot the results of the filter.')
    arg_parser.add_argument('-plot_type', type=str, choices=['xs_ys', 'xs_times', 'ellipses'], help='specify the type of plot to produce, if you want one')
    arg_parser.add_argument('-o', type=str, help='path to save the plot produced, default: tracker_plot, current working directory', default='tracker_plot')
    args = arg_parser.parse_args()

    try:
        ct = ContactTracker()
        ct.run(args)

    except rospy.ROSInterruptException:
        rospy.loginfo('Falied to initialize the ContactTracker')
        pass


if __name__=='__main__':
    main()


