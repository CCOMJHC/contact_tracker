#!/usr/bin/env python

# A contact is identifed by position, not id, and is
# independent of the sensor that produced it.

# Author: Rachel White
# University of New Hampshire
# Date last modified: 01/15/2020

import math
import time
import rospy
import datetime
import numpy as np
import matplotlib.pyplot as plt

import contact_tracker.contact
from marine_msgs.msg import Detect

from filterpy.kalman import KalmanFilter
from filterpy.kalman import update
from filterpy.kalman import predict
from filterpy.common import Q_discrete_white_noise

from dynamic_reconfigure.server import Server
from contact_tracker.cfg import contact_trackerConfig

DEBUG = True


class KalmanTracker:
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


    def plot_x_vs_y(self):
        """
        Visualize results of the Kalman filter by plotting the measurements against the 
        predictions of the Kalman filter.
        """

        c = self.all_contacts[1]
        
        m_xs = []
        m_ys = []
        p_xs = []
        p_ys = []

        print('PRINTING MEASUREMENTS')
        for i in c.zs:
            print(i)
            m_xs.append(i[0])
            m_ys.append(i[1])
        
        print('PRINTING PREDICTIONS')
        for i in c.xs:
            print(i)
            p_xs.append(i[0])
            p_ys.append(i[1])

        plt.scatter(m_xs, m_ys, linestyle='-', label='measurements', color='y')
        plt.plot(p_xs, p_ys, label='predictions', color='b')
        plt.legend()
        plt.xlabel('x position')
        plt.ylabel('y position')
        plt.xlim(c.xs[0][0], 300)
        plt.ylim(c.xs[0][0], 300)
        plt.show()


    def plot_x_vs_time(self):
        """
        Visualize results of the Kalman filter by plotting the measurements against the 
        predictions of the Kalman filter.
        """

        c = self.all_contacts[1]
        
        m_xs = []
        p_xs = []

        for i in c.zs:
            m_xs.append(i[0])
        
        for i in c.xs:
            p_xs.append(i[0])
 
        print('PRINTING TIMES')
        for i in c.times:
            print(i)


        plt.scatter(c.times, m_xs, linestyle='-', label='measurements', color='y')
        plt.plot(c.times, p_xs, label='predictions', color='b')
        plt.legend()
        plt.xlabel('time')
        plt.ylabel('x position')
        plt.ylim(c.xs[0][0], 300)
        plt.show()


    def reconfigure_callback(self, config, level):
        """
        Get the parameters from the cfg file and assign them to the member variables of the 
        KalmanTracker class.
        """

        self.qhat = config['qhat']
        self.max_time = config['max_time']
        self.initial_velocity = config['initial_velocity']
        return config


    def callback(self, data):
        """
        Listen for detects and add to dictionary and filter if not already there.

        Keyword arguments:
        data -- the Detect message transmitted
        """
        
        ####################################
        ####### INITIALIZE VARIABLES #######
        ####################################

        # Get necessary info from the Detect data
        detect_info = {
                'header': data.header,
                'sensor_id': data.sensor_id,
                'pos_seq': data.p.header.seq,
                'twist_seq': data.t.header.seq,
                'pos_stamp': data.p.header.stamp,
                'twist_stamp': data.t.header.stamp,
                'pos_frame_id': data.p.header.frame_id,
                'twist_frame_id': data.t.header.frame_id,
                'pos_covar': data.p.pose.covariance,
                'twist_covar': data.t.twist.covariance,
                'x_pos': float('nan'),
                'x_vel': float('nan'),
                'y_pos': float('nan'),
                'y_vel': float('nan'),
                'z_pos': float('nan'),
                'z_vel': float('nan')
                }

        # Assign values only if they are not NaNs
        if data.p.pose.pose.position.x != 'NaN':
            detect_info['x_pos'] = float(data.p.pose.pose.position.x)

        if data.p.pose.pose.position.y != 'NaN':
            detect_info['y_pos'] = float(data.p.pose.pose.position.y)

        if data.p.pose.pose.position.z != 'NaN':
            detect_info['z_pos'] = float(data.p.pose.pose.position.z)

        if data.t.twist.twist.linear.x != 'NaN':
            detect_info['x_vel'] = float(data.t.twist.twist.linear.x)

        if data.t.twist.twist.linear.y != 'NaN':
            detect_info['y_vel'] = float(data.t.twist.twist.linear.y)

        if data.t.twist.twist.linear.z != 'NaN':
            detect_info['z_vel'] = float(data.t.twist.twist.linear.z)


        # Check to see that if one coordinate is not NaN, neither is the other
        if ((detect_info['x_pos'] != float('nan') and detect_info['y_pos'] == float('nan')) or (detect_info['x_pos'] == float('nan') and detect_info['y_pos'] != float('nan'))):
           return 
        if ((detect_info['x_vel'] != float('nan') and detect_info['y_vel'] == float('nan')) or (detect_info['x_vel'] == float('nan') and detect_info['y_vel'] != float('nan'))):
           return 
        
        if DEBUG:
            contact_id = 1
        else:    
            contact_id = (detect_info['x_pos'], detect_info['y_pos']) # TODO: Refine this to account for movement in the contact


        #######################################################
        ####### CREATE OR UPDATE CONTACT WITH VARIABLES #######
        #######################################################

        # Create new contact object.
        epoch = 0
        if not contact_id in self.all_contacts:
          
            kf = None
            c = None
            start_time = time.time()
            
            if detect_info['x_pos'] != float('nan') and detect_info['x_vel'] == float('nan'):
                rospy.loginfo('Instantiating first-order Kalman filter with position but without velocity')
                kf = KalmanFilter(dim_x=4, dim_z=2)
                c = contact_tracker.contact.Contact(detect_info, kf, start_time, contact_id)
                c.init_kf_without_velocity()
            
            elif detect_info['x_pos'] == float('nan') and detect_info['x_vel'] != float('nan'):
                rospy.loginfo('Instantiating first-order Kalman filter with velocity but without position')
                kf = KalmanFilter(dim_x=4, dim_z=2)
                c = contact_tracker.contact.Contact(detect_info, kf, start_time, contact_id)
                c.init_kf_with_velocity()
            
            '''elif detect_info['x_pos'] != math.nan and detect_info['x_vel'] != math.nan and detect_info['x_acc'] == math.nan:
                rospy.loginfo('Instantiating first-order Kalman filter with velocity and position')
                kf = KalmanFilter(dim_x=4, dim_z=2)
                c = contact_tracker.contact.Contact(detect_info, kf, start_time, contact_id)
                c.init_kf()
            
            elif detect_info['x_acc'] != math.nan:
                rospy.loginfo('Instantiating second-order Kalman filter')
                kf = KalmanFilter(dim_x=4, dim_z=2)
                c = contact_tracker.contact.Contact(detect_info, kf, start_time, contact_id)
                c.init_kf_with_acceleration()'''

            # Add this new object to all_contacts
            self.all_contacts[contact_id] = c

        else:
            # Recompute the value for dt, and use it to update this Contact's KalmanFilter's Q.
            # Then update the time stamp for when this contact was last accessed so we know not
            # to remove it anytime soon. 
            c = self.all_contacts[contact_id]
            c.last_accessed = time.time()
            epoch = c.last_accessed - c.first_accessed
            c.dt = epoch
            c.kf.Q = Q_discrete_white_noise(dim=4, dt=epoch*self.qhat, var = 0.04**2) #TODO: Figure out what the variance should be here.
            c.info = detect_info

        # Add to self.kalman_filter
        rospy.loginfo('Calling predict() and update()')
        c = self.all_contacts[contact_id]
        c.kf.predict()
        c.kf.update((c.info['x_pos'], c.info['y_pos']))
        
        # Append appropriate prior and measurements to lists here
        c.xs.append(c.kf.x)
        c.zs.append(c.kf.z)
        c.times.append(epoch)

        # Remove items from the dictionary that have not been accessed in a while
        '''for contact_id in self.all_contacts:
            cur_contact = self.all_contacts[contact_id]
            if int(timestamp.second / 60) - int(cur_contact.last_accessed.second / 60) > MAX_TIME:
                del self.all_contacts[cur_contact]'''


    def run(self):
        """
        Initialize the node and set it to subscribe to the detects topic.
        """

        rospy.init_node('tracker', anonymous=True)
        srv = Server(contact_trackerConfig, self.reconfigure_callback)
        rospy.Subscriber('/detects', Detect, self.callback)
        rospy.spin()

        if DEBUG:
            rospy.loginfo('plotting the results')
            #self.plot_x_vs_y()
            self.plot_x_vs_time()


if __name__=='__main__':

    try:
        kt = KalmanTracker()
        kt.run()

    except rospy.ROSInterruptException:
        rospy.loginfo('Falied to initialize KalmanTracker')
        pass
