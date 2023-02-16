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
import argparse

import contact_tracker.contact
from .msg import Detect

from filterpy.kalman import KalmanFilter
from filterpy.kalman import update
from filterpy.kalman import predict
from filterpy.common import Q_discrete_white_noise
from filterpy.stats.stats import plot_covariance

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


    def dump(self, detect_info):
        """
        Print the contents of a contact's detect_info dictionary for debugging purposes.
        """
        
        print('+++++++ CONTACTS +++++++')
        for k, v in detect_info.items():
            print(k, v)


    def reconfigure_callback(self, config, level):
        """
        Get the parameters from the cfg file and assign them to the member variables of the 
        KalmanTracker class.
        """

        self.qhat = config['qhat']
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
        
        ####################################
        ####### INITIALIZE VARIABLES #######
        ####################################

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
            if DEBUG: print('ERROR: x_pos and y_pos both were not nans...returning')
            return 
        if ((not math.isnan(detect_info['x_vel']) and math.isnan(detect_info['y_vel'])) or (math.isnan(detect_info['x_vel']) and not math.isnan(detect_info['y_vel']))):
            if DEBUG: print('ERROR: x_vel and y_vel both were not nans...returning')
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
            
            if not math.isnan(detect_info['x_pos']) and math.isnan(detect_info['x_vel']):
                rospy.loginfo('Instantiating first-order Kalman filter with position but without velocity')
                kf = KalmanFilter(dim_x=4, dim_z=2)
                c = contact_tracker.contact.Contact(detect_info, kf, self.variance, data.header.stamp, contact_id)
                c.init_kf_with_position_only()
            
            elif math.isnan(detect_info['x_pos']) and not math.isnan(detect_info['x_vel']):
                rospy.loginfo('Instantiating first-order Kalman filter with velocity but without position')
                kf = KalmanFilter(dim_x=4, dim_z=2)
                c = contact_tracker.contact.Contact(detect_info, kf, self.variance, data.header.stamp, contact_id)
                c.init_kf_with_velocity_only()
            
            elif not math.isnan(detect_info['x_pos']) and not math.isnan(detect_info['x_vel']):
                rospy.loginfo('Instantiating first-order Kalman filter with velocity and position')
                kf = KalmanFilter(dim_x=4, dim_z=4)
                c = contact_tracker.contact.Contact(detect_info, kf, self.variance, data.header.stamp, contact_id)
                c.init_kf_with_position_and_velocity()
            
            '''elif not math.isnan(detect_info['x_acc']):
                rospy.loginfo('Instantiating second-order Kalman filter')
                kf = KalmanFilter(dim_x=6, dim_z=4)
                c = contact_tracker.contact.Contact(detect_info, kf, self.variance, data.header.stamp, contact_id)
                c.init_kf_with_acceleration()'''

            # Add this new object to all_contacts
            self.all_contacts[contact_id] = c

        else:
            # Recompute the value for dt, and use it to update this Contact's KalmanFilter's Q.
            # Then update the time stamp for when this contact was last measured so we know not
            # to remove it anytime soon. 
            c = self.all_contacts[contact_id]
            c.last_measured = data.header.stamp
            epoch = (c.last_measured - c.first_measured).to_sec()
            
            if DEBUG:
                rospy.loginfo(c.first_measured)
                rospy.loginfo(c.last_measured)
                rospy.loginfo(epoch)

            c.dt = epoch
            c.kf.Q = Q_discrete_white_noise(dim=4, dt=epoch*self.qhat, var=self.variance) 
            c.info = detect_info

        # Add to self.kalman_filter
        rospy.loginfo('Calling predict() and update()')
        c = self.all_contacts[contact_id]
        c.kf.predict()

        if c.kf.dim_z == 4:
            c.kf.update((c.info['x_pos'], c.info['y_pos'], c.info['x_vel'], c.info['y_vel']))
        else:
            c.kf.update((c.info['x_pos'], c.info['y_pos']))
        
        # Append appropriate prior and measurements to lists here
        c.xs.append(c.kf.x)
        c.zs.append(c.kf.z)
        c.ps.append(c.kf.P)
        c.times.append(epoch)

        # Remove items from the dictionary that have not been measured in a while
        for contact_id in self.all_contacts:
            cur_contact = self.all_contacts[contact_id]
            time_between_now_and_last_measured = (rospy.get_rostime() - cur_contact.last_measured).to_sec()

            if DEBUG:
                rospy.loginfo(cur_contact.last_measured)
                rospy.loginfo(time_between_now_and_last_measured)

            if time_between_now_and_last_measured > self.max_stale_contact_time:
                if DEBUG: print('deleting stale Contact from dictionary')
                del self.all_contacts[cur_contact]


    def run(self, args):
        """
        Initialize the node and set it to subscribe to the detects topic.
        """

        rospy.init_node('tracker', anonymous=True)
        srv = Server(contact_trackerConfig, self.reconfigure_callback)
        rospy.Subscriber('/detects', Detect, self.callback)
        rospy.spin()
        

def main():
    
    arg_parser = argparse.ArgumentParser(description='Track contacts by applying Kalman filters to incoming detect messages.')
    args = arg_parser.parse_args()

    try:
        kt = KalmanTracker()
        kt.run(args)

    except rospy.ROSInterruptException:
        rospy.loginfo('Falied to initialize KalmanTracker')
        pass


if __name__=='__main__':
    main()


