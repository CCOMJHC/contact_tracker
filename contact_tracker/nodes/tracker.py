#!/usr/bin/env python

# A contact is identifed by position, not id, and is
# independent of the sensor that produced it.

# Author: Rachel White
# University of New Hampshire
# Date last modified: 01/07/2020

import math
import time
import rospy
import datetime
import numpy as np

import contact_tracker.contact
from marine_msgs.msg import Detect

from filterpy.kalman import KalmanFilter
from filterpy.kalman import update
from filterpy.kalman import predict


# TODO: Move these to a config file
MAX_TIME = 50.00
INITIAL_VELOCITY = 5**2
dt = 1

class KalmanTracker:
    """
    Class to create custom Kalman filter.
    """


    def __init__(self):
        """
        Define the constructor.
        """
        self.all_contacts = {}


    def detect_not_already_contact(self, all_contacts, hash_key):
        """
        Returns true if this contact is already in the dictionary,
        false otherwise.
        """

        if hash_key in all_contacts:
            return True 

        return False 


    def callback(self, data):
        """
        Listen for detects and add to dictionary and filter if not already there.
        """
        
        rospy.loginfo(data.p)

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
                'x_pos': 0,
                'x_vel': 0,
                'y_pos': 0,
                'y_vel': 0,
                'z_pos': 0,
                'z_vel': 0
                }

        # Assign values only if they are not NaNs
        if data.p.pose.pose.position.x != 'NaN':
            detect_info['x_pos'] = data.p.pose.pose.position.x

        if data.p.pose.pose.position.y != 'NaN':
            detect_info['y_pos'] = data.p.pose.pose.position.y

        if data.p.pose.pose.position.z != 'NaN':
            detect_info['z_pos'] = data.p.pose.pose.position.z

        if data.t.twist.twist.linear.x != 'NaN':
            detect_info['x_vel'] = data.t.twist.twist.linear.x

        if data.t.twist.twist.linear.y != 'NaN':
            detect_info['y_vel'] = data.t.twist.twist.linear.y

        if data.t.twist.twist.linear.z != 'NaN':
            detect_info['z_vel'] = data.t.twist.twist.linear.z


        # Check to see that if one coordinate is not NaN, neither is the other
        if ((detect_info['x_pos'] != 0 and detect_info['y_pos'] == 0) or (detect_info['x_pos'] == 0 and detect_info['y_pos'] != 0)):
           pass
            # TODO: figure out what to do in these cases?
        if ((detect_info['x_vel'] != 0 and detect_info['y_vel'] == 0) or (detect_info['x_vel'] == 0 and detect_info['y_vel'] != 0)):
           pass

        contact_id = (detect_info['x_pos'], detect_info['y_pos']) # TODO: Refine this to account for movement in the contact
        timestamp = str(datetime.datetime.now())


        #######################################################
        ####### CREATE OR UPDATE CONTACT WITH VARIABLES #######
        #######################################################

        # Create new contact object.
        if not contact_id in self.all_contacts:

            # If there was no velocity, the state vector will only have two values.
            kf = None
            if detect_info['x_vel'] == 0:
                kf = KalmanFilter(dim_x=2, dim_z=2)
            else:
                kf = KalmanFilter(dim_x=4, dim_z=4)

            c = contact_tracker.contact.Contact(detect_info, kf, timestamp, contact_id)
            if kf.dim_x == 2:
                c.init_kf(dt)
            else:
                c.init_kf_with_velocity(dt)

            # Add this new object to all_contacts
            self.all_contacts[contact_id] = c

        else:
            # Update the time stamp for when this contact was last accessed so we know not
            # to remove it anytime soon.
            c = self.all_contacts[contact_id]
            c.last_accessed = timestamp

        # Add to self.kalman_filter
        c = self.all_contacts[contact_id]
        c.kf.predict()
        c.kf.update([c.info['x_pos'], c.info['y_pos']])

        # Remove items from the dictionary that have not been accessed in a while
        for item in self.all_contacts:
            if timestamp - item.last_accessed > MAX_TIME:
                del self.all_contacts[item]


    def run(self):
        """
        Initialize the node and set it to subscribe to the detects topic.
        """

        rospy.init_node('tracker', anonymous=True)
        rospy.Subscriber('/detects', Detect, self.callback)
        rospy.spin()


if __name__=='__main__':

    try:
        kt = KalmanTracker()
        kt.run()

    except rospy.ROSInterruptException:
        print('Falied to initialize KalmanTracker')
        pass
