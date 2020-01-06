#!/usr/bin/env python
        
# A contact is identifed by position, not id, and is
# independent of the sensor that produced it. 

# Rachel White
# University of New Hampshire
# Date last modified: 01/07/2020

import math
import time
import rospy
import datetime

import kalman_tracker.contact
from marine_msgs.msgs import Detect

from filterpy.kalman import KalmanFilter
from filterpy.kalman import update
from filterpy.kalman import predict

MAX_TIME = 50.00
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


    def detect_not_already_contact(all_contacts, hash_key):
        """
        Returns true if this contact is already in the dictionary, 
        false otherwise.
        """

        if hash_key in all_contacts:
            return true
        
        return false


    def callback(self, data):
        """
        Listen for detects and add to dictionary and filter if not already there. 
        """
       
        ####### INITIALIZE VARIABLES #######

        # Get necessary info from the Detect data
        detect_info = {
                'header': data.header,
                'sensor_id': data.sensor_id,
                'pos_seq': data.pose.header.seq,
                'twist_seq': data.twist.header.seq,
                'pos_stamp': data.pose.header.stamp,
                'twist_stamp': data.twist.header.stamp, 
                'pos_frame_id': data.pose.header.frame_id,
                'twist_frame_id': data.twist.header.frame_id,
                'pos_covar': data.pose.pose.covariance, 
                'twist_covar': data.twist.twist.covariance,
                'x_pos': 0, 
                'x_vel': 0, 
                'y_pos': 0, 
                'y_vel': 0, 
                'z_pos': 0, 
                'z_vel': 0
                }
        
        # Assign values only if they are not NaNs
        if data.pose.pose.pose.position.x != 'NaN': 
            detect_info['x_pos'] = data.pose.pose.position.x
        
        if data.pose.pose.pose.position.y != 'NaN':
            detect_info['y_pos'] = data.pose.pose.position.y

        if data.pose.pose.pose.position.z != 'NaN':
            detect_info['z_pos'] = data.pose.pose.position.z

        if data.twist.twist.twist.linear.x != 'NaN':
            detect_info['x_vel'] = data.twist.twist.twist.linear.x

        if data.twist.twist.twist.linear.y != 'NaN':
            detect_info['y_vel'] = data.twist.twist.twist.linear.y

        if data.twist.twist.twist.linear.z != 'NaN':
            detect_info['z_vel'] = data.twist.twist.twist.linear.z

        
        # Check to see that if one coordinate is not NaN, neither is the other 
        if (detect_info['x_pos'] != 0 and detect_info['y_pos'] == 0) or (detect_info['x_pos'] == 0 and detect_info['y_pos'] != 0):
            continue # TODO: figure out what to do in these cases? 
        if (detect_info['x_vel'] != 0 and detect_info['y_vel'] == 0) or (detect_info['x_vel'] == 0 and detect_info['y_vel'] != 0):
            continue

        contact_id = (x_pos, y_pos) # TODO: Refine this to account for movement in the contact
        timestamp = datetime.datetime.now().timestamp()


        ####### CREATE OR UPDATE CONTACT WITH VARIABLES #######
        
        # Create new contact object.
        if detect_not_already_contact(self.all_contacts, contact_id): 
            
            # If there was no velocity, the vector will only have two fields
            kf = None 
            if x_vel == 0:
                kf = KalmanFilter(dim_x=2, dim_z=2)
            else:
                kf = KalmanFilter(dim_x=4, dim_z=4)

            c = Contact(detect_info, kf, timestamp, contact_id)  
            if kf.dim_x == 2:
                c.init_kf(dt)
            else:
                c.init_kf_with_velocity(dt)

            # Add this new object to all_contacts
            self.all_contacts[contact_id] = c
         
        else:
            # Update the time stamp for when this contact was last accessed so we know not
            # to remove it anytime soon.
            c = all_contacts[contact_id]
            c.last_accessed = timestamp 

        # Integrate this with self.kalman_filter 
        c.kf.predict()
        c.kf.update(c.z)

        # Remove items from the dictionary that have not been accessed in a while    
        for item in self.all_contacts:
            if timestamp - item.last_accessed > MAX_TIME:
                del all_contacts[item]


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
