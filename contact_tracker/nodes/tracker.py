#!/usr/bin/env python

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
       
        # Get necessary info from the Detect data
        header = data.header
        seq = data.header.seq
        stamp = data.header.stamp
        frame_id = data.header.frame_id
        x_pos = data.pose.pose.position.x
        y_pos = data.pose.pose.position.y
        z_pos = data.pose.pose.position.z

        # TODO: Figure out how to determine whether two contacts are the same
        # (I think they each have a unique ID, but I need to verify this).
        #contact_id = ???
        timestamp = datetime.datetime.now().timestamp()
        
        if detect_not_already_contact(self.all_contacts, contact_id): 
            # Create new contact object
            kf = KalmanFilter(dim_x=6, dim_z=6) # TODO: Are these the correct dimensions?

            # TODO: Figure out how to use the detect info to initialize the Kalman
            # filter for this Contact object
            c = Contact(header, kf, timestamp, contact_id)  

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
        # TODO: make this more efficient
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
