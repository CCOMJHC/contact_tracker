#!/usr/bin/env python

# Rachel White
# University of New Hampshire
# Date last modified: 01/07/2020

import math
import time
import rospy
import hashlib
import datetime

from filterpy.kalman import KalmanFilter

import kalman_tracker.contact


class KalmanTracker:
   """ 
   Class to create custom Kalman filter.
   """ 


   def __init__(self):
       """
       Define the constructor.
       """
       self.kalman_filter = KalmanFilter(dim_x=6, dim_z=6) # TODO: determine the parameters to pass here
       self.all_contacts = {}


    def callback(self, data):
        """
        Listen for detects and add to dictionary and filter if not already there. 
        """
       
        # Get necessary info from the Detect data
        x0 = 
        R = 
        z = 
        timestamp = datetime.now()
        
        # Generate new hash key
        hash_str = str(datetime.datetime.now().timestamp())
        hash_object = hashlib.md5(hash_str.encode())
        hash_key = hash_object.hexdigest()

        if detect_is_contact(self.all_contacts, hash_key): 
            # Create new contact object
            c = Contact(x0, R, z, timestamp, hash_key)  

            # Integrate this with self.kalman_filter 
            self.kalman_filter.predict()
            self.kalman_filter.update(c.z)

            # Also add to all_contacts
            self.all_contacts[hash_key] = c
         
        else:
            # update the time stamp for when this contact was last accessed so we know not
            # to remove it anytime soon.
            c = all_contacts[hash_key]
            c.last_accessed = datetime.now()

        # TODO: remove items from the dictionary that have not been accessed in a while    


    def run(self):
        """
        Initialize the node and set it to subscribe to the detects topic.
        """

        rospy.init_node('tracker', anonymous=True)
        rospy.Subscriber('/detects', Detect, self.callback) # TODO: Not sure about the 2nd param
        rospy.spin()


if __name__=='__main__':
    try:
        kt = KalmanTracker()
        kt.run()
    except rospy.ROSInterruptException:
        print('Falied to initialize KalmanTracker')
        pass
