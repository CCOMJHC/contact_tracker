#!/usr/bin/env python

# Rachel White
# University of New Hampshire
# Date last modified: 01/07/2020

import math
import time
import rospy
import hashlib
import datetime

import kalman_tracker.contact

from filterpy.kalman import KalmanFilter
from filterpy.kalman import update
from filterpy.kalman import predict



class KalmanTracker:
   """ 
   Class to create custom Kalman filter.
   """ 


   def __init__(self):
       """
       Define the constructor.
       """
       self.all_contacts = {}


    def detect_is_contact(all_contacts, hash_key):
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
        x0 = 
        timestamp = datetime.now()
        
        # Generate new hash key
        hash_str = str(datetime.datetime.now().timestamp())
        hash_object = hashlib.md5(hash_str.encode())
        hash_key = hash_object.hexdigest()

        if detect_is_contact(self.all_contacts, hash_key): 
            # Create new contact object
            kf = KalmanFilter(dim_x=6, dim_z=6)
            c = Contact(x0, kf, timestamp, hash_key)  

            # Integrate this with self.kalman_filter 
            c.kf.predict()
            c.kf.update(c.z)

            # Also add to all_contacts
            self.all_contacts[hash_key] = c
         
        else:
            # update the time stamp for when this contact was last accessed so we know not
            # to remove it anytime soon.
            c = all_contacts[hash_key]
            c.last_accessed = datetime.now()

            # TODO: Figure out if I need to call predict and update here too

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
