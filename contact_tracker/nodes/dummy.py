#!/usr/bin/env python

# Generates dummy data for initial testing of tracker.py

import rospy
import sys

from marine_msgs.msg import Detect

class Dummy():

    def __init__(self):
        
        self.pub = rospy.Publisher('/detects', Detect, queue_size=1)
        r = rospy.Rate(50)

        while not rospy.is_shutdown():
            msg = Detect()
            #msg.header = 'this is my cool header so hands off!'
            #msg.header = Detect.header
 
            #rospy.loginfo(msg.header)
            self.pub.publish(msg)
            rospy.loginfo('data published!')
            r.sleep()


if __name__=='__main__':
    
    rospy.init_node('dummy')
    try:
        d = Dummy()
    except:
        rospy.ROSInterruptException
        pass


