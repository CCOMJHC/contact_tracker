#!/usr/bin/env python

# Generates dummy data for initial testing of tracker.py

import rospy
import sys
import random

from marine_msgs.msg import Detect

class Dummy():

    def __init__(self):
        
        self.pub = rospy.Publisher('/detects', Detect, queue_size=1)
        r = rospy.Rate(50)

        while not rospy.is_shutdown():
            msg = Detect()

            msg.p.pose.pose.position.x = float(random.randrange(0, 1000, 1))
            msg.p.pose.pose.position.y = float(random.randrange(0, 1000, 1))
            msg.t.twist.twist.linear.x = 1.0
            msg.t.twist.twist.linear.y = 1.0
            
            rospy.loginfo(msg)
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


