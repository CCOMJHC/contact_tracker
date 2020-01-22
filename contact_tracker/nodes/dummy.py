#!/usr/bin/env python

# Generates dummy data for initial testing of tracker.py

import rospy
import sys
from numpy.random import randn

from marine_msgs.msg import Detect

class Dummy():

    def __init__(self):
        
        self.pub = rospy.Publisher('/detects', Detect, queue_size=1)

        x_pos = 0
        y_pos = 0
        test_with_velocity = False 

        while x_pos < 300:
            d = rospy.Duration(randn() * 1)
            rospy.loginfo(d)
            msg = Detect()
            msg.p.pose.pose.position.x = x_pos + randn() * 10
            msg.p.pose.pose.position.y = y_pos + randn() * 10
            
            if test_with_velocity:    
                msg.t.twist.twist.linear.x = 1.0
                msg.t.twist.twist.linear.y = 1.0
            
            x_pos += 1
            y_pos += 1

            self.pub.publish(msg)
            rospy.sleep(d)


if __name__=='__main__':
    
    rospy.init_node('dummy')
    try:
        d = Dummy()
    except:
        rospy.ROSInterruptException
        pass


