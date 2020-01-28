#!/usr/bin/env python

# Generates dummy data for initial testing of tracker.py

import rospy
import argparse
import sys
from numpy.random import randn

from marine_msgs.msg import Detect

class Dummy():

    def __init__(self, args):
        
        self.pub = rospy.Publisher('/detects', Detect, queue_size=1)

        x_pos = 0
        y_pos = 0
        x_vel = 1 
        y_vel = 1 
        
        while x_pos < 300:
            d = rospy.Duration(randn() * 1)
            msg = Detect()
            msg.header.stamp = rospy.get_rostime()
            coin_flip = randn() 
           
            # Generate message with position and velocity
            if coin_flip > 0:    
                msg.pose.pose.position.x = x_pos + randn()
                msg.pose.pose.position.y = y_pos + randn()
                msg.twist.twist.linear.x = 1.0
                msg.twist.twist.linear.y = 1.0
                print(msg.header.stamp, ': Generating message with position and velocity: ', msg.pose.pose.position.x)
            
            # Generate message with position and not velocity
            elif coin_flip <= 0 and coin_flip >= -1: 
                msg.pose.pose.position.x = x_pos + randn() 
                msg.pose.pose.position.y = y_pos + randn()
                msg.twist.twist.linear.x = float('nan') 
                msg.twist.twist.linear.y = float('nan')
                print(msg.header.stamp, ': Generating message with position and not velocity: ', msg.pose.pose.position.x)

            # Generate message with velocity and not position
            else:
                msg.pose.pose.position.x = float('nan') 
                msg.pose.pose.position.y = float('nan') 
                msg.twist.twist.linear.x = x_vel + randn() 
                msg.twist.twist.linear.y = x_vel + randn() 
                print(msg.header.stamp, ': Generating message with velocity and not position: ', msg.pose.pose.position.x)
            
            x_pos += 1
            y_pos += 1
            self.pub.publish(msg)
            rospy.sleep(d)


def main():
    
    arg_parser = argparse.ArgumentParser(description='TBD')
    arg_parser.add_argument('-detect_fields', type=str, choices=['pos_only', 'vel_only', 'both', 'x_pos_no_y_pos', 'y_pos_no_x_pos', 'x_vel_no_y_vel', 'y_vel_no_x_vel'], help='type of data to be sent with the Detect messages')
    args = arg_parser.parse_args()

    rospy.init_node('dummy')
    try:
        d = Dummy(args)
    except:
        rospy.ROSInterruptException
        pass


if __name__=='__main__':
    main()

