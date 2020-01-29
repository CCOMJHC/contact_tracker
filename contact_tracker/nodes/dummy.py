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

        x_pos = args.xpos
        y_pos = args.ypos 
        x_vel = args.xvel
        y_vel = args.yvel 
        
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
    
    arg_parser = argparse.ArgumentParser(description='Send fake Detect data to the tracker node for testing purposes.')
    arg_parser.add_argument('xpos', type=float, help='initial x position of the object')
    arg_parser.add_argument('ypos', type=float, help='initial y position of the object')
    arg_parser.add_argument('xvel', type=float, help='initial x velocity of the object')
    arg_parser.add_argument('yvel', type=float, help='initial y velocity of the object')
    args = arg_parser.parse_args()

    rospy.init_node('dummy')
    try:
        d = Dummy(args)
    except:
        rospy.ROSInterruptException
        pass


if __name__=='__main__':
    main()

