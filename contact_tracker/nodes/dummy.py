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
        x_vel = 0
        y_vel = 0
        
        while x_pos < 300:
            d = rospy.Duration(randn() * 1)
            rospy.loginfo(d)
            msg = Detect()
            msg.header.stamp = rospy.get_rostime()
           
            if args.detect_fields == 'both':    
                msg.p.pose.pose.position.x = x_pos + randn() * 10
                msg.p.pose.pose.position.y = y_pos + randn() * 10
                msg.t.twist.twist.linear.x = 1.0
                msg.t.twist.twist.linear.y = 1.0
                x_pos += 1
                y_pos += 1
                x_vel += 1
                y_vel += 1
            
            elif args.detect_fields == 'pos_only':
                msg.p.pose.pose.position.x = x_pos + randn() * 10
                msg.p.pose.pose.position.y = y_pos + randn() * 10
                msg.t.twist.twist.linear.x = float('nan') 
                msg.t.twist.twist.linear.y = float('nan')
                x_pos += 1
                y_pos += 1

            elif args.detect_fields == 'vel_only':
                msg.p.pose.pose.position.x = float('nan') 
                msg.p.pose.pose.position.y = float('nan') 
                msg.t.twist.twist.linear.x = x_vel + randn() * 10 
                msg.t.twist.twist.linear.y = x_vel + randn() * 10 
                x_vel += 1
                y_vel += 1
            
            # The following are for validation purposes only
            elif args.detect_fields == 'x_pos_no_y_pos':
                msg.p.pose.pose.position.x = x_pos + randn() * 10 
                msg.p.pose.pose.position.y = float('nan') 

            elif args.detect_fields == 'y_pos_no_x_pos':
                msg.p.pose.pose.position.x = float('nan') 
                msg.p.pose.pose.position.y = y_pos + randn() * 10 
            
            elif args.detect_fields == 'x_vel_no_y_vel':
                msg.t.twist.twist.linear.x = x_vel + randn() * 10 
                msg.t.twist.twist.linear.y = float('nan') 
            
            elif args.detect_fields == 'y_vel_no_x_vel':
                msg.t.twist.twist.linear.x = float('nan')
                msg.t.twist.twist.linear.y = y_vel + randn() * 10 
                 
            self.pub.publish(msg)
            rospy.sleep(d)


def main():
    
    arg_parser = argparse.ArgumentParser(description='TBD')
    arg_parser.add_argument('detect_fields', type=str, choices=['pos_only', 'vel_only', 'both', 'x_pos_no_y_pos', 'y_pos_no_x_pos', 'x_vel_no_y_vel', 'y_vel_no_x_vel'], help='type of data to be sent with the Detect messages')
    args = arg_parser.parse_args()

    rospy.init_node('dummy')
    try:
        d = Dummy(args)
    except:
        rospy.ROSInterruptException
        pass


if __name__=='__main__':
    main()

