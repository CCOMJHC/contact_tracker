#!/usr/bin/env python

# Generates dummy data for initial testing of tracker.py

import rospy
import argparse
import sys
from numpy.random import randn

from marine_msgs.msg import Detect

class Dummy():
    
    def __init__(self, args):
        
        print('initializing...')
        self.x_pos = args.xpos
        self.y_pos = args.ypos 
        self.x_vel = args.xvel
        self.y_vel = args.yvel
        self.direction = args.direction
        self.niter = 0
        self.step = 1 

    def move(self):
        """
        Move the object in the direction specified by the user 
        from the command line.
        """
        
        if self.direction == 'ne':
            self.x_pos += self.step + randn()
            self.y_pos += self.step + randn()
 
        elif self.direction == 'nw':
            self.x_pos -= self.step + randn()
            self.y_pos += self.step + randn()
             
        elif self.direction == 'se':
            self.x_pos += self.step + randn()
            self.y_pos -= self.step + randn()
  
        elif self.direction == 'sw':
            self.x_pos -= self.step + randn()
            self.y_pos -= self.step + randn()
         

    def run(self):
        
        self.pub = rospy.Publisher('/detects', Detect, queue_size=1)
        
        while self.niter < 500:
            d = rospy.Duration(randn())
            msg = Detect()
            msg.header.stamp = rospy.get_rostime()
            coin_flip = 1
           
            # Generate message with position and velocity
            msg.pose.covariance = [50, 0, 0, 0, 0, 0,
                                   0, 50, 0, 0, 0, 0,
                                   0, 0, 0, 0, 0, 0,
                                   0, 0, 0, 0, 0, 0,
                                   0, 0, 0, 0, 0, 0,
                                   0, 0, 0, 0, 0, 0]
            if coin_flip > 0:    
                self.move()
                msg.pose.pose.position.x = self.x_pos 
                msg.pose.pose.position.y = self.y_pos 
                msg.twist.twist.linear.x = 1.0
                msg.twist.twist.linear.y = 1.0
                print(msg.header.stamp, ': Generating message with position and velocity: ', msg.pose.pose.position.x)
            
            # Generate message with position and not velocity
            elif coin_flip <= 0 and coin_flip >= -1: 
                self.move()
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
            
            self.niter += 1
            self.pub.publish(msg)
            rospy.sleep(d)


def main():
    
    arg_parser = argparse.ArgumentParser(description='Send fake Detect data to the tracker node for testing purposes.')
    arg_parser.add_argument('-xpos', type=float, help='initial x position of the object')
    arg_parser.add_argument('-ypos', type=float, help='initial y position of the object')
    arg_parser.add_argument('-xvel', type=float, help='initial x velocity of the object')
    arg_parser.add_argument('-yvel', type=float, help='initial y velocity of the object')
    arg_parser.add_argument('-step', type=float, help='step to increment positions each iteration')
    arg_parser.add_argument('-direction', type=str, choices=['nw', 'ne', 'se', 'sw'], help='direction the simulated object should move')
    args = arg_parser.parse_args(rospy.myargv()[1:])

    rospy.init_node('dummy')
    try:
        simulation = Dummy(args)
        print('initialized')
        simulation.run()
    except:
        rospy.ROSInterruptException
        rospy.loginfo('Falied to initialize the simulation')
        pass


if __name__=='__main__':
    main()

