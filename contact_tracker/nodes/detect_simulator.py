#!/usr/bin/env python

# Generates simulated Detect messages for initial testing of tracker.py

# Author: Rachel White
# University of New Hampshire
# Date last modified: 03/20/2020

import rospy
import argparse
import sys
from numpy.random import randn
import matplotlib.pyplot as plt

from marine_msgs.msg import Detect

class DetectSimulator():
    
    def __init__(self, args):
        
        self.x_pos = args.xpos
        self.y_pos = args.ypos 
        self.x_vel = args.xvel
        self.y_vel = args.yvel
        self.direction = args.direction
        self.return_enabled = args.return_enabled
        self.niter = 1 
        self.step = 1 
        self.name = args.name
        self.xs = []
        self.ys = []


    def plot_course(self, output_path):
        """
        Plot the track that the Detect messages followed.
        For debugging purposes mainly.

        Keyword arguments:
        output_path -- path to save the plot produced
        """
        
        plt.plot(self.xs, self.ys, color='b')
        plt.xlabel('x position')
        plt.ylabel('y position')
        plt.xlim(0, 500)
        plt.ylim(0, 500)
        plt.savefig(output_path + '.png')
        plt.close()

    def move(self):
        """
        Move the object in the direction specified by the user 
        from the command line.
        """

        if self.direction == 'n':
            self.y_pos += self.step + randn()
 
        elif self.direction == 's':
            self.y_pos -= self.step + randn()
             
        elif self.direction == 'e':
            self.x_pos += self.step + randn()
  
        elif self.direction == 'w':
            self.x_pos -= self.step + randn()
        
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

        self.xs.append(self.x_pos)
        self.ys.append(self.y_pos)
         

    def turn(self):
        """
        Turn the object 90 degrees clockwise from its 
        initial direction.
        """

        if self.direction == 'n':
            self.direction = 'e'
 
        elif self.direction == 's':
            self.direction = 'w'
             
        elif self.direction == 'e':
            self.direction = 's'
  
        elif self.direction == 'w':
            self.direction = 'n'
 
        if self.direction == 'ne':
            self.direction = 'se'

        elif self.direction == 'nw':
            self.direction = 'ne'
             
        elif self.direction == 'se':
            self.direction = 'sw'
  
        elif self.direction == 'sw':
            self.direction = 'nw'
 

    def run(self):
        
        self.pub = rospy.Publisher('/detects', Detect, queue_size=1)
        
        while self.niter < 500 and not rospy.is_shutdown():
            d = rospy.Duration(1)
            msg = Detect()
            msg.header.stamp = rospy.get_rostime()
            coin_flip = 1
            msg.sensor_id = self.name
            msg.pose.covariance = [10., 0., 0., 0., 0., 0.,
                                   0., 10., 0., 0., 0., 0.,
                                   0., 0., 2., 0., 0., 0.,
                                   0., 0., 0., 2., 0, 0.,
                                   0., 0., 0., 0., .2, 0.,
                                   0., 0., 0., 0., 0., .2]
            
            # Generate message with position and velocity
            if coin_flip > 0:    
                if self.direction != 'none':
                    self.move()
                 
                if self.niter % 250 == 0:
                    self.turn()

                msg.pose.pose.position.x = self.x_pos 
                msg.pose.pose.position.y = self.y_pos 
                msg.twist.twist.linear.x = 1.0
                msg.twist.twist.linear.y = 1.0
                print(msg.header.stamp, ': Generating message with position and velocity: ', msg.pose.pose.position.x, msg.pose.pose.position.y)
            
            # Generate message with position and not velocity
            elif coin_flip < 0 and coin_flip >= -1: 
                if self.direction != 'none':
                    self.move()
                
                if self.niter % 100 == 0:
                    self.turn()

                msg.twist.twist.linear.x = float('nan') 
                msg.twist.twist.linear.y = float('nan')
                #print(msg.header.stamp, ': Generating message with position and not velocity: ', msg.pose.pose.position.x)

            # Generate message with velocity and not position
            elif coin_flip == 0:
                msg.pose.pose.position.x = float('nan') 
                msg.pose.pose.position.y = float('nan') 
                msg.twist.twist.linear.x = self.x_vel + randn() 
                msg.twist.twist.linear.y = self.x_vel + randn() 
                #print(msg.header.stamp, ': Generating message with velocity and not position: ', msg.pose.pose.position.x)
            
            self.niter += 1
            
            if self.return_enabled:
                raw_input()
            
            self.pub.publish(msg)
            rospy.sleep(d)


def main():
    
    arg_parser = argparse.ArgumentParser(description='Send fake Detect data to the tracker node for testing purposes.')
    arg_parser.add_argument('-xpos', type=float, help='initial x position of the object')
    arg_parser.add_argument('-ypos', type=float, help='initial y position of the object')
    arg_parser.add_argument('-xvel', type=float, help='initial x velocity of the object')
    arg_parser.add_argument('-yvel', type=float, help='initial y velocity of the object')
    arg_parser.add_argument('-step', type=float, help='step to increment positions each iteration')
    arg_parser.add_argument('-direction', type=str, choices=['n', 's', 'e', 'w', 'nw', 'ne', 'se', 'sw', 'none'], help='direction the simulated object should move')
    arg_parser.add_argument('-return_enabled', type=bool, help='generate Detect message only when the return key is pressed')
    arg_parser.add_argument('-show_plot', type=bool, help='plot the course the simulation followed')
    arg_parser.add_argument('-o', type=str, help='path to save the plot produced, default: tracker_plot, current working directory', default='sim_plot')
    arg_parser.add_argument('-name', type=str, help='identifier for this node')
    args = arg_parser.parse_args(rospy.myargv()[1:])

    rospy.init_node('detect_simulator')

    rospy.loginfo('Initializing the simulation')
    simulation = DetectSimulator(args)
    rospy.loginfo('Simulation was successfully initialized')
    rospy.loginfo('Beginning to run the simulation')
    simulation.run()
    rospy.loginfo('Simulation was successfully run')

    if args.show_plot == True:
        rospy.loginfo('Plotting the course of the simulation')
        simulation.plot_course(args.o)

    #except:
    #    rospy.ROSInterruptException
    #    rospy.loginfo('Falied to initialize the simulation')
    #    pass


if __name__=='__main__':
    main()

