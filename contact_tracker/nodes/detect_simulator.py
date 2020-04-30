#!/usr/bin/env python

# Generates simulated Detect messages for initial testing of tracker.py

# Author: Rachel White
# University of New Hampshire
# Date last modified: 03/20/2020

import rospy
import argparse
import sys
from numpy import nan
from numpy.random import randn
import numpy as np
import matplotlib.pyplot as plt

from marine_msgs.msg import Detect


class DetectSimulator():

    def __init__(self, args):

        self.x_pos = args.xpos
        self.y_pos = args.ypos
        self.speed = args.speed
        self.pub = 1
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
        minx = np.min(np.array(self.xs))
        maxx = np.max(np.array(self.xs))
        miny = np.min(np.array(self.ys))
        maxy = np.max(np.array(self.ys))
        plt.xlim(minx - 20, maxx + 20)
        plt.ylim(miny - 20, maxy + 20)
        plt.grid(True)
        plt.savefig(output_path + '.png')
        plt.close()

    def move(self):
        """
        Move the object in the direction specified by the user
        from the command line.
        """
        
        print ("Move Direction: %s" % self.direction)


        if self.direction == 'n':
            self.x_vel = randn()
            self.y_vel = self.speed + 0.5 * randn()

        elif self.direction == 's':
            self.x_vel = randn()
            self.y_vel = -self.speed + 0.5 *randn()

        elif self.direction == 'e':
            self.x_vel = self.speed + 0.5 *randn()
            self.y_vel = randn()

        elif self.direction == 'w':
            self.x_vel = -self.speed + 0.5 *randn()
            self.y_vel = randn()

        if self.direction == 'ne':
            self.x_vel = self.speed / np.sqrt(2) + 0.5 * randn()
            self.y_vel = self.speed / np.sqrt(2) + 0.5 * randn()

        elif self.direction == 'nw':
            self.x_vel = -self.speed / np.sqrt(2) + 0.5 *randn()
            self.y_vel = self.speed / np.sqrt(2) + 0.5 * randn()

        elif self.direction == 'se':
            self.x_vel = self.speed / np.sqrt(2) + 0.5 * randn()
            self.y_vel = -self.speed / np.sqrt(2) + 0.5 * randn()

        elif self.direction == 'sw':
            self.x_vel = -self.speed / np.sqrt(2) + 0.5 * randn()
            self.y_vel = -self.speed / np.sqrt(2) + 0.5 * randn()
        
        elif self.direction == 'nne':
            self.x_vel = self.speed * np.cos(67.5*np.pi/180.) + 0.5 * randn()
            self.y_vel = self.speed * np.sin(67.5*np.pi/180.) + 0.5 * randn()
        
        elif self.direction == 'ene':
            self.x_vel = self.speed * np.cos(22.5*np.pi/180.) + 0.5 * randn()
            self.y_vel = self.speed * np.sin(22.5*np.pi/180.) + 0.5 * randn()
        
        elif self.direction == 'ese':
            self.x_vel = self.speed * np.cos(337.5*np.pi/180.) + 0.5 * randn()
            self.y_vel = self.speed * np.sin(337.5*np.pi/180.) + 0.5 * randn()

        elif self.direction == 'sse':
            self.x_vel = self.speed * np.cos(292.5*np.pi/180.) + 0.5 * randn()
            self.y_vel = self.speed * np.sin(292.5*np.pi/180.) + 0.5 * randn()

        elif self.direction == 'ssw':
            self.x_vel = self.speed * np.cos(247.5*np.pi/180.) + 0.5 * randn()
            self.y_vel = self.speed * np.sin(247.5*np.pi/180.) + 0.5 * randn()

        elif self.direction == 'wsw':
            self.x_vel = self.speed * np.cos(202.5*np.pi/180.) + 0.5 * randn()
            self.y_vel = self.speed * np.sin(202.5*np.pi/180.) + 0.5 * randn()

        elif self.direction == 'wnw':
            self.x_vel = self.speed * np.cos(157.5*np.pi/180.) + 0.5 * randn()
            self.y_vel = self.speed * np.sin(157.5*np.pi/180.) + 0.5 * randn()

        elif self.direction == 'nnw':
            self.x_vel = self.speed * np.cos(112.5*np.pi/180.) + 0.5 * randn()
            self.y_vel = self.speed * np.sin(112.5*np.pi/180.) + 0.5 * randn()


        self.x_pos += self.x_vel * self.dt
        self.y_pos += self.y_vel * self.dt


        self.xs.append(self.x_pos)
        self.ys.append(self.y_pos)


    def slow_turn_right(self):
        
        print("Slow Turn Right.")
   
        if self.direction == 'n':
            self.direction = 'nnw'

        elif self.direction == 'nnw':
            self.direction = 'nw'

        elif self.direction == 'nw':
            self.direction = 'wnw'

        elif self.direction == 'wnw':
            self.direction = 'w'

        elif self.direction == 'w':
            self.direction = 'wsw'
            
        elif self.direction == 'wsw':
            self.direction = 'sw'

        elif self.direction == 'sw':
            self.direction = 'ssw'

        elif self.direction == 'ssw':
            self.direction = 's'

        elif self.direction == 's':
            self.direction = 'sse'

        elif self.direction == 'sse':
            self.direction = 'se'

        elif self.direction == 'se':
            self.direction = 'ese'

        elif self.direction == 'ese':
            self.direction = 'e'

        elif self.direction == 'e':
            self.direction = 'ene'

        elif self.direction == 'ene':
            self.direction = 'ne'

        elif self.direction == 'ne':
            self.direction = 'nne'

        elif self.direction == 'nne':
            self.direction = 'n'
            
        self.x_pos += self.x_vel * self.dt 
        self.y_pos += self.y_vel * self.dt


        self.xs.append(self.x_pos)
        self.ys.append(self.y_pos)

    def slow_turn_left(self):

        print("Slow Turn Left.")

        
        if self.direction == 'n':
            self.direction = 'nne'

        elif self.direction == 'nne':
            self.direction = 'ne'

        elif self.direction == 'ne':
            self.direction = 'ene'

        elif self.direction == 'ene':
            self.direction = 'e'

        elif self.direction == 'e':
            self.direction = 'ese'
            
        elif self.direction == 'ese':
            self.direction = 'se'

        elif self.direction == 'se':
            self.direction = 'sse'

        elif self.direction == 'sse':
            self.direction = 's'

        elif self.direction == 's':
            self.direction = 'ssw'

        elif self.direction == 'ssw':
            self.direction = 'sw'

        elif self.direction == 'sw':
            self.direction = 'wsw'

        elif self.direction == 'wsw':
            self.direction = 'w'

        elif self.direction == 'w':
            self.direction = 'wnw'

        elif self.direction == 'wnw':
            self.direction = 'nw'

        elif self.direction == 'nw':
            self.direction = 'nnw'

        elif self.direction == 'nnw':
            self.direction = 'n'

        self.x_pos += self.x_vel * self.dt 
        self.y_pos += self.y_vel * self.dt


        self.xs.append(self.x_pos)
        self.ys.append(self.y_pos) 
        
    
    
    def turn_right(self):
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

    def turn_left(self):
        """
        Turn the object 90 degrees clockwise from its
        initial direction.
        """
        print("Turning left.")
        if self.direction == 'n':
            self.direction = 'w'

        elif self.direction == 's':
            self.direction = 'e'

        elif self.direction == 'e':
            self.direction = 'n'

        elif self.direction == 'w':
            self.direction = 's'

        if self.direction == 'ne':
            self.direction = 'nw'

        elif self.direction == 'nw':
            self.direction = 'sw'

        elif self.direction == 'se':
            self.direction = 'ne'

        elif self.direction == 'sw':
            self.direction = 'se'

    def run(self):
        
        self.pub_detects = rospy.Publisher('/detects', Detect, queue_size=1)
        
        self.slow_turn_blinker = 'off' # allows 90 deg turns spaced over 2 interations.
        

        while self.niter < 500 and not rospy.is_shutdown():
            self.dt = 1
            d = rospy.Duration(self.dt)
            msg = Detect()
            msg.header.stamp = rospy.get_rostime()
            coin_flip = 1
            msg.sensor_id = self.name
            msg.header.frame_id = "map"
            msg.pose.covariance = [3., 0., nan, nan, nan, nan,
                                   0., 3., nan, nan, nan, nan,
                                   nan, nan, nan, nan, nan, nan,
                                   nan, nan, nan, nan, nan, nan,
                                   nan, nan, nan, nan, nan, nan,
                                   nan, nan, nan, nan, nan, nan]
            msg.twist.covariance = [1.0, 0., nan, nan, nan, nan,
                                   0., 1.0, nan, nan, nan, nan,
                                   nan, nan, nan, nan, nan, nan,
                                   nan, nan, nan, nan, nan, nan,
                                   nan, nan, nan, nan, nan, nan,
                                   nan, nan, nan, nan, nan, nan]
            
            # Generate message with position and velocity
            if coin_flip > 0:
                if self.direction != 'none':
                    self.move()
                 
                # This mess allows the simulator to execute 45 degree turns 
                # in 2 22.5 degree steps to slow the turn rate and (hopefully)
                # make more realistically moving targets. 
                if self.slow_turn_blinker == 'right':
                    self.slow_turn_right()
                    self.slow_turn_blinker = 'off'                    

                if self.slow_turn_blinker == 'left':
                    self.slow_turn_left()
                    self.slow_turn_blinker = 'off'
                    

                if self.niter % 30 == 0:

                    if np.random.randn() >= 0 and self.slow_turn_blinker == 'off':                    
                        self.slow_turn_blinker = 'right'
                        self.slow_turn_right()
                    elif self.slow_turn_blinker == 'off':
                        self.slow_turn_blinker = 'left'
                        self.slow_turn_left()
                    
                    '''
                    if np.random.randn() >= 0.:
                        self.turn_right()
                    else:
                        self.turn_left()
                    '''

                msg.pose.pose.position.x = self.x_pos
                msg.pose.pose.position.y = self.y_pos
                msg.twist.twist.linear.x = self.x_vel
                msg.twist.twist.linear.y = self.y_vel

                print("%d, %i,: Msg: [x:%0.3f, y:%0.3f, vx: %0.3f, vy: %0.3f]" %
                      (self.niter, msg.header.stamp.secs,
                       msg.pose.pose.position.x,
                       msg.pose.pose.position.y,
                       msg.twist.twist.linear.x,
                       msg.twist.twist.linear.y))
            
            # Generate message with position and not velocity
            elif coin_flip < 0 and coin_flip >= -1:
                if self.direction != 'none':
                    self.move()

                if self.niter % 100 == 0:
                    self.turn()

                msg.twist.twist.linear.x = float('nan')
                msg.twist.twist.linear.y = float('nan')

            # Generate message with velocity and not position
            elif coin_flip == 0:
                msg.pose.pose.position.x = float('nan')
                msg.pose.pose.position.y = float('nan')
                msg.twist.twist.linear.x = self.x_vel + randn()
                msg.twist.twist.linear.y = self.x_vel + randn()


            ######################
            # 2) Publish message #
            ######################
            self.niter += 1
            if self.return_enabled:
                raw_input()

            self.pub_detects.publish(msg)
            rospy.sleep(d)


def main():

    arg_parser = argparse.ArgumentParser(description='Send fake Detect data to the tracker node for testing purposes.')
    arg_parser.add_argument('-xpos', type=float, help='initial x position of the object')
    arg_parser.add_argument('-ypos', type=float, help='initial y position of the object')
    arg_parser.add_argument('-speed', type=float, help='nominal speed of the object')
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


if __name__=='__main__':
    main()
