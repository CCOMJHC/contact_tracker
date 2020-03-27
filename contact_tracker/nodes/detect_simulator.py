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
import matplotlib.pyplot as plt

from marine_msgs.msg import Detect, Contact

from project11_transformations.srv import MapToLatLong

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
        
        self.pub_detects = rospy.Publisher('/detects', Detect, queue_size=1)
        self.pub_contactmap = rospy.Publisher('/contact_map', Detect, queue_size=1)        
        self.pub_contacts = rospy.Publisher('/contacts', Contact, queue_size=1)        
        
        while self.niter < 500 and not rospy.is_shutdown():
            d = rospy.Duration(1)
            coin_flip = 1
            
            #####################################
            # Set fields for the Detect message #
            #####################################
            detect_msg = Detect()
            detect_msg.header.stamp = rospy.get_rostime()
            detect_msg.sensor_id = self.name
            detect_msg.header.frame_id = "map"
            detect_msg.pose.covariance = [10., 0., nan, nan, nan, nan,
                                          0., 10., nan, nan, nan, nan,
                                          nan, nan, nan, nan, nan, nan,
                                          nan, nan, nan, nan, nan, nan,
                                          nan, nan, nan, nan, nan, nan,
                                          nan, nan, nan, nan, nan, nan]
            detect_msg.twist.covariance = [1.0, 0., nan, nan, nan, nan,
                                           0., 1.0, nan, nan, nan, nan,
                                           nan, nan, nan, nan, nan, nan,
                                           nan, nan, nan, nan, nan, nan,
                                           nan, nan, nan, nan, nan, nan,
                                           nan, nan, nan, nan, nan, nan]
            
            # Generate message with position and velocity
            if coin_flip > 0:    
                if self.direction != 'none':
                    self.move()
                 
                if self.niter % 250 == 0:
                    self.turn()

                detect_msg.pose.pose.position.x = self.x_pos 
                detect_msg.pose.pose.position.y = self.y_pos 
                detect_msg.twist.twist.linear.x = 1.0
                detect_msg.twist.twist.linear.y = 1.0
                print("msg(x,y): ", detect_msg.pose.pose.position.x, detect_msg.pose.pose.position.y)
            
            # Generate message with position and not velocity
            elif coin_flip < 0 and coin_flip >= -1: 
                if self.direction != 'none':
                    self.move()
                
                if self.niter % 100 == 0:
                    self.turn()

                detect_msg.twist.twist.linear.x = float('nan') 
                detect_msg.twist.twist.linear.y = float('nan')

            # Generate message with velocity and not position
            elif coin_flip == 0:
                detect_msg.pose.pose.position.x = float('nan') 
                detect_msg.pose.pose.position.y = float('nan') 
                detect_msg.twist.twist.linear.x = self.x_vel + randn() 
                detect_msg.twist.twist.linear.y = self.x_vel + randn() 
            

            #########################################
            # 2) Set fields for the Contact message #
            #########################################
            contact_msg = Contact()
            #contact_msg.header.stamp = rospy.get_rostime()
            #contact_msg.header.frame_id = "map"
            
            # Do a service call to MaptoLatLong.srv to convert map coordinates to 
            # latitude and longitude.
            try:
                print('making a service call')
                rospy.wait_for_service('map_to_long')
                map2long_service = rospy.ServiceProxy('map_to_long', MapToLong)
                print('ServiceProxy made')
                
                map2long_req = MapToLongRequest()
                print('New request instantiated')
                map2long_req.map.point.x = self.x_pos
                map2long_req.map.point.y = self.y_pos

                llcords = map2long_service(map2long_req)
                print(llcoords)
                
            except rospy.ServiceException, e:
                print("Service call failed: %s", e)
            
            contact_msg.position.latitude = llcoords.wgs84.position.latitude
            contact_msg.position.longitude = llcoords.wgs84.position.longitude
 
            # Convert velocity in x and y into course over ground 
            # and speed over ground.
            contact_msg.cog = 0 
            contact_msg.sog = 0 
            contact_msg.heading = '' 

            # These fields are assigned arbitrary values for now. 
            contact_msg.mmsi = 0
            contact_msg.dimension_to_srbd = 0
            contact_msg.dimension_to_port = 0
            contact_msg.dimension_to_bow = 0
            contact_msg.dimension_to_stern = 0
 

            #####################################################           
            # 3) Publish both of the messages to the publishers #
            #####################################################
            self.niter += 1
            if self.return_enabled:
                raw_input()
            
            print('before pub')
            self.pub_detects.publish(detect_msg)
            self.pub_contactmap.publish(detect_msg)
            self.pub_contacts.publish(contact_msg)
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

