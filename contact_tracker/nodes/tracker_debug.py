#!/usr/bin/env python

# A contact is identifed by position, not id, and is
# independent of the sensor that produced it.

# Author: Rachel White
# University of New Hampshire
# Date last modified: 02/04/2020

import math
import time
import rospy
import datetime
import argparse
import numpy as np
import matplotlib.pyplot as plt

import contact_tracker.contact
from marine_msgs.msg import Detect

from filterpy.kalman import KalmanFilter
from filterpy.kalman import update
from filterpy.kalman import predict
from filterpy.stats.stats import plot_covariance

from dynamic_reconfigure.server import Server
from contact_tracker.cfg import contact_trackerConfig

DEBUG = True 

class ContactTracker:
    """
    Class to create custom Kalman filter.
    """


    def __init__(self):
        """
        Define the constructor.

        max_time -- amount of time that must ellapse before an item is deleted from all_contacts
        dt -- time step for the Kalman filters
        initial_velocity -- velocity at the start of the program
        """

        self.all_contacts = {}


    def plot_x_vs_y(self, output_path):
        """
        Visualize results of the Kalman filter by plotting the measurements against the 
        predictions of the Kalman filter.

        Keyword arguments:
        output_path -- The path that the plot will be saved to
        """
        
        all_mxs = []
        all_mys = []
        all_pxs = []
        all_pys = []

        for contact in self.all_contacts:
            c = self.all_contacts[contact]
        
            m_xs = []
            m_ys = []
            p_xs = []
            p_ys = []

            for i in c.zs:
                m_xs.append(i[0])
                m_ys.append(i[1])
        
            for i in c.xs:
                p_xs.append(i[0])
                p_ys.append(i[1])

            all_mxs.append(m_xs)
            all_mys.append(m_ys)
            all_pxs.append(p_xs)
            all_pys.append(p_ys)

        for i in range(0, len(all_mxs)):
            plt.scatter(all_mxs[i], all_mys[i], linestyle='-', label='kf ' + str(i) + ' measurements', color='y')
            plt.plot(all_pxs[i], all_pys[i], label='kf ' + str (i) + ' predictions')

        plt.legend()
        plt.xlabel('x position')
        plt.ylabel('y position')
        plt.xlim(0, 500)
        plt.ylim(0, 500)
        plt.savefig(output_path + '.png')


    def plot_x_vs_time(self, output_path):
        """
        Visualize results of the Kalman filter by plotting the measurements against the 
        predictions of the Kalman filter.
        
        Keyword arguments:
        output_path -- The path that the plot will be saved to
        """

        all_mxs = []
        all_pxs = []
        all_times = []

        for contact in self.all_contacts:
            c = self.all_contacts[contact]
         
            m_xs = []
            p_xs = []

            for i in c.zs:
                m_xs.append(i[0])
        
            for i in c.xs:
                p_xs.append(i[0])

            all_mxs.append(m_xs)
            all_pxs.append(p_xs)
            all_times.append(c.times)

        for i in range(0, len(all_mxs)):
            plt.scatter(all_times[i], all_mxs[i], linestyle='-', label='kf ' + str(i) + ' measurements', color='y')
            plt.plot(all_times[i], all_pxs[i], label='kf ' + str (i) + ' predictions')

        plt.legend()
        plt.xlabel('time')
        plt.ylabel('x position')
        plt.ylim(0, 300)
        plt.savefig(output_path + '.png')


    def plot_ellipses(self, output_path):
        """
        Visualize results of the Kalman filter by plotting the measurements against the 
        predictions of the Kalman filter.
        
        Keyword arguments:
        output_path -- The path that the plot will be saved to
        """

        all_pxs = []
        all_pys = []
        all_zs = []
        all_ps = []

        for contact in self.all_contacts:
            c = self.all_contacts[contact]
            p_xs = []
            p_ys = []
            cur_ps = []
            z_means = []

            for i in c.xs:
                p_xs.append(i[0])
                p_ys.append(i[1])

            all_pxs.append(p_xs)
            all_pys.append(p_ys)

            for i in range(0, len(c.xs), 4):
                z_mean = np.array([c.zs[i][0], c.zs[i][1]])
                cur_p = c.ps[i]
                plot_covariance(mean=z_mean, cov=cur_p)

            all_zs.append(z_means)
            all_ps.append(cur_ps)

            
        for i in range(0, len(all_pxs)):
            plt.plot(all_pxs[i], all_pys[i], label='predictions', color='g')

        plt.xlabel('x position')
        plt.ylabel('y position')
        plt.xlim(0, 300)
        plt.ylim(0, 300)
        plt.legend()
        plt.savefig(output_path + '.png')
        
    
    def dump_detect(self, detect_info):
        """
        Print the contents of a contact's detect_info dictionary for debugging purposes.

        Keyword arguments:
        detect_info -- the dictionary containing the detect info to be printed
        """
        
        print('+++++++ DETECT +++++++')
        for k, v in detect_info.items():
            print(k, ': ', v)


    def dump_contacts(self):
        """
        Print the contents of the all_contacts  dictionary for debugging purposes.
        """
        
        print('+++++++ CONTACTS +++++++')
        for k, v in self.all_contacts.items():
            print(k, ': ', v)


    def populate_detect_info(self, data):
        """
        Initialize the data structure for this detect message and check 
        for empty position and velocity fields. Return an empty dictionary 
        if one position or velocity field is empty and the other position
        or velocity field is not.

        Keyword arguments:
        data -- The detect message that was just transmitted
        """
        
        # Get necessary info from the Detect data
        detect_info = {
                'header': data.header,
                'sensor_id': data.sensor_id,
                'pos_covar': data.pose.covariance,
                'twist_covar': data.twist.covariance,
                'x_pos': float('nan'),
                'x_vel': float('nan'),
                'y_pos': float('nan'),
                'y_vel': float('nan'),
                'z_pos': float('nan'),
                'z_vel': float('nan')
                }
        
        # Assign values only if they are not NaNs
        if not math.isnan(data.pose.pose.position.x):
            detect_info['x_pos'] = float(data.pose.pose.position.x)

        if not math.isnan(data.pose.pose.position.y):
            detect_info['y_pos'] = float(data.pose.pose.position.y)

        if not math.isnan(data.pose.pose.position.z):
            detect_info['z_pos'] = float(data.pose.pose.position.z)

        if not math.isnan(data.twist.twist.linear.x):
            detect_info['x_vel'] = float(data.twist.twist.linear.x)

        if not math.isnan(data.twist.twist.linear.y):
            detect_info['y_vel'] = float(data.twist.twist.linear.y)

        if not math.isnan(data.twist.twist.linear.z):
            detect_info['z_vel'] = float(data.twist.twist.linear.z)


        # Check to see that if one coordinate is not NaN, neither is the other
        if ((not math.isnan(detect_info['x_pos']) and math.isnan(detect_info['y_pos'])) or (math.isnan(detect_info['x_pos']) and not math.isnan(detect_info['y_pos']))):
            if DEBUG: print('ERROR: x_pos and y_pos both were not nans...returning')
            return {} 
        if ((not math.isnan(detect_info['x_vel']) and math.isnan(detect_info['y_vel'])) or (math.isnan(detect_info['x_vel']) and not math.isnan(detect_info['y_vel']))):
            if DEBUG: print('ERROR: x_vel and y_vel both were not nans...returning')
            return {}

        return detect_info


    def check_all_contacts(self, detect_info, new_stamp):
        """
        Iterate over every contact in the dictionary and return contact the current Detect is 
        most likely associated with. Otherwise, return the timestamp of the current Detect message
        as the new hash_key for the new Contact that will be made.

        Keyword arguments:
        detect_info -- the dictionary containing the detect info to be checked
        new_stamp -- the timestamp from the header of the contact we're checking all others against
        """
 
        for contact in self.all_contacts:
            c = self.all_contacts[contact]

            '''if DEBUG:
                print('x measurement: ', detect_info['x_pos'])
                print('x prediction: ', c.kf.x[0])
                print('y measurement: ', detect_info['y_pos'])
                print('y prediction: ', c.kf.x[1])'''
                       
            # measurement - prediction < uncertainty_in_measurement + uncertainty_in_prediction
            for kf in c.filter_bank.filters:
                if ((abs(detect_info['x_pos'] - kf.x[0]) < (detect_info['pos_covar'][0] + kf.R[0][0])) and 
                    (abs(detect_info['y_pos'] - kf.x[1]) < (detect_info['pos_covar'][7] + kf.R[1][1]))):
                     return c.id
                 
        # No appropriate contacts were found, so return the stamp of the Detect message being checked
        return new_stamp

        
    def delete_stale_contacts(self):
        """
        Remove items from the dictionary that have not been measured in a while
        """
        
        for contact_id in self.all_contacts:
            cur_contact = self.all_contacts[contact_id]
            time_between_now_and_last_measured = (rospy.get_rostime() - cur_contact.last_measured).to_sec()

            if time_between_now_and_last_measured > self.max_stale_contact_time:
                if DEBUG: print('deleting stale Contact from dictionary')
                del self.all_contacts[cur_contact]


    def reconfigure_callback(self, config, level):
        """
        Get the parameters from the cfg file and assign them to the member variables of the 
        KalmanTracker class.
        """

        self.max_stale_contact_time = config['max_stale_contact_time']
        self.initial_velocity = config['initial_velocity']
        self.variance = config['variance']
        self.Q_vars = {
                'q0_xy': config['q0_xy'],
                'q0_v': config['q0_v'],
                'q0_a': config['q0_a'],
                'dq_xy': config['dq_xy'],
                'dq_v': config['dq_v'],
                'dq_a': config['dq_a'],
        }

        return config


    def callback(self, data):
        """
        Listen for detects and add to dictionary and filter if not already there.

        Keyword arguments:
        data -- the Detect message transmitted
        """

        ########################################################
        ###### VARIABLE INITIALIZATION AND ERROR HANDLING ######
        ########################################################
         
        # Initialize variables and store in a dictionary.
        detect_info = self.populate_detect_info(data)
        if len(detect_info) == 0: 
            return

        #  Compare new measurement to prediction at same time.
        #  If there are no contacts yet, no need to traverse empty dictionary
        #  Otherwise, we have to check each contact in the dictionary to see if 
        #  it is a potential match for our current Detect message. 
        if len(self.all_contacts) == 0:
            contact_id  = data.header.stamp 
        else:
            contact_id = self.check_all_contacts(detect_info, data.header.stamp)
            

        #######################################################
        ####### CREATE OR UPDATE CONTACT WITH VARIABLES #######
        #######################################################
        
        epoch = 0
        if not contact_id in self.all_contacts: 
          
            first_order_kf = KalmanFilter(dim_x=6, dim_z=6)
            second_order_kf = KalmanFilter(dim_x=6, dim_z=6)
            filter_bank = [first_order_kf, second_order_kf]
            c = contact_tracker.contact.Contact(detect_info, filter_bank, self.variance, data.header.stamp)
            
            if not math.isnan(detect_info['x_pos']) and math.isnan(detect_info['x_vel']):
                rospy.loginfo('Instantiating first-order Kalman filter with position but without velocity')
                c.init_kf_with_position_only()
            
            elif math.isnan(detect_info['x_pos']) and not math.isnan(detect_info['x_vel']):
                rospy.loginfo('Instantiating first-order Kalman filter with velocity but without position')
                c.init_kf_with_velocity_only()
            
            elif not math.isnan(detect_info['x_pos']) and not math.isnan(detect_info['x_vel']):
                rospy.loginfo('Instantiating first-order Kalman filter with velocity and position')
                c.init_kf_with_position_and_velocity()
            
            self.all_contacts[data.header.stamp] = c

        else:
            # Recompute the value for dt, and use it to update this Contact's KalmanFilter's Q(s).
            # Then update the time stamp for when this contact was last measured so we know not
            # to remove it anytime soon. 
            c = self.all_contacts[contact_id]
            c.last_measured = data.header.stamp
            epoch = (c.last_measured - c.first_measured).to_sec()
            
            c.dt = epoch
            for kf in c.filter_bank.filters:
                kf.Q = c.recompute_q(self.Q_vars) 
            c.info = detect_info

            if not math.isnan(detect_info['x_pos']):
                c.last_xpos = detect_info['x_pos']
                c.last_ypos = detect_info['y_pos']
           
            if not math.isnan(detect_info['x_vel']):
                c.last_xvel = detect_info['x_vel']
                c.last_yvel = detect_info['y_vel']
        
        # Incorporate with filters in the filter_bank 
        c = self.all_contacts[contact_id]
        c.filter_bank.predict()
        
        if math.isnan(detect_info['x_pos']):
            c.filter_bank.update((c.last_xpos, c.last_ypos, c.info['x_vel'], c.info['y_vel'], 0, 0))
        elif math.isnan(detect_info['x_vel']):
            c.filter_bank.update((c.info['x_pos'], c.info['y_pos'], c.last_xvel, c.last_yvel, 0, 0))
        else:
            c.filter_bank.update((c.info['x_pos'], c.info['y_pos'], c.info['x_vel'], c.info['y_vel'], 0, 0))
        
        # Append appropriate prior and measurements to lists here
        c.xs.append(c.filter_bank.x)
        c.zs.append(c.filter_bank.z)
        c.ps.append(c.filter_bank.P)
        c.times.append(epoch)

        # Remove items from the dictionary that have not been measured in a while
        self.delete_stale_contacts()


    def run(self, args):
        """
        Initialize the node and set it to subscribe to the detects topic.
        """

        rospy.init_node('tracker', anonymous=True)
        srv = Server(contact_trackerConfig, self.reconfigure_callback)
        rospy.Subscriber('/detects', Detect, self.callback)
        rospy.spin()
        
        if args.plot_type == 'xs_ys':
            self.plot_x_vs_y(args.o)
        elif args.plot_type =='xs_times':
            self.plot_x_vs_time(args.o)
        elif args.plot_type == 'ellipses':
            self.plot_ellipses(args.o)


def main():
    
    arg_parser = argparse.ArgumentParser(description='Track contacts by applying Kalman filters to incoming detect messages. Optionally plot the results of the filter.')
    arg_parser.add_argument('-plot_type', type=str, choices=['xs_ys', 'xs_times', 'ellipses'], help='specify the type of plot to produce, if you want one')
    arg_parser.add_argument('-o', type=str, help='path to save the plot produced, default: tracker_plot, current working directory', default='tracker_plot')
    args = arg_parser.parse_args()

    try:
        ct = ContactTracker()
        ct.run(args)

    except rospy.ROSInterruptException:
        rospy.loginfo('Falied to initialize the ContactTracker')
        pass


if __name__=='__main__':
    main()


