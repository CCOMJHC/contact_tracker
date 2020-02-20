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
from numpy import zeros
import matplotlib.pyplot as plt

import contact_tracker.contact
import contact_tracker.extended_kf
from marine_msgs.msg import Detect

from filterpy.kalman import IMMEstimator
from filterpy.kalman import KalmanFilter
from filterpy.kalman import update
from filterpy.kalman import predict
from filterpy.stats.stats import plot_covariance

from dynamic_reconfigure.server import Server
from contact_tracker.cfg import contact_trackerConfig

DEBUG = True 

class ContactTracker:
    """
    Class to create custom contact tracker.
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
            plt.scatter(all_mxs[i], all_mys[i], linestyle='-', label='contact' + str(i) + ' measurements', color='y')
            plt.plot(all_pxs[i], all_pys[i], label='contact' + str (i) + ' predictions')

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
        for k in self.all_contacts.items():
            print(k)
        print('++++++++++++++++++++++++')


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
                }
        
        # Assign values only if they are not NaNs
        if not math.isnan(data.pose.pose.position.x):
            detect_info['x_pos'] = float(data.pose.pose.position.x)

        if not math.isnan(data.pose.pose.position.y):
            detect_info['y_pos'] = float(data.pose.pose.position.y)

        if not math.isnan(data.twist.twist.linear.x):
            detect_info['x_vel'] = float(data.twist.twist.linear.x)

        if not math.isnan(data.twist.twist.linear.y):
            detect_info['y_vel'] = float(data.twist.twist.linear.y)

        # Check to see that if one coordinate is not NaN, neither is the other
        if ((not math.isnan(detect_info['x_pos']) and math.isnan(detect_info['y_pos'])) or (math.isnan(detect_info['x_pos']) and not math.isnan(detect_info['y_pos']))):
            rospy.loginfo('ERROR: x_pos and y_pos both were not nans...returning')
            detect_info = {} 
        if ((not math.isnan(detect_info['x_vel']) and math.isnan(detect_info['y_vel'])) or (math.isnan(detect_info['x_vel']) and not math.isnan(detect_info['y_vel']))):
            rospy.loginfo('ERROR: x_vel and y_vel both were not nans...returning')
            detect_info = {}
        
        return detect_info


    def check_all_contacts(self, detect_info, data):
        """
        Iterate over every contact in the dictionary and return contact the current Detect is 
        most likely associated with. Otherwise, return the timestamp of the current Detect message
        as the new hash_key for the new Contact that will be made.

        Keyword arguments:
        detect_info -- the dictionary containing the detect info to be checked

        Returns:
        None if no appropriate contacts are found, otherwise the found contact's id
        """
        
        greatest_logBF = 0
        return_contact_id = None 

        #print('___________________________')
        #print('x position: ', detect_info['x_pos'])
        
        for contact in self.all_contacts:
            # Recompute the value for dt, and use it to update this Contact's KalmanFilter's Q(s).
            # Then update the time stamp for when this contact was last measured so we know not
            # to remove it anytime soon. Updating Q is also a prerequsite for calculating a contact's
            # Bayes factor.
            c = self.all_contacts[contact]
            c.last_measured = data.header.stamp
            epoch = (c.last_measured - c.first_measured).to_sec()
            c.dt = epoch
            
            for kf in c.filter_bank.filters:
                kf.set_Q(c) 
                kf.set_F(c)
                kf.set_H(c, detect_info)
                kf.set_R(c, detect_info) 
                kf.set_Z(c, detect_info)
                kf.set_bayes_factor(2)
            
            logBF1 = c.filter_bank.filters[0].get_bayes_factor()
            logBF2 = c.filter_bank.filters[1].get_bayes_factor()

            if logBF1 > 2 and logBF2 > 2: 
                if logBF1 + logBF2 > greatest_logBF:
                    greatest_logBF = logBF1 + logBF2
                    return_contact_id = c.id

        #self.dump_contacts()
        return return_contact_id 
 
        
    def delete_stale_contacts(self):
        """
        Remove items from the dictionary that have not been measured in a while
        """
        
        for contact_id in self.all_contacts:
            cur_contact = self.all_contacts[contact_id]
            time_between_now_and_last_measured = (rospy.get_rostime() - cur_contact.last_measured).to_sec()

            if time_between_now_and_last_measured > self.max_stale_contact_time:
                rospy.loginfo('deleting stale Contact from dictionary')
                del self.all_contacts[cur_contact]


    def reconfigure_callback(self, config, level):
        """
        Get the parameters from the cfg file and assign them to the member variables of the 
        KalmanTracker class.
        """

        self.max_stale_contact_time = config['max_stale_contact_time']
        self.initial_velocity = config['initial_velocity']
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
        contact_id = None 
        
        if len(self.all_contacts) > 0:
            contact_id = self.check_all_contacts(detect_info, data)
        
        if contact_id is None:
           contact_id = data.header.stamp 
            

        #######################################################
        ####### CREATE OR UPDATE CONTACT WITH VARIABLES #######
        #######################################################
        
        epoch = 0
        
        if not contact_id in self.all_contacts: 
            first_order_kf = contact_tracker.extended_kf.ExtendedKalmanFilter(dim_x=6, dim_z=4, filter_order='first')
            second_order_kf = contact_tracker.extended_kf.ExtendedKalmanFilter(dim_x=6, dim_z=4, filter_order='second')
            all_filters = [first_order_kf, second_order_kf]
            c = contact_tracker.contact.Contact(detect_info, all_filters, data.header.stamp)
            c.init_filters()
            c.filter_bank = IMMEstimator(all_filters, c.mu, c.M)
            self.all_contacts[data.header.stamp] = c

        else:
            c = self.all_contacts[contact_id]
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
            c.filter_bank.update(c.Z)

        # Append appropriate prior and measurements to lists here
        c.xs.append(np.array([c.filter_bank.x[0], c.filter_bank.x[1]]))
        c.zs.append(np.array([c.info['x_pos'], c.info['y_pos']]))
        c.ps.append(c.filter_bank.P)
        c.times.append(epoch)

        # Remove items from the dictionary that have not been measured in a while
        #self.delete_stale_contacts()


    def run(self, args):
        """
        Initialize the node and set it to subscribe to the detects topic.
        """

        rospy.init_node('tracker_debug', anonymous=True)
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


