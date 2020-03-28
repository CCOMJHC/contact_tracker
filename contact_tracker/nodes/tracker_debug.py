#!/usr/bin/env python

# A contact is identifed by position, not id, and is
# independent of the sensor that produced it.

# Author: Rachel White
# University of New Hampshire
# Date last modified: 02/20/2020

import math
import time
import rospy
import datetime
import argparse
import numpy as np
from numpy import zeros
import matplotlib.pyplot as plt

import contact_tracker.contact
import contact_tracker.contact_kf
from marine_msgs.msg import Detect, Contact
from project11_transformations.srv import MapToLatLong

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
        """

        self.all_contacts = {}
        self.pub_contactmap = None
        self.pub_contacts = None


    def plot_x_vs_y(self, output_path):
        """
        Plot the measurement's x and y positions 
        alongside the prediction's x and y positions.

        Keyword arguments:
        output_path -- path that the plot will be saved to
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
        Plot the measurement's x position at each time unit 
        alongside the prediction's x position at each time 
        unit.
        
        Keyword arguments:
        output_path -- path that the plot will be saved to
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
        Plot the covariance ellipses of the predictions alongside the 
        measurements. 
        
        Keyword arguments:
        output_path -- path that the plot will be saved to
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
        Print the contents of the all_contacts dictionary for debugging purposes.
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


    def check_all_contacts_by_distance(self, detect_info, data):
        """
        FOR DEBUGGING PURPOSES 
        Iterate over every contact in the dictionary and return the contact 
        the current detect is most likely associated with by checking Euclidean 
        distance between the prediction and the measurement. If no contact 
        is asociated with this detect, return the timestamp of the current detect 
        message as the new hash_key for the new contact that will be made.

        Keyword arguments:
        detect_info -- the dictionary containing the detect info to be checked
        data -- data from the detect message that was just transmitted

        Returns:
        None if no appropriate contacts are found, otherwise the found contact's id
        """
        
        greatest_pred = float('inf')
        return_contact_id = None 

        for contact in self.all_contacts:
            c = self.all_contacts[contact]
            # Recompute the value for dt, so we can use it to update this Contact's 
            # KalmanFilter's Q(s).
            c.dt = (data.header.stamp - c.last_measured).to_sec()
            # Update the last_measured field for this contact so we know not to 
            # remove it from all_contacts anytime soon. 
            c.last_measured = data.header.stamp
            c.set_Z(detect_info)
          
            for kf in c.filter_bank.filters:
                kf.set_Q(c) 
                kf.set_F(c)
                kf.set_H(c, detect_info)
                kf.set_R(c, detect_info) 
                
                if DEBUG:
                    print('sensor_id: ', detect_info['sensor_id'])
                    print('filter type: ', kf.filter_type)
                    print('contact id: ', c.id)
                    print('________________')  

            c.filter_bank.predict()
    
            # Get the distance between the measurement and the prediction for each filter.
            side1a = abs(detect_info['x_pos'] - c.filter_bank.filters[0].x[0])
            side1b = abs(detect_info['y_pos'] - c.filter_bank.filters[0].x[1])
            side2a = abs(detect_info['x_pos'] - c.filter_bank.filters[1].x[0])
            side2b = abs(detect_info['y_pos'] - c.filter_bank.filters[1].x[1])

            H1 = math.sqrt(side1a**2 + side1b**2)
            H2 = math.sqrt(side2a**2 + side2b**2)
            
            # If both filters have predictions within 10 of the measurement, incorporate 
            # the measurement into the filter.
            if H1 <= 10 and H2 <= 10: 
                if H1 + H2 <= greatest_pred:
                    greatest_pred = H1 + H2 
                    return_contact_id = c.id

        return return_contact_id 
 
 
    def check_all_contacts_by_likelihood(self, detect_info, data):
        """
        FOR DEBUGGING PURPOSES 
        Iterate over every contact in the dictionary and return the contact 
        the current detect is most likely associated with by checking log  
        likilehood of each Kalman filter in the contact. If no contact 
        is asociated with this detect, return the timestamp of the current detect 
        message as the new hash_key for the new contact that will be made.

        Keyword arguments:
        detect_info -- the dictionary containing the detect info to be checked
        data -- data from the detect message that was just transmitted

        Returns:
        None if no appropriate contacts are found, otherwise the found contact's id
        """
        
        greatest_likelihood = 0
        return_contact_id = None 

        for contact in self.all_contacts:
            c = self.all_contacts[contact]
            # Recompute the value for dt, so we can use it to update this Contact's 
            # KalmanFilter's Q(s).
            c.dt = (data.header.stamp - c.last_measured).to_sec()
            # Update the last_measured field for this contact so we know not to 
            # remove it from all_contacts anytime soon. 
            c.last_measured = data.header.stamp
            c.set_Z(detect_info)
            
            for kf in c.filter_bank.filters:
                kf.set_Q(c) 
                kf.set_F(c)
                kf.set_H(c, detect_info)
                kf.set_R(c, detect_info) 
                
            c.filter_bank.predict()
            
            for kf in c.filter_bank.filters:
                kf.set_log_likelihood(c)

                if DEBUG:
                    print('sensor_id: ', detect_info['sensor_id'])
                    print('filter type: ', kf.filter_type)
                    print('contact id: ', c.id)
                    print('likelihood :', kf.get_log_likelihood())
                    print('________________')  

            L1 = c.filter_bank.filters[0].get_log_likelihood()
            L2 = c.filter_bank.filters[1].get_log_likelihood()
            
            # Not sure about this condition
            if L1 / L2 > 0.5: 
                if L1 / L2 > greatest_likelihood:
                    greatest_likelihood = L1 / L2
                    return_contact_id = c.id

        return return_contact_id 
 

    def check_all_contacts_by_BF(self, detect_info, data):
        """
        Iterate over every contact in the dictionary and return the contact 
        the current detect is most likely associated with by checking the  
        Bayes factor of each Kalman filter in the contact. If no contact 
        is asociated with this detect, return the timestamp of the current detect 
        message as the new hash_key for the new contact that will be made.

        Keyword arguments:
        detect_info -- the dictionary containing the detect info to be checked
        data -- data from the detect message that was just transmitted

        Returns:
        None if no appropriate contacts are found, otherwise the found contact's id
        """
        
        greatest_logBF = 0
        return_contact_id = None 

        for contact in self.all_contacts:
            c = self.all_contacts[contact]
            # Recompute the value for dt, so we can use it to update this Contact's 
            # KalmanFilter's Q(s).
            c.dt = (data.header.stamp - c.last_measured).to_sec()
            # Update the last_measured field for this contact so we know not to 
            # remove it from all_contacts anytime soon. 
            c.last_measured = data.header.stamp
            c.set_Z(detect_info)
           
            for kf in c.filter_bank.filters:
                kf.set_Q(c) 
                kf.set_F(c)
                kf.set_H(c, detect_info)
                kf.set_R(c, detect_info) 
                
            c.filter_bank.predict()
            
            for kf in c.filter_bank.filters:
                kf.set_bayes_factor(c, 2.0)

                if DEBUG:
                    print('sensor_id: ', detect_info['sensor_id'])
                    print('filter type: ', kf.filter_type)
                    print('contact id: ', c.id)
                    print('BF :', kf.get_bayes_factor())
                    print('________________')  

            logBF1 = c.filter_bank.filters[0].get_bayes_factor()
            logBF2 = c.filter_bank.filters[1].get_bayes_factor()
            
            if logBF1 > 2 and logBF2 > 2: 
                if logBF1 + logBF2 > greatest_logBF:
                    greatest_logBF = logBF1 + logBF2
                    return_contact_id = c.id

        return return_contact_id 
 
        
    def delete_stale_contacts(self):
        """
        Remove items from the dictionary that have not been measured recently. 
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
        ContactTracker class.
        """

        self.max_stale_contact_time = config['max_stale_contact_time']
        self.initial_velocity = config['initial_velocity']
        return config


    def callback(self, data):
        """
        Listen for detects and and incorporates with filters as approprate.

        Keyword arguments:
        data -- data from the detect message that was just transmitted
        """
        
        ########################################################
        ###### VARIABLE INITIALIZATION AND ERROR HANDLING ######
        ########################################################
        
        # Initialize variables and store in a dictionary.
        detect_info = self.populate_detect_info(data)
        if len(detect_info) == 0: 
            return

        #  If there are no contacts yet, no need to traverse empty dictionary
        #  Otherwise, we have to check each contact in the dictionary to see if 
        #  it is a potential match for our current detect message. 
        contact_id = None 
        if len(self.all_contacts) > 0:
            contact_id = self.check_all_contacts_by_BF(detect_info, data)
        
        if contact_id is None:
           contact_id = data.header.stamp 
            

        #######################################################
        ####### CREATE OR UPDATE CONTACT WITH VARIABLES #######
        #######################################################
        
        if not contact_id in self.all_contacts: 
            first_order_kf = contact_tracker.contact_kf.ContactKalmanFilter(dim_x=6, dim_z=4, filter_type='first')
            second_order_kf = contact_tracker.contact_kf.ContactKalmanFilter(dim_x=6, dim_z=4, filter_type='second')
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
        
            # Incorporate with filters in the filter_bank.
            c.filter_bank.update(c.Z)

        # Append appropriate prior and measurements to lists.
        c.xs.append(np.array([c.filter_bank.x[0], c.filter_bank.x[1]]))
        c.zs.append(np.array([c.info['x_pos'], c.info['y_pos']]))
        c.ps.append(c.filter_bank.P)


        ################################################
        ###### Set fields for the Contact message ######
        ################################################
        contact_msg = Contact()
        contact_msg.header.stamp = detect_info['header'] 
        contact_msg.header.frame_id = "wgs84"
        contact_msg.name = str(c.id)
        contact_msg.callsign = "UNKNOWN"
        #contact_msg.heading = course_made_good # Should I subscribe to the cmg node? 
        #contact_msg.contact_souce = "contact_tracker" #This is supposed to be a float

        # Do a service call to MaptoLatLong.srv to convert map coordinates to 
        # latitude and longitude.
        try:
            print('making a service call')
            rospy.wait_for_service('map_to_long')
            map2long_service = rospy.ServiceProxy('map_to_long', MapToLong)
            print('ServiceProxy made')
                
            map2long_req = MapToLongRequest()
            print('New request instantiated')
            map2long_req.map.point.x = detect_info['x_pos']
            map2long_req.map.point.y = detect_info['y_pos'] 

            llcords = map2long_service(map2long_req)
            print(llcoords)
               
        except rospy.ServiceException, e:
            print("Service call failed: %s", e)
            
        contact_msg.position.latitude = llcoords.wgs84.position.latitude
        contact_msg.position.longitude = llcoords.wgs84.position.longitude
 
        # Convert velocity in x and y into course over ground 
        # and speed over ground.
        vx = detect_info['x_vel'] 
        vy = detect_info['y_vel']
        contact_msg.cog = np.mod(np.arctan2(vx, vy) * 180/np.pi + 360, 360) 
        contact_msg.sog = np.sqrt(vx**2 + vy **2)

        # These fields are assigned arbitrary values for now. 
        contact_msg.mmsi = 0
        contact_msg.dimension_to_srbd = 0
        contact_msg.dimension_to_port = 0
        contact_msg.dimension_to_bow = 0
        contact_msg.dimension_to_stern = 0
        
        ################################################
        ###### Set fields for the Detect message ######
        ################################################
        detect_msg = Detect()
        contact_msg.header.stamp = detect_info['header'] 
        contact_msg.header.frame_id = "map"
        detect_msg.sensor_id = self.name

        # Not entirely sure what to use for these fields. 
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


        ##################################
        ###### Publish the messages ######
        ##################################
        self.pub_contactmap.publish(detect_msg)
        self.pub_contacts.publish(contact_msg)


        ###################################
        ###### Delete stale contacts ######
        ###################################
        self.delete_stale_contacts()


    def run(self, args):
        """
        Initialize the node and set it to subscribe to the detects topic.
        Initialize publishers.
        Plot results.
        """

        rospy.init_node('tracker_debug', anonymous=True)
        srv = Server(contact_trackerConfig, self.reconfigure_callback)
        rospy.Subscriber('/detects', Detect, self.callback)
        
        self.pub_contactmap = rospy.Publisher('/contact_map', Detect, queue_size=1)
        self.pub_contacts = rospy.Publisher('/contacts', Contact, queue_size=1)

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


