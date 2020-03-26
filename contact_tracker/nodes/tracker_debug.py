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
import matplotlib.cm as cm

import contact_tracker.contact
import contact_tracker.contact_kf
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
        """

        self.all_contacts = {}
        self.plotcolors = {}


    def plot_x_vs_y(self, output_path):
        """
        Plot the measurement's x and y positions 
        alongside the prediction's x and y positions.

        Keyword arguments:
        output_path -- path that the plot will be saved to
        """
        
        '''
        all_mxs = []
        all_mys = []
        all_pxs = []
        all_pys = []
        '''
        minx = 0
        maxx = 0
        miny = 0
        maxy = 0
        
        plt.figure(figsize=(10,10))            
        
        for contact in self.all_contacts:
            c = self.all_contacts[contact]
        
            m_xs = []
            m_ys = []
            p_xs = []
            p_ys = []
            e_xs = []
            e_ys = []

            for i in c.zs:
                m_xs.append(i[0])
                m_ys.append(i[1])
            
            for i in c.xs:
                p_xs.append(i[0])
                p_ys.append(i[1])
                
            for i in c.xs:
                e_xs.append(i[0])
                e_ys.append(i[1])

            plt.scatter(m_xs, m_ys, marker='o', 
                        label='contact' + str(c.id) + ' meas', 
                        color=self.plotcolors[contact])
            plt.plot(p_xs, p_ys, marker='x',
                     label='contact' + str(c.id) + ' pred',
                     color=self.plotcolors[contact])
            plt.scatter(e_xs, e_ys,marker='.', linestyle='-', 
                        label='contact' + str(c.id) + ' est', 
                        color=self.plotcolors[contact])

                
            tmp = np.min(np.array(m_xs))
            if tmp < minx: minx = tmp
            tmp = np.max(np.array(m_xs))
            if tmp > maxx: maxx = tmp
            tmp = np.min(np.array(m_ys))
            if tmp < miny: miny = tmp
            tmp = np.max(np.array(m_ys))
            if tmp > maxy: maxy = tmp
            
            '''
            all_mxs.append(m_xs)
            all_mys.append(m_ys)
            all_pxs.append(p_xs)
            all_pys.append(p_ys)
            '''
        '''    
        minx = np.min([np.min(np.array(x)) for x in all_mxs])
        maxx = np.max([np.max(np.array(x)) for x in all_mxs])
        miny = np.min([np.min(np.array(x)) for x in all_mys])
        maxy = np.max([np.max(np.array(x)) for x in all_mys])

        for i in range(0, len(all_mxs)):
            plt.scatter(all_mxs[i], all_mys[i], linestyle='-', 
                        label='contact' + str(i) + ' measurements', 
                        color=self.plotcolors[c])
            plt.plot(all_pxs[i], all_pys[i], 
                     label='contact' + str (i) + ' predictions',
                     color=self.plotcolors[c])
        '''
        plt.legend()
        plt.xlabel('x position')
        plt.ylabel('y position')
        plt.xlim(minx - 20, maxx + 20)
        plt.ylim(miny - 20, maxy + 20)
        plt.grid(True)
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
            plt.scatter(all_times[i], all_mxs[i], linestyle='-', 
                        label='kf ' + str(i) + ' measurements', color='y')
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

        minx = np.min([np.min(np.array(x)) for x in all_pxs])
        maxx = np.max([np.max(np.array(x)) for x in all_pxs])
        miny = np.min([np.min(np.array(x)) for x in all_pys])
        maxy = np.max([np.max(np.array(x)) for x in all_pys])
            
        for i in range(0, len(all_pxs)):
            plt.plot(all_pxs[i], all_pys[i], label='predictions', color='g')

        plt.xlabel('x position')
        plt.ylabel('y position')
        plt.xlim(minx - 20, maxx + 20)
        plt.ylim(miny - 20, maxy + 20)
        plt.grid(True)
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
        
        greatest_likelihood = 0.0
        return_contact_id = None 

        for contact in self.all_contacts:
            c = self.all_contacts[contact]
            # Recompute the value for dt, so we can use it to update this Contact's 
            # KalmanFilter's Q(s).
            c.dt = (data.header.stamp - c.last_measured).to_sec()
            # Update the last_measured field for this contact so we know not to 
            # remove it from all_contacts anytime soon. 
            c.last_measured = data.header.stamp
            
            for kf in c.filter_bank.filters:
                kf.set_Q(c) 
                kf.set_F(c)
                kf.set_H(c, detect_info)
                kf.set_R(c, detect_info) 
                
            c.filter_bank.predict()
            # Setting the measurement vector must happen after the predict step.
            # This is because we have to adjust the measurement matrix to ensure
            # that the predicted state and measured quantities are identical for 
            # fields we didn't measure (their innovatin is 0). Normally we would just omit what we didn't
            # measure, and use the measurement matrix, H, to adjust the sizes of
            # the other calculations to acccomodate. But in the IMM framework, we
            # need a uniform size. 
            c.set_Z(detect_info)
            
            for kf in c.filter_bank.filters:
                kf.set_log_likelihood(c)

                if DEBUG:
                    '''
                    print('sensor_id: ', detect_info['sensor_id'])
                    print('filter type: ', kf.filter_type)
                    print('contact id: ', c.id)
                    print('likelihood :', kf.get_log_likelihood())
                    print('________________')  
                    '''
            L1 = c.filter_bank.filters[0].get_log_likelihood()
            L2 = c.filter_bank.filters[1].get_log_likelihood()
            
            # EXPERIMENTAL
            # Here only the second order filter is being evaluated, as it 
            # accommodates the largest changes. I'm using the likelihood rather
            # than the log liklihood, which is equivalent, but easier to 
            # understand. 
            
            for kf in c.filter_bank.filters:
                if kf.filter_type == 'second':
                    L = np.exp(kf.get_log_likelihood())
                    print("Contact: %s Likelihood: %0.4f" % (c.id,L))
 
                   # This requires the measurement to be somewhat likely (1/20).
                   # Otherwise the measurement is not considered to be a candidate
                   # measurement of the contact.
                    if L > 0.05:              
                        # Here we keep track of the contact for whom the measurement
                        # has the greatest likelihood.
                        if L > greatest_likelihood:
                            greatest_likelihood = L
                            return_contact_id = c.id
                            
                        
            
            # Not sure about this condition
            #if L1 / L2 > 0.5: 
            #    if L1 / L2 > greatest_likelihood:
            #        greatest_likelihood = L1 / L2
            #        return_contact_id = c.id
        print("   Greatest Likelihood: %0.4f, Contact: %s" % (greatest_likelihood,return_contact_id))
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

    def add_contact(self,id,detect_info):
        
        first_order_kf = contact_tracker.contact_kf.ContactKalmanFilter(dim_x=6, dim_z=4, filter_type='first')
        second_order_kf = contact_tracker.contact_kf.ContactKalmanFilter(dim_x=6, dim_z=4, filter_type='second')
        all_filters = [first_order_kf, second_order_kf]
        c = contact_tracker.contact.Contact(detect_info, all_filters, id)
        c.init_filters()
        c.filter_bank = IMMEstimator(all_filters, c.mu, c.M)
        self.all_contacts[id] = c
        colors = cm.rainbow(np.linspace(0, 1, 8))
        self.plotcolors[id] = colors[np.mod(len(self.all_contacts),len(colors))]
        

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
            contact_id = self.check_all_contacts_by_likelihood(detect_info, data)
        
        if contact_id is None:
           contact_id = data.header.stamp 
            

        #######################################################
        ####### CREATE OR UPDATE CONTACT WITH VARIABLES #######
        #######################################################
        
        if not contact_id in self.all_contacts: 
            self.add_contact(contact_id,detect_info)
            c = self.all_contacts[contact_id]
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

        # Remove items from the dictionary that have not been measured recently. 
        self.delete_stale_contacts()


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


