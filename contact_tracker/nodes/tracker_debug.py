#!/usr/bin/env python

# A contact is identifed by position, not id, and is
# independent of the sensor that produced it.

# Authors: Rachel White, Val Schmidt
# University of New Hampshire
# Date last modified: 03/30/2020

import math
import rospy
import sys
#import datetime
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from numpy import nan
from copy import deepcopy

import contact_tracker.contact
import contact_tracker.contact_kf
from contact_tracker.cfg import contact_trackerConfig
from .msg import Detect, Contact
from geographic_visualization_msgs.msg import GeoVizItem, GeoVizPointList
from geographic_visualization_msgs.msg import GeoVizPolygon, GeoVizSimplePolygon
from geographic_msgs.msg import GeoPoint
from project11_transformations.srv import MapToLatLong
from project11_transformations.srv import MapToLatLongRequest
import cPickle as pickle

from filterpy.stats.stats import plot_covariance
from dynamic_reconfigure.server import Server


DEBUG = True

class ContactTracker:
    """
    Class to create custom contact tracker.
    """


    def __init__(self):
        """
        Define the constructor.
        """
        self.Ncontacts = 0                      # Number of contacts tracked so far.
        self.all_contacts = {}                  # Contacts currently being tracked.
        self.contact_history_buffer = {}        # All contacts history since 'duration'
        self.contact_buffer_duration = 200.      # Time in seconds to buffer contact history.
        self.detect_history_buffer = []         # List of detects received since 'duration.'
        self.detect_buffer_duration = 200

        # Default value Bayes Filter Hypothesis Test Factor
        # Roughly determines the number of standard deviations away
        # from a contact a detect must be before it is assumed to
        # be a new contact.
        self.Nsigma = 10
        
        # Display parameters.
        self.contact_display_duration = 20      # Duration in secinds of contact information to display.
        self.detect_display_duration = 20
        self.plotcolors = {}           # Used for auto-assigning colors for plots.
        # A color wheel created from the Color Sequence Editor Using CIELuv Uniform Color Space
        # Oliveri and Ware (Ware, 1988)
        self.colorwheel =  [(np.array([.6421733329734752,    
                            0.05260355996805199,    
                            0.0656277553662272])),
                        (np.array([0.4957569461595583,    
                             0.10021478784902092,    
                             0.049194253546821255]) * 255).astype(int),
                        (np.array([0.36800484167532843,
                                      0.1417999955715154,
                                      0.03441488578246513]) * 255).astype(int),
                        (np.array([0.2560770265961858, 
                                   0.17669515411201658,
                                   0.03717623024142687]) * 255).astype(int),
                        (np.array([0.12479017030494177,
                                      0.2142569004760196,
                                      0.07480436427688678]) * 255).astype(int),    
                        (np.array([0.08041813835636266,
                                      0.2165442636649675,
                                      0.19376225757731255]) * 255).astype(int),
                        (np.array([0.08667725914402996,
                                      0.19755071563708018,
                                      0.3675725959108131]) * 255).astype(int),
                        (np.array([0.1134911076177515    ,
                                      0.16324047987890247,
                                      0.6318123323716915]) * 255).astype(int),
                        (np.array([0.36154562492049597,
                                      0.07127852687409003,
                                      0.775006337089268]) * 255).astype(int),
                        (np.array([0.5097685163700237,
                                      0.036552303829553205,
                                      0.6541173361674163]) * 255).astype(int)]
                

    def plot_x_vs_y(self, output_path):
        """
        Plot the measurement's x and y positions
        alongside the prediction's x and y positions.

        Keyword arguments:
        output_path -- path that the plot will be saved to
        """

        minx = 0
        maxx = 0
        miny = 0
        maxy = 0
        fig, ax1 = plt.subplots(1,figsize=(12,12))
        
        tstart = 0
        for id in self.contact_history_buffer:
            contactList = self.contact_history_buffer[id]
            #print(contactList[0].Z)
            
            m_xs = [c.filter_bank.filters[0].z[0] for c in contactList]
            m_ys = [c.filter_bank.filters[0].z[1] for c in contactList]
            e_xs = [c.filter_bank.x[0] for c in contactList]
            e_ys = [c.filter_bank.x[1] for c in contactList]
            p_xs = [c.all_filters[1].x_prior[0] for c in contactList]
            p_ys = [c.all_filters[1].x_prior[1] for c in contactList]
            sigma_mx = np.sqrt(np.array([c.all_filters[0].R[0,0] for c in contactList]))
            sigma_my = np.sqrt(np.array([c.all_filters[0].R[1,1] for c in contactList]))
            sigma_ex = np.sqrt(np.array([c.filter_bank.P[0,0] for c in contactList]))
            sigma_ey = np.sqrt(np.array([c.filter_bank.P[1,1] for c in contactList]))
            sigma_px = np.sqrt(np.array([c.all_filters[1].P_prior[0,0] for c in contactList]))
            sigma_py = np.sqrt(np.array([c.all_filters[1].P_prior[1,1] for c in contactList]))
            tt = np.array([c.evalTime for c in contactList] )
            # It's possible that the script was stopped before this was populated,
            # so catch it.
            if len(tt) == 0:
                continue
            if tstart == 0:
                tstart = tt[0]
            tt = tt - tstart
            

            tmp = np.min(np.array(m_xs))
            if tmp < minx: minx = tmp
            tmp = np.max(np.array(m_xs))
            if tmp > maxx: maxx = tmp
            tmp = np.min(np.array(m_ys))
            if tmp < miny: miny = tmp
            tmp = np.max(np.array(m_ys))
            if tmp > maxy: maxy = tmp

            ax1.scatter(m_xs, m_ys, marker='.', 
                        label='contact' + str(id) + ' meas', 
                        color=self.plotcolors[id])
            ax1.scatter(p_xs, p_ys, marker='x',
                     label='contact' + str(id) + ' pred',
                     color=self.plotcolors[id])
            ax1.plot(e_xs, e_ys,marker='P', linestyle='-', 
                        label='contact' + str(id) + ' est',
                        color = 'r')
            #color=self.plotcolors[contact])
            
            
        plt.legend()
        plt.xlabel('x position')
        plt.ylabel('y position')
        plt.xlim(minx - 20, maxx + 20)
        plt.ylim(miny - 20, maxy + 20)
        plt.grid(True)
        plt.savefig(output_path + 'x_vs_y.png')


    def plot_x_vs_time(self, output_path):
        """
        Plot the measurement's x position at each time unit
        alongside the prediction's x position at each time
        unit.

        Keyword arguments:
        output_path -- path that the plot will be saved to
        """

        minx = 0
        maxx = 0
        miny = 0
        maxy = 0
        fig, (ax1,ax2,ax3,ax4) = plt.subplots(4,sharex=True,figsize=(12,8))
        
        tstart = 0
        
        for cid in self.contact_history_buffer:
            contactList = self.contact_history_buffer[cid]
            #print(contactList[0].Z)

            m_xs = [c.filter_bank.filters[0].z[0] for c in contactList]
            m_ys = [c.filter_bank.filters[0].z[1] for c in contactList]
            e_xs = [c.filter_bank.x[0] for c in contactList]
            e_ys = [c.filter_bank.x[1] for c in contactList]
            p_xs = [c.filter_bank.x_prior[0] for c in contactList]
            p_ys = [c.filter_bank.x_prior[1] for c in contactList]
            sigma_mx = np.sqrt(np.array([c.all_filters[0].R[0,0] for c in contactList]))
            sigma_my = np.sqrt(np.array([c.all_filters[0].R[1,1] for c in contactList]))
            sigma_ex = np.sqrt(np.array([c.filter_bank.P[0,0] for c in contactList]))
            sigma_ey = np.sqrt(np.array([c.filter_bank.P[1,1] for c in contactList]))
            sigma_px = np.sqrt(np.array([c.all_filters[1].P_prior[0,0] for c in contactList]))
            sigma_py = np.sqrt(np.array([c.all_filters[1].P_prior[1,1] for c in contactList]))
            BF = [c.bayes_factor for c in contactList]
            #MU = [x for x in c.filter_bank.mu]
            #print(MU)
            tt = np.array([c.evalTime for c in contactList] )
            # It's possible that the script was stopped before this was populated,
            # so catch it.
            if len(tt) == 0:
                continue

            if tstart == 0:
                tstart = tt[0]
            tt = tt - tstart

            tmp = np.min(np.array(m_xs))
            if tmp < minx: minx = tmp
            tmp = np.max(np.array(m_xs))
            if tmp > maxx: maxx = tmp
            tmp = np.min(np.array(m_ys))
            if tmp < miny: miny = tmp
            tmp = np.max(np.array(m_ys))
            if tmp > maxy: maxy = tmp

            ax1.errorbar(tt,m_xs,yerr = sigma_mx, marker='x',
                        label='meas',ls = '',
                        color=self.plotcolors[cid])
            ax2.errorbar(tt,m_ys, yerr= sigma_my, marker='x',
                        label='meas', ls = '',
                        color=self.plotcolors[cid])

            ax1.errorbar(tt+.1,e_xs,yerr = sigma_ex, marker='P',
                        label='est',
                        color='r')
            ax2.errorbar(tt+.1,e_ys, yerr= sigma_ey, marker='P',
                        label='est',
                        color='r')

            ax1.errorbar(tt+.15,p_xs,yerr = sigma_px, marker='.',
                        label='pred',ls = '',
                        color='g')
            ax2.errorbar(tt+.15,p_ys, yerr= sigma_py, marker='.',
                        label='pred',ls = '',
                        color='g')
            ax3.plot(tt,BF,'-',
                     marker='x',
                     color = 'm')
            '''ax4.plot(tt,[x[0] for x in MU],'-',
                     marker='v',
                     color = 'g')
            ax4.plot(tt,[x[1] for x in MU],'-',
                     marker='o',
                     color='b')
            '''

        ax1.set_ylabel('x')
        ax1.set_ylim(minx-5, maxx+5)
        ax1.grid(True)
        ax2.set_ylabel('y')
        ax2.set_ylim(miny-5, maxy+5)
        ax2.grid(True)
        ax3.grid(True)
        ax3.set_ylim(0,20)
        ax3.set_ylabel('LOG BF')
        #ax4.grid(True)
        #ax4.ylabel('MU, V (g), A (b)')
        #ax1.legend()
        #ax2.legend()
        #plt.legend()
        #plt.xlabel('time')
        #plt.ylabel('position')
        #plt.ylim(0, 300)
        
        plt.savefig(output_path + 'x_vs_time.png')



    def plot_ellipses(self, output_path):
        """
        Plot the covariance ellipses of the predictions alongside the
        measurements.

        Keyword arguments:
        output_path -- path that the plot will be saved to
        """

        minx = 0
        maxx = 0
        miny = 0
        maxy = 0
        fig, ax1 = plt.subplots(1,figsize=(12,12))
        
        tstart = 0
        for id in self.contact_history_buffer:
            contactList = self.contact_history_buffer[id]
            #print(contactList[0].Z)
            
            m_xs = [c.filter_bank.filters[0].z[0] for c in contactList]
            m_ys = [c.filter_bank.filters[0].z[1] for c in contactList]
            e_xs = [c.filter_bank.x[0] for c in contactList]
            e_ys = [c.filter_bank.x[1] for c in contactList]
            p_xs = [c.all_filters[1].x_prior[0] for c in contactList]
            p_ys = [c.all_filters[1].x_prior[1] for c in contactList]
            R = [c.all_filters[0].R for c in contactList]
            P = [c.filter_bank.P for c in contactList]
            P_prior = [c.all_filters[1].P_prior for c in contactList]
            tt = np.array([c.evalTime for c in contactList] )
            # It's possible that the script was stopped before this was populated,
            # so catch it.
            if len(tt) == 0:
                continue
            if tstart == 0:
                tstart = tt[0]
            tt = tt - tstart
            
            tmp = np.min(np.array(m_xs))
            if tmp < minx: minx = tmp
            tmp = np.max(np.array(m_xs))
            if tmp > maxx: maxx = tmp
            tmp = np.min(np.array(m_ys))
            if tmp < miny: miny = tmp
            tmp = np.max(np.array(m_ys))
            if tmp > maxy: maxy = tmp

            ax1.scatter(m_xs, m_ys, marker='.', 
                        label='contact' + str(id) + ' meas', 
                        color=self.plotcolors[id])
            ax1.scatter(p_xs, p_ys, marker='x',
                     label='contact' + str(id) + ' pred',
                     color=self.plotcolors[id])
            ax1.plot(e_xs, e_ys,marker='P', linestyle='-', 
                        label='contact' + str(id) + ' est',
                        color = 'r')

            for i in range(len(m_xs)):
                plot_covariance([m_xs[i],m_ys[i]],
                                cov = np.array(R[i][:2,:2]),
                                show_center= True,
                                ec = 'b')
                plot_covariance([e_xs[i], e_ys[i]],
                                cov = np.array(P[i][:2,:2]),
                                show_center= True,
                                ec = 'r')
                plot_covariance([p_xs[i],p_ys[i]],
                                cov = np.array(P_prior[i][:2,:2]),
                                show_center=True,
                                ec = 'g')

        plt.xlabel('x position')
        plt.ylabel('y position')
        plt.xlim(minx - 20, maxx + 20)
        plt.ylim(miny - 20, maxy + 20)
        plt.grid(True)
        plt.legend()
        plt.savefig(output_path + 'ellipses.png')
        
    
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
                'type': "UNKNOWN"
                }
        
        # Assign values only if they are not NaNs
        if ((not math.isnan(data.pose.pose.position.x)) and
            (not math.isnan(data.pose.pose.position.y))):
            detect_info['x_pos'] = float(data.pose.pose.position.x)
            detect_info['y_pos'] = float(data.pose.pose.position.y)
            detect_info['type'] = "POSITIONONLY"

        if ((not math.isnan(data.twist.twist.linear.x)) and
                (not math.isnan(data.twist.twist.linear.y))):

            detect_info['x_vel'] = float(data.twist.twist.linear.x)
            detect_info['y_vel'] = float(data.twist.twist.linear.y)
            if detect_info["type"] == "POSITIONONLY":
                detect_info["type"] = "POSITIONANDVELOCITY"
            else:
                detect_info["type"] = "VELOCITYONLY"

        if detect_info["type"] == "UNKNOWN":
            rospy.loginfo("Missing complete detection data.")
            rospy.loginfo(data)
            detect_info = None

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

        for contact_id in self.all_contacts:
            c = self.all_contacts[contact_id]
            
            for kf in c.filter_bank.filters:
                
                if DEBUG:
                    print('sensor_id: ', detect_info['sensor_id'])
                    print('filter type: ', kf.filter_type)
                    print('contact id: ', c.id)
                    print('likelihood :', kf.get_log_likelihood())
                    print('________________')
                    
            # L1 = c.filter_bank.filters[0].get_log_likelihood()
            # L2 = c.filter_bank.filters[1].get_log_likelihood()

            # EXPERIMENTAL
            # Here only the second order filter is being evaluated, as it
            # accommodates the largest changes. I'm using the likelihood rather
            # than the log liklihood, which is equivalent, but easier to
            # understand.

            for kf in c.filter_bank.filters:
                if kf.filter_type == 'second':
                    #L = np.exp(kf.get_log_likelihood())
                    L = kf.L
                    #print("Contact: %s L: %0.4f Last dT: %f" % 

                    #      (contact_id,L,c.dt))

                   # This requires the measurement to be somewhat likely (1/20).
                   # Otherwise the measurement is not considered to be a candidate
                   # measurement of the contact.
                    if L > 0.05:
                        # Here we keep track of the contact for whom the measurement
                        # has the greatest likelihood.
                        if L > greatest_likelihood:
                            greatest_likelihood = L
                            return_contact_id = contact_id



            # Not sure about this condition
            #if L1 / L2 > 0.5:
            #    if L1 / L2 > greatest_likelihood:
            #        greatest_likelihood = L1 / L2
            #        return_contact_id = c.id
        print("   Greatest Likelihood: %0.4f, Contact: %s" %
              (greatest_likelihood,return_contact_id))
        return return_contact_id

    def check_all_contacts_by_IMM_BF(self):

        greatest_logBF = 0
        return_contact_id = None

        for contact in self.all_contacts:
            c = self.all_contacts[contact]
            if c.bayes_factor > 2.:
                if c.bayes_factor > greatest_logBF:
                    greatest_logBF = c.bayes_factor
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
            
            
            for kf in c.filter_bank.filters:
                '''
                kf.set_bayes_factor(c, 2.0)

                if DEBUG:
                    
                    print('sensor_id: ', detect_info['sensor_id'])
                    print('filter type: ', kf.filter_type)
                    print('contact id: ', c.id)
                    print('BF :', kf.get_bayes_factor())

                    print('________________')  
                    
                '''
                    
            #logBF1 = c.filter_bank.filters[0].get_bayes_factor()
            if kf.filter_type == 'second':
                logBF2 = c.filter_bank.filters[1].get_bayes_factor()
            
            # Sets a minimum criteria for an detect to be associated with a
            # contact, and then returns the contact id for whom the odds are 
            # greatest that the detect matches. 
            if logBF2 > 2.:
                if logBF2 > greatest_logBF:
                    greatest_logBF = logBF2
                    return_contact_id = c.id
        
        print("   Greatest LOG(BF): %0.4f, Contact: %s" % 
        (greatest_logBF,return_contact_id))
                    
        '''
        if logBF1 > 2 and logBF2 > 2: 
            if logBF1 + logBF2 > greatest_logBF:
                greatest_logBF = logBF1 + logBF2
                return_contact_id = c.id
        '''
        return return_contact_id 
 
    def setupContactsForDetect(self, detect_info):
        '''
        Loops through the contacts, populates Q, F, H and R for the measured
        time and parameters. Sets c.dt which is the time since the last time
        the contact position was predicted. Finally, for each filter, predicts
        the location of the contact at the measurement time.

        These steps are required prior to evaulating whether the received detect
        is likely a measure of a given contact, or a new contact altogether.

        NB. The prediction step done here populates ONLY KF.x_prior and
        KF.P_prior, and NOT KF.x and KF.P. This detail is important, because
        the "prior" variables allow us to use these predicted states to
        test for whether a measurement should be associated with the contact
        without actually modifying the contact's state (just in case the test
        fails).

        Keyword arguments:
        detect_info -- the dictionary containing the detect info to use 
        '''

        for contact_id in self.all_contacts:
            c = self.all_contacts[contact_id]
            # Recompute the value for dt, so we can use it to update this Contact's
            # KalmanFilter's Q(s).
            c.dt = (detect_info['header'].stamp - c.last_measured).to_sec()
            c.last_detect_info = detect_info
            c.set_Z()
            c.set_R()
            c.set_Q()    # sets Q for all filters.
            c.set_H()
            c.set_F(dt =  c.dt)
            # Unlike predict(), predict_prior() does not update the state, x, just x_prior and P_prior.
            # This is useful for statistical tests between the measurement and the predicted
            # state of the contact at measurement time, but prior to the state update without
            # actually changing the state, which predict() would do.
            c.predictIMMprior()  # Does not update state.
            #c.calculate_measurement_likelihood()
            c.calculate_IMM_bayes_factor(Nsigma = self.Nsigma)

            '''
            for kf in c.filter_bank.filters:
                kf.set_F(c)
                kf.set_H(c, detect_info)
                kf.set_R(c, detect_info)
                kf.predict_prior()   # This does not update the state, x. Just x_prior.
                #kf.set_log_likelihood(c)
                kf.set_likelihood(c)
                #kf.set_bayes_factor(c, 2.0)
                kf.set_bayes_factor(c, 3.0)
            '''
            '''
            print("C: %s: Prior X,Y: %0.3f,%0.3f L: %f ll: %f, logBF: %0.2f" %
                  (c.id,
                   np.sqrt(c.filter_bank.P_prior[0,0]),
                   np.sqrt(c.filter_bank.P_prior[1,1]),
                   c.filter_bank.likelihood[0], kf.ll, c.bayes_factor))
                    #               np.exp(kf.get_log_likelihood())))
            '''
            print(c.filter_bank)

    def delete_stale_contacts(self):
        """
        Remove items from the dictionary that have not been measured recently,
        or whose uncertainty is too large.
        """

        for contact_id in list(self.all_contacts):
            cur_contact = self.all_contacts[contact_id]
            time_between_now_and_last_measured = (rospy.get_rostime() -
                                                  cur_contact.last_measured).to_sec() / 60.0
            print(" Contact: %s, dT %0.3f m" %
                  (contact_id,time_between_now_and_last_measured))

            # CONDITIONS FOR DROPPING CONTACTS:
            # TIME
            if (time_between_now_and_last_measured >
                self.max_stale_contact_time):
                rospy.loginfo('Deleting stale Contact from dictionary, %0.3f' % 
                              time_between_now_and_last_measured)
                del self.all_contacts[contact_id]
                
            # UNCERTAINTY
            # TODO: Move the max uncertainty to a config file.
            # TODO: Make he max uncertainty a function of the detection range
            # so that it grows with distance. This will allow us to track
            # distant objects
            for kf in cur_contact.all_filters:
                if kf.filter_type == 'second':
                    if (kf.P_prior[0,0] > 1000**2 or kf.P_prior[1,1] > 1000**2):
                        rospy.loginfo(('Deleting state contact -' + 
                                      'x,y Var too large: %0.1f,%0.1f')
                                      % (np.sqrt(kf.P_prior[0,0]), 
                                         np.sqrt(kf.P_prior[1,1])))
                        del self.all_contacts[contact_id]


    def reconfigure_callback(self, config, level):
        """
        Get the parameters from the cfg file and assign them to the member variables of the
        ContactTracker class.
        """

        self.max_stale_contact_time = config['max_stale_contact_time']
        self.initial_velocity = config['initial_velocity']
        return config


    def add_contact(self, cid, detect_info):
        """
        Initialize new contact and add it to all_contacts.

        Keyword arguments:
        cid -- timestamp representing unique id of this contact object
        detect_info -- the dictionary containing the detect info to use 
        """

        '''
        first_order_kf = contact_tracker.contact_kf.ContactKalmanFilter(dim_x=6, dim_z=4, filter_type='first')
        second_order_kf = contact_tracker.contact_kf.ContactKalmanFilter(dim_x=6, dim_z=4, filter_type='second')
        all_filters = [first_order_kf, second_order_kf]
        '''
        self.all_contacts[cid] = contact_tracker.contact.Contact(detect_info, cid)

        # Extra stuff for plotting/debugging.
        colors = cm.rainbow(np.linspace(0, 1, 8)) # Generate 8 colors from the rainbow colormap
        self.plotcolors[cid] = colors[np.mod(len(self.all_contacts), len(colors))] # Pick the subsequent color from this colormap each time we make a new contact
        
    def publish_all_contacts(self, timer):
        ''' publish all tracked contacts '''
        for cid in self.all_contacts:
            self.publish_contact(self.all_contacts[cid])

    def publish_contact(self, c):
        ''' publish a tracked contact '''

        msg = Contact()
        msg.header.stamp = c.last_measured
        msg.header.frame_id = "wgs84"
        msg.name = "T-%s" % str(c.id)
        msg.callsign = "UNKNOWN"
        
        try:
            rospy.wait_for_service('map_to_wgs84')
            project11_transformation_node = rospy.ServiceProxy('map_to_wgs84', MapToLatLong)

            req = MapToLatLongRequest()
            req.map.point.x = c.filter_bank.x[0]
            req.map.point.y = c.filter_bank.x[1]

            llcoords = project11_transformation_node(req)
            msg.position.latitude = llcoords.wgs84.position.latitude
            msg.position.longitude = llcoords.wgs84.position.longitude

        except rospy.ServiceException, e:
            print("Service call failed: %s", e)
        
        vx = c.filter_bank.x[2]
        vy = c.filter_bank.x[3]
        msg.heading = np.mod(np.arctan2(vx, vy) + 2. * np.pi, 2. * np.pi)
        msg.cog = msg.heading
        msg.sog = np.sqrt(vx**2 + vy**2)
        msg.mmsi = c.id + 10000
        msg.dimension_to_stbd = 1.
        msg.dimension_to_port = 1.
        msg.dimension_to_bow = 3.
        msg.dimension_to_stern = 3.
        
        self.pub_contacts.publish(msg)
        
    def publish_detect_to_CAMP(self):
        """
        Publish detection to CAMP.
        
        Keyword arguments:
        detect_info -- the dictionary containing the detect info to publish         
        c -- Contact object for which to publish data 
        """
        
        # First we need to convert the position of the latest detect 
        # to lat/lon.
        
        # Do a service call to MaptoLatLong.srv to convert map coordinates to
        # latitude and longitude.
        try:
            rospy.wait_for_service('map_to_wgs84')
            project11_transformation_node = rospy.ServiceProxy('map_to_wgs84', 
                                                               MapToLatLong)

            req = MapToLatLongRequest()
            req.map.point.x = self.detect_history_buffer[-1]['x_pos']
            req.map.point.y = self.detect_history_buffer[-1]['y_pos']

            llcoords = project11_transformation_node(req)
            self.detect_history_buffer[-1]['latitude'] = llcoords.wgs84.position.latitude
            self.detect_history_buffer[-1]['longitude'] = llcoords.wgs84.position.longitude
            
            '''
            msg.label_position.latitude = llcoords.wgs84.position.latitude
            msg.label_position.longitude = llcoords.wgs84.position.longitude

            point = GeoPoint()
            point.latitude = llcoords.wgs84.position.latitude
            point.longitude = llcoords.wgs84.position.longitude
            '''
            
        except rospy.ServiceException, e:
            print("Service call failed: %s", e)
            return        
        

        detect_sensors = np.unique([x['sensor_id'] for x in self.detect_history_buffer])
        
        index = 0
        for detect_sensor in detect_sensors:
            msg = GeoVizItem()
            msg.id = 'R0'
                    
            #msg.id = str(detect_info['header'].stamp.to_sec())
            msg.label = "DETECTS" 
            pointlist = GeoVizPointList()
            
            for detect_info in self.detect_history_buffer:
                if detect_info['sensor_id'] == detect_sensor:    
                    point = GeoPoint()
                    point.latitude = detect_info['latitude']
                    point.longitude = detect_info['longitude']            
                    pointlist.points.append(point)

            pointlist.color.r = self.colorwheel[index * len(self.colorwheel)][0]
            pointlist.color.g = self.colorwheel[index * len(self.colorwheel)][1]
            pointlist.color.b = self.colorwheel[index * len(self.colorwheel)][2]
            pointlist.color.a = 0.2
            pointlist.size = 10.
            msg.point_groups = [pointlist]
            msg.label_position.latitude = point.latitude
            msg.label_position.longitude = point.longitude
            print(msg)
            self.pub_detects.publish(msg)
            index = index + 1
        

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
        if detect_info is None:
            return

        # Capture a buffer of received detections for debugging.
        self.detect_history_buffer.append(deepcopy(detect_info))

        # Remove buffered items older than detect_buffer_duration (seconds)
        self.detect_history_buffer = [x for x in self.detect_history_buffer if
                                      ((rospy.get_rostime().to_sec() - x['header'].stamp.to_sec()) <
                                       self.detect_buffer_duration)]

        self.publish_detect_to_CAMP()

        #########################################################
        ####### CREATE OR UPDATE CONTACT WITH MEASUREMENT #######
        #########################################################
        # Sets up existing contacts for comparison with this detection data.
        if len(self.all_contacts) > 0:
            self.setupContactsForDetect(detect_info)
            contact_id = self.check_all_contacts_by_IMM_BF()
            # If contact_id is None, the detect was not associated with any existing contact.
            # so make a new one.
            if contact_id is None:
                self.Ncontacts += 1
                contact_id = self.Ncontacts
                self.add_contact(self.Ncontacts, detect_info)
        else:
            self.Ncontacts += 1
            contact_id = self.Ncontacts
            self.add_contact(contact_id, detect_info)

        c = self.all_contacts[contact_id]

        # Incorporate with filters in the filter_bank.
        c.filter_bank.predict()
        # setupContacts... above sets the measurement vector, Z, for each filter
        # in the IMM filter bank. This is somewhat atypical, but inconsequential
        # and we can extract it from one to conform to the standard IMM.update() call.
        c.filter_bank.update(c.all_filters[0].Z)

        print("MU: %0.3f, %0.3f" % (c.filter_bank.mu[0],c.filter_bank.mu[1]))
        c.last_measured = detect_info['header'].stamp

        #########################################################
        # Capture states of tracked contacts in contact history #
        #########################################################
        for cid in self.all_contacts:
            # This will be a little memory hungry, but it will capture the
            # entire state of all contacts at each time step. Needed to debug.
            self.all_contacts[cid].evalTime = detect_info['header'].stamp.to_sec()

            if cid not in self.contact_history_buffer:
                self.contact_history_buffer[cid] = []

            self.contact_history_buffer[cid].append(deepcopy(self.all_contacts[cid]))
            c = self.all_contacts[cid]

        ########################################################
        # Remove states from contact history that are too old. #
        ########################################################
        
        for cid in self.contact_history_buffer:
            contactList = self.contact_history_buffer[cid]
            # Remove items older than the intended buffer duration.
            self.contact_history_buffer[cid] = [x for x in contactList if 
                               (rospy.get_rostime().to_sec() - x.evalTime) < 
                               self.contact_buffer_duration]
            

        ###################################
        ###### DELETE STALE CONTACTS ######
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

        #self.pub_contactmap = rospy.Publisher('/contact_map', Detect, queue_size=1)
        self.pub_contacts = rospy.Publisher('/contact', Contact, queue_size=1)
        self.pub_detects = rospy.Publisher('/udp/project11/display', 
                                           GeoVizItem, 
                                           queue_size=1 )
        timer = rospy.Timer(rospy.Duration(1), self.publish_all_contacts)

        rospy.spin()

        # Save detect and contact buffers for debugging.
        print("Saving buffer data to tracker_debug.dat")
        with open('tracker_debug.dat', 'wb') as F:
            pickle.dump(self.contact_history_buffer,F)
            pickle.dump(self.detect_history_buffer,F)
        F.close()

        for plottype in args.plot_type:
            print("Plotting: %s" % plottype)
            if plottype == 'xs_ys':
                self.plot_x_vs_y(args.o)
            elif plottype =='xs_times':
                self.plot_x_vs_time(args.o)
            elif plottype == 'ellipses':
                self.plot_ellipses(args.o)


def main():

    arg_parser = argparse.ArgumentParser(description='Track contacts by applying Kalman filters to incoming detect messages. Optionally plot the results of the filter.')
    arg_parser.add_argument('-plot_type', type=str, action='append',
                            choices=['xs_ys', 'xs_times', 'ellipses'], 
                            help='specify the type of plot to produce, if you want one')
    arg_parser.add_argument('-o', type=str, help='path to save the plot produced, default: tracker_plot, current working directory', default='tracker_plot')
    arg_parser.add_argument('-Nsigma', type=float, default=10.0,
                            help=('Roughly indicates how many sigma away from contact a detect' 
                                'can be to still be considered a measure of the contact.'))
    args = arg_parser.parse_args()
    

    try:
        ct = ContactTracker()
        ct.Nsigma = args.Nsigma
        ct.run(args)

    except rospy.ROSInterruptException:
        rospy.loginfo('Falied to initialize the ContactTracker')
        pass


if __name__=='__main__':
    main()
