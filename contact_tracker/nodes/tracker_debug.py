#!/usr/bin/env python

# A contact is identifed by position, not id, and is
# independent of the sensor that produced it.

# Authors: Rachel White, Val Schmidt
# University of New Hampshire
# Date last modified: 03/30/2020

import math
import rospy
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from numpy import nan
from copy import deepcopy

import contact_tracker.contact
import contact_tracker.contact_kf
from contact_tracker.cfg import contact_trackerConfig
from marine_msgs.msg import Detect, Contact
from project11_transformations.srv import MapToLatLong
from project11_transformations.srv import MapToLatLongRequest

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

        self.all_contacts = {}
        self.plotcolors = {}
        self.all_contact_history = {}


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
            
            plt.scatter(m_xs, m_ys, marker='.',
                        label='contact' + str(c.id) + ' meas',
                        color=self.plotcolors[c.id])
            plt.plot(p_xs, p_ys, marker='x',
                     label='contact' + str(c.id) + ' pred',
                     color=self.plotcolors[c.id])
            plt.scatter(e_xs, e_ys,marker='P', linestyle='-',
                        label='contact' + str(c.id) + ' est',
                        color = 'r')
                        #color=self.plotcolors[contact])

            tmp = np.min(np.array(m_xs))
            if tmp < minx: minx = tmp
            tmp = np.max(np.array(m_xs))
            if tmp > maxx: maxx = tmp
            tmp = np.min(np.array(m_ys))
            if tmp < miny: miny = tmp
            tmp = np.max(np.array(m_ys))
            if tmp > maxy: maxy = tmp

        #plt.legend()
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

        minx = 0
        maxx = 0
        miny = 0
        maxy = 0
        fig, (ax1,ax2) = plt.subplots(2, sharex=True, figsize=(12,6))
        tstart = 0
        
        for cid in self.all_contact_history:
            contactList = self.all_contact_history[cid]
            print(contactList[0].Z)

            m_xs = [c.Z[0] for c in contactList]
            m_ys = [c.Z[1] for c in contactList]
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


        ax1.set_ylabel('x')
        ax1.set_ylim(minx-5, maxx+5)
        ax1.grid(True)
        ax2.set_ylabel('y')
        ax2.set_ylim(miny-5, maxy+5)
        ax2.grid(True)
        ax1.legend()
        ax2.legend()
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
                kf.set_log_likelihood(c)

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
                    L = np.exp(kf.get_log_likelihood())
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


    def setup_contacts_for_detect(self, detect_info):
        """ 
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
        """ 

        for contact_id in self.all_contacts:
            c = self.all_contacts[contact_id]
            # Recompute the value for dt, so we can use it to update this Contact's
            # KalmanFilter's Q(s).
            c.dt = (detect_info['header'].stamp - c.last_measured).to_sec()
            c.set_Z(detect_info)
            c.set_Q()    # sets Q for all filters.
            
            for kf in c.filter_bank.filters:
                kf.set_F(c)
                kf.set_H(c, detect_info)
                kf.set_R(c, detect_info)
                kf.predict_prior()   # This does not update the state, x. Just x_prior.
                
                if kf.filter_type == 'second':
                    print("C: %s: Prior X,Y: %0.3f,%0.3f" %
                          (c.id,
                           np.sqrt(kf.P_prior[0,0]),
                           np.sqrt(kf.P_prior[1,1])))


    def delete_stale_contacts(self):
        """
        Remove items from the dictionary that have not been measured recently.
        """

        for contact_id in list(self.all_contacts):
            cur_contact = self.all_contacts[contact_id]
            time_between_now_and_last_measured = (rospy.get_rostime() -
                                                  cur_contact.last_measured).to_sec() / 60.0
            print(" Contact: %s, dT %0.3f m" %
                  (contact_id,time_between_now_and_last_measured))
            if time_between_now_and_last_measured > self.max_stale_contact_time:
                rospy.loginfo('Deleting stale Contact from dictionary, %0.3f' %
                              time_between_now_and_last_measured)
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

        first_order_kf = contact_tracker.contact_kf.ContactKalmanFilter(dim_x=6, dim_z=4, filter_type='first')
        second_order_kf = contact_tracker.contact_kf.ContactKalmanFilter(dim_x=6, dim_z=4, filter_type='second')
        all_filters = [first_order_kf, second_order_kf]
        c = contact_tracker.contact.Contact(detect_info, all_filters, cid)
        self.all_contacts[cid] = c
        colors = cm.rainbow(np.linspace(0, 1, 8)) # Generate 8 colors from the rainbow colormap
        self.plotcolors[cid] = colors[np.mod(len(self.all_contacts), len(colors))] # Pick the subsequent color from this colormap each time we make a new contact
        

    def publish_msgs(self, c, detect_info):
        """
        Initialize new contact and add it to all_contacts.

        Keyword arguments:
        detect_info -- the dictionary containing the detect info to publish         
        c -- Contact object for which to publish data 
        """

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
            rospy.wait_for_service('map_to_wgs84')
            project11_transformation_node = rospy.ServiceProxy('map_to_wgs84', MapToLatLong)

            req = MapToLatLongRequest()
            req.map.point.x = detect_info['x_pos']
            req.map.point.y = detect_info['y_pos']

            llcoords = project11_transformation_node(req)
            contact_msg.position.latitude = llcoords.wgs84.position.latitude
            contact_msg.position.longitude = llcoords.wgs84.position.longitude

        except rospy.ServiceException, e:
            print("Service call failed: %s", e)

        # Convert velocity in x and y into course over ground
        # and speed over ground.
        vx = detect_info['x_vel']
        vy = detect_info['y_vel']
        contact_msg.cog = np.mod(np.arctan2(vx, vy) * 180/np.pi + 360, 360)
        contact_msg.sog = np.sqrt(vx**2 + vy **2)

        # These fields are assigned arbitrary values for now.
        contact_msg.mmsi = 0
        contact_msg.dimension_to_stbd = 0
        contact_msg.dimension_to_port = 0
        contact_msg.dimension_to_bow = 0
        contact_msg.dimension_to_stern = 0


        ################################################
        ###### Set fields for the Detect message ######
        ################################################
        detect_msg = Detect()
        detect_msg.header.stamp = detect_info['header']
        detect_msg.header.frame_id = "map"
        detect_msg.sensor_id = detect_info['sensor_id']

        # Not sure if this is the right thing to do...
        for kf in c.filter_bank.filters:
            if kf.filter_type == 'first':
                detect_msg.pose.covariance = kf.P
            elif kf.filter_type == 'second':
                detect_msg.twist.covariance = kf.P 


        ##################################
        ###### Publish the messages ######
        ##################################
        self.pub_contactmap.publish(detect_msg)
        self.pub_contacts.publish(contact_msg)



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
        self.setup_contacts_for_detect(detect_info)

        contact_id = None
        if len(self.all_contacts) > 0:
            contact_id = self.check_all_contacts_by_BF(detect_info, data)

        if contact_id is None:
           contact_id = data.header.stamp


        #######################################################
        ####### CREATE OR UPDATE CONTACT WITH VARIABLES #######
        #######################################################

        if not contact_id in self.all_contacts:
            self.add_contact(contact_id, detect_info)
            self.all_contacts[contact_id].set_Z(detect_info)
        
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
            c.filter_bank.predict()
            c.filter_bank.update(c.Z)
            c.last_measured = detect_info['header'].stamp
        
            # Publish info about this detect and contact
            self.publish_msgs(c, detect_info)


        #######################################################
        # Append appropriate prior and measurements to lists. #
        #######################################################
        for cid in self.all_contacts:
            # This will be a little memory hungry, but it will capture the
            # entire state of all contacts at each time step. Needed to debug.
            self.all_contacts[cid].evalTime = detect_info['header'].stamp.to_sec()

            if cid not in self.all_contact_history:
                self.all_contact_history[cid] = []

            self.all_contact_history[cid].append(deepcopy(self.all_contacts[cid]))
            c = self.all_contacts[cid]
            
            # Iterate over all contacts associated with the measurement.
            if cid == contact_id:
                c.xs.append(np.array([c.filter_bank.x[0], c.filter_bank.x[1]]))
                c.zs.append(np.array([c.info['x_pos'], c.info['y_pos']]))
                c.ps.append(c.filter_bank.P)
            
            else:
                # For contacts not associated with the measurement, capture their
                # predicted location for the measurement time. This is calcualted
                # and stored in the "prior" parameters of the kalman filter.
                # Capture only the values for the 1st order filter for simplicity.
                # Why 1st and not 2nd? dunno.
                c.xs.append(np.array([c.all_filters[0].x_prior[0], c.all_filters[0].x[1]]))
                #c.xs.append(np.array([c.filter_bank.filters[0].x_prior[0], c.filter_bank.filters[0].x[1]]))
                c.zs.append(np.array([np.nan, np.nan]))
                c.ps.append(c.all_filters[0].P_prior)


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
