import rospy
from std_msgs.msg import String


class Contact:
    """
    Class to create contact object to update the Kalman filter.
    """

    def __init__(self, x0, R, z, timestamp, hash_key):
        """
        Define the constructor.
        """

        self.x0 = x0
        self.R = R
        self.z = z
        self.timestamp = timestamp
        self.last_accessed = timestamp 
        self.hash_key = hash_key


    def detect_is_contact(all_contacts, hash_key):
        """
        Returns true if this contact is already in the dictionary, 
        false otherwise.
        """

        if hash_key in all_contacts:
            return true
        
        return false

