import rospy
from std_msgs.msg import String

from filterpy.kalman import KalmanFilter
from filterpy.kalman import update
from filterpy.kalman import predict


class Contact:
    """
    Class to create contact object to update the Kalman filter.
    """

    def __init__(self, x0, kf, timestamp, hash_key):
        """
        Define the constructor.
        """

        self.x0 = x0
        self.kf = kf
        self.timestamp = timestamp
        self.last_accessed = timestamp 
        self.hash_key = hash_key
       '''self.kf.x = np.array([]) # prior
       self.kf.P =                 # covariance matrix
       self.kf.Q =                 # noise covariance 
       self.kf.F = np.array([])    # process model matrix 
       self.kf.Q = 0               # noise
       self.kf.dt = '''             



