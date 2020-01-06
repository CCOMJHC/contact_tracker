import rospy
from std_msgs.msg import String

from filterpy.kalman import KalmanFilter
from filterpy.kalman import update
from filterpy.kalman import predict


class Contact:
    """
    Class to create contact object with its own KalmanFilter.
    """

    def __init__(self, detect_info, kf, timestamp, contact_id):
        """
        Define the constructor.
        """

        self.info = detect_info
        self.kf = kf 
        self.last_accessed = timestamp 
        self.id = contact_id


    def init_kf(self, dt):
        """
        Initialize the kalman filter for this contact.
        """

       # Define the state variable vector
       self.kf.x = np.array([self.info['x_pos'], self.info['y_pos']]) 
       
       # Define the state covariance matrix
       self.kf.P = np.array([self.info['pos_cov'][0], 0, 
                             0, self.info['pos_cov'][7])]
 
       # Define the noise covariance (TBD)
       self.kf.Q = 0             

       # Define the process model matrix
       fu = np.array([1, dt, 
                      0, 1)]
       self.kf.F = self.kf.x.dot(fu)    

       # Define the measurement function
       self.kf.H = np.array([1, 0,
                             0, 1)]

       # Define the measurement covariance
       self.kf.R = 


    def init_kf_with_velocity(self, dt):
        """
        Initialize the kalman filter for this contact.
        """

       # Define the state variable vector
       self.kf.x = np.array([self.info['x_pos'], self.info['y_pos'], self.info['x_vel'], self.info['y_vel']]) 
       
       # Define the state covariance matrix
       self.kf.P = np.array([self.info['pos_cov'][0], 0, 0, 0,
                             0, self.info['pos_cov'][7], 0 , 0,
                             0, 0, self.info['twist_cov'][0], 0,
                             0, 0, 0, self.info['twist_cov'][7]
                             )]    
 
       # Define the noise covariance (TBD)
       self.kf.Q = 0             

       # Define the process model matrix
       fu = np.array([1, 0, dt, 0,
                      0, 1, 0, dt, 
                      0, 0, 1, 0,
                      0, 0, 0, 1])
       self.kf.F = self.kf.x.dot(fu)    

       # Define the measurement function
       self.kf.H = np.array([1, 0, 0, 0,
                             0, 1, 0, 0,
                             0, 0, 1, 0,
                             0, 0, 0, 1])

       # Define the measurement covariance
       self.kf.R = 


