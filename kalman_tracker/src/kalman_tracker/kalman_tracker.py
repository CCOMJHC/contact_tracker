import rospy
from std_msgs.msg import String

class kalman_tracker:
    
    def __init__(self):
        pass

    def init_service(self):
        rospy.init_node('tracker.py')
        ROSINFO('starting up the kalman_tracker')

    def run(self):
        rospy.spin() 

