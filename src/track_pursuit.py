#!/usr/bin/env python

import rospy
import numpy as np
import time
import tf

from geometry_msgs.msg import PoseArray, PoseStamped
from visualization_msgs.msg import Marker
from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import Odometry
from tf.transformations import euler_from_quaternion
from visualization_tools import *
from final_challenge_2022.msg import TrackLane

class TrackPursuit(object):
    """ Implements Pure Pursuit trajectory tracking with a fixed lookahead and speed.
    """
    def __init__(self):
        self.speed              = rospy.get_param("~speed", 1.0)
        self.fast_speed         = rospy.get_param("~fast_speed", 1.0)
        self.wheelbase_length   = rospy.get_param("~wheelbase_length", 0.3)
        self.small_angle        = rospy.get_param("~small_steering_angle", 0.01)
        self.drive_topic        = rospy.get_param("~drive_topic", "/drive")
        self.track_topic        = rospy.get_param("~track_topic", "/track_line")
        self.visual_topic       = rospy.get_param("~visual_topic", "/track_lane_visualizer")
        self.draw_lines         = rospy.get_param("~draw_lines", True)
        self.front_point        = 1 # look at 3 meters ahead
        self.half_track_width   = 0.83 / 2.0
        self.GAIN_P             = 1
        self.WEIGHT_DISTANCE    = 0.7
        self.WEIGHT_ANGLE       = 0.3

        self.left_cam_offset    = 0.14
        # publish drive commands
        drive_msg = AckermannDriveStamped()
        drive_msg.drive.acceleration = 0
        drive_msg.drive.steering_angle_velocity = 0
        drive_msg.drive.jerk = 0
        self.drive_msg = drive_msg

        self.drive_pub = rospy.Publisher(self.drive_topic, AckermannDriveStamped, queue_size=1)
        self.line_pub = rospy.Publisher(self.visual_topic, Marker, queue_size=1)

        #TODO: fill details
        self.track_sub = rospy.Subscriber(self.track_topic, TrackLane, self.track_callback, queue_size=10)

        self.point_car          = np.array([0, 0])
        self.car_unit_vec       = np.array([1, 0])

        rospy.Timer(rospy.Duration(1.0/20.0), self.send_cmd)

    def send_cmd(self, event):
        self.drive_msg.header.stamp = rospy.Time.now()
        self.drive_pub.publish(self.drive_msg)

    def track_callback(self, track_msg):
        ''' Get 2 lines represent the track lane
        '''
        m_1    = track_msg.slope_left
        b_1    = track_msg.intercept_left
        m_2    = track_msg.slope_right
        b_2    = track_msg.intercept_right
        # assert m_1 != m_2, "2 lines are parallel, cannot happen"

        if self.draw_lines:
            self.__draw_line(m_1, b_1, self.line_pub)
            self.__draw_line(m_2, b_2, self.line_pub)

        # find intersection of 2 line
        # x = (b_2 - b_1) / (m_1 - m_2)
        # y = m_1*x + b_1
        # lookahead = np.array([x, y])

        # x = self.front_point
        # if not np.isnan(m_1) and not np.isnan(m_2):
        #     side = 1
        #     # y = (b_1 + b_2) / 2.0
        #     distance_error = abs(b_1 - self.left_cam_offset) - self.half_track_width
        #     angle_error = np.arctan(m_1) if m_1 != 0 else 0
        #     rospy.loginfo("See both")
        #     rospy.loginfo([m_1, b_1])
        #     rospy.loginfo("distance error")
        #     rospy.loginfo(distance_error)
        #     rospy.loginfo("angel error")
        #     rospy.loginfo(angle_error)
        # elif not np.isnan(m_1):
        #     side = 1
        #     # y = b_1 - self.half_track_width
        #     distance_error = abs(b_1 - self.left_cam_offset) - self.half_track_width
        #     angle_error = np.arctan(m_1) if m_1 != 0 else 0
        #     rospy.loginfo("See only left")
        #     rospy.loginfo([m_1, b_1])
        # elif not np.isnan(m_2):
        #     side = -1
        #     # y = self.half_track_width + b_2
        #     distance_error = self.half_track_width - abs(b_2 + self.left_cam_offset)
        #     angle_error = np.arctan(m_2) if m_2 != 0 else 0
        #     rospy.loginfo("See only right")
        #     rospy.loginfo([m_2, b_2])
        # else:
        #     rospy.logerr("did not get any track lines")
        #     return
        
        # total_error = angle_error*self.WEIGHT_ANGLE + distance_error*self.WEIGHT_DISTANCE

        # pid = total_error * self.GAIN_P


        x = self.front_point
        if not np.isnan(m_1) and not np.isnan(m_2):
            y = (b_1 + b_2) / 2.0
        elif not np.isnan(m_1):
            y = b_1 - self.half_track_width
        elif not np.isnan(m_2):
            y = self.half_track_width + b_2
        else:
            rospy.logerr("did not get any track lines")
            return
        
        lookahead = np.array([x, y])


        ## find distance between car and lookahead
        lookahead_vec = lookahead - self.point_car
        distance = np.linalg.norm(lookahead_vec)
        
        ## find alpha: angle of the car to lookahead point
        lookahead_unit_vec = lookahead_vec / distance
        dot_product = np.dot(self.car_unit_vec, lookahead_unit_vec)
        dot_product = max(-1, dot_product) if dot_product < 0 else min(1, dot_product)
        assert -1 <= dot_product <= 1, dot_product
        alpha = np.arccos(dot_product)

        # steering angle
        steer_ang = np.arctan(2*self.wheelbase_length*np.sin(alpha)
                        / (distance))
        steer_ang = steer_ang if y >= 0 else -steer_ang

        # publish drive commands
        self.drive_msg = AckermannDriveStamped()
        # optimization: run fast if steer_ang is small
        self.drive_msg.drive.speed = self.fast_speed if abs(steer_ang) <= self.small_angle else self.speed
        self.drive_msg.drive.steering_angle = steer_ang

        # self.drive_msg.drive.speed = self.speed
        # self.drive_msg.drive.steering_angle = pid if (-0.34 <= pid <= 0.34) else -0.34 if pid <= 0 else 0.34
        
    @staticmethod
    def __draw_line(slope, y_intercept, publisher, frame = "/base_link"):
        x = np.linspace(0, 5, num=20)
        y = slope*x + y_intercept
        VisualizationTools.plot_line(x, y, publisher, frame=frame)

    
if __name__=="__main__":
    rospy.init_node("track_pursuit")
    pf = TrackPursuit()
    rospy.spin()
