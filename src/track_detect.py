#!/usr/bin/env python

import numpy as np
import rospy

import cv2 as cv
from cv_bridge import CvBridge, CvBridgeError

from sensor_msgs.msg import Image
from final_challenge_2022.msg import TrackLane
from homography_transformer import HomographyTransformer

LEFT_COLOR = (184, 87, 0)
RIGHT_COLOR = (0, 221, 254)


class TrackDetector():
    
    def __init__(self):
        image_topic = rospy.get_param("~image_topic")
        track_topic = rospy.get_param("~track_topic")
        debug_topic = rospy.get_param("~image_debug_topic")
        self.send_debug = rospy.get_param("~send_debug_image", False)
        pts_image_plane = rospy.get_param("~pts_image_plane")
        pts_ground_plane = rospy.get_param("~pts_ground_plane")
        self.lower_bound = np.array(rospy.get_param("~color_lower_bound"), np.uint8)
        self.upper_bound = np.array(rospy.get_param("~color_upper_bound"), np.uint8)
        self.pt_left = rospy.get_param("~pt_left")
        self.pt_right = rospy.get_param("~pt_right")

        # Subscribe to ZED camera RGB frames
        if self.send_debug:
            rospy.loginfo("Track detector with debug")
            self.debug_pub = rospy.Publisher(debug_topic, Image, queue_size=10)
        self.track_pub = rospy.Publisher(track_topic, TrackLane, queue_size=10)
        self.image_sub = rospy.Subscriber(image_topic, Image, self.image_callback)
        self.bridge = CvBridge() # Converts between ROS images and OpenCV Images
        self.homography_transformer = HomographyTransformer(pts_image_plane, pts_ground_plane)

    def image_callback(self, image_msg):
        image = self.bridge.imgmsg_to_cv2(image_msg, "bgr8")
        height, width, channels = image.shape
        cv.rectangle(image, (0,height), (width, height-150), (0,0,0), -1)
        
        lines, mask = TrackDetector.__get_hough_lines(image, self.lower_bound, self.upper_bound, cv.COLOR_BGR2HLS)

        if self.send_debug:
            image = cv.bitwise_and(image, image, mask=mask)

        left_line, right_line = self.__get_track(lines, image)

        track_msg = TrackLane()

        if left_line is not None:
            track_msg.slope_left, track_msg.intercept_left = TrackDetector.__get_slope_intercept(left_line)
            if self.send_debug:
                rospy.loginfo("Left line detected!")
                self.__draw_xy_line(left_line, image, LEFT_COLOR)
                self.__draw_xy_point(self.pt_left, image, LEFT_COLOR)
        
        if right_line is not None:
            track_msg.slope_right, track_msg.intercept_right = TrackDetector.__get_slope_intercept(right_line)
            if self.send_debug:
                rospy.loginfo("Right line detected!")
                self.__draw_xy_line(right_line, image, RIGHT_COLOR)
                self.__draw_xy_point(self.pt_right, image, RIGHT_COLOR)

        if self.send_debug:
            debug_msg = self.bridge.cv2_to_imgmsg(image, "bgr8")
            self.debug_pub.publish(debug_msg)

        self.track_pub.publish(track_msg)
        
    
    @staticmethod
    def __get_slope_intercept(line):
        delta = (np.array(line[1]) - np.array(line[0]))
        slope = delta[1]/delta[0]
        intercept = line[0][1] - line[0][0]*slope
        return slope, intercept


    def __draw_xy_point(self, point, image, color):
        p_uv = self.homography_transformer.transform_xy_to_uv(point)
        cv.circle(image, p_uv, 10, color, -1)

    def __draw_xy_line(self, line, image, color):
        p1_uv = self.homography_transformer.transform_xy_to_uv(line[0])
        p2_uv = self.homography_transformer.transform_xy_to_uv(line[1])
        cv.line(image, p1_uv, p2_uv, color, 5, cv.LINE_AA)

    @staticmethod
    def __track_update(line, p, best_line, best_dist):
        p1_xy, p2_xy = line


        new_dist = TrackDetector.__min_distance(p1_xy, p2_xy, p)

        if best_dist is None or (new_dist < best_dist and new_dist < 1.0):
            return line, new_dist
        return best_line, best_dist
    
    def __get_track(self, lines, image=None):
        best_dist_left = None
        best_line_left = None

        best_dist_right = None
        best_line_right = None

        if lines is not None:
            for line in lines:
                p1_u, p1_v, p2_u, p2_v = line[0]
                p1_xy = self.homography_transformer.transform_uv_to_xy([p1_u, p1_v])
                p2_xy = self.homography_transformer.transform_uv_to_xy([p2_u, p2_v])
                delta = (np.array(p2_xy) - np.array(p1_xy))
                slope = delta[1]/delta[0]
                intercept = p1_xy[1] - p1_xy[0]*slope
                line = [p1_xy, p2_xy]
                if np.abs(slope) < 0.5:
                    left_y = self.pt_left[0]*slope + intercept
                    if left_y > self.pt_left[1]:
                        best_line_left, best_dist_left = TrackDetector.__track_update(line, 
                                                                                      self.pt_left, 
                                                                                      best_line_left, 
                                                                                      best_dist_left)
                    right_y = self.pt_right[0]*slope + intercept
                    if right_y < self.pt_right[1]:
                        best_line_right, best_dist_right = TrackDetector.__track_update(line, 
                                                                                        self.pt_right, 
                                                                                        best_line_right, 
                                                                                        best_dist_right)
                if self.send_debug:
                    cv.line(image, [p1_u, p1_v], [p2_u, p2_v], (0,0,255), 3, cv.LINE_AA)
        
        return best_line_left, best_line_right

    @staticmethod
    def __get_outline_image(image, lower_bound, upper_bound, code=None):
        edge_image = image.copy()
        if code is not None:
            edge_image = cv.cvtColor(edge_image, code)
        
        edge_image = cv.inRange(edge_image, lower_bound, upper_bound)
        edge_image = cv.dilate(
            edge_image,
            cv.getStructuringElement(cv.MORPH_RECT, (6, 6)),
            iterations=1
        )
        edge_image = cv.GaussianBlur(edge_image, (51, 51), 1)
        edge_image = cv.Canny(edge_image, 50, 150, L2gradient=True)
        return edge_image

    @staticmethod
    def __get_hough_lines(image, lower_bound, upper_bound, code=None):
        dst = TrackDetector.__get_outline_image(image, lower_bound, upper_bound, code)
        return cv.HoughLinesP(dst, 1, np.pi / 720.0, 100, None, 50, 10), dst

    #Adapted from https://www.geeksforgeeks.org/minimum-distance-from-a-point-to-the-line-segment-using-vectors/
    @staticmethod
    def __min_distance(A, B, E):
        # vector AB
        AB = [None, None]
        AB[0] = B[0] - A[0]
        AB[1] = B[1] - A[1]
    
        # vector BP
        BE = [None, None]
        BE[0] = E[0] - B[0]
        BE[1] = E[1] - B[1]
    
        # vector AP
        AE = [None, None]
        AE[0] = E[0] - A[0]
        AE[1] = E[1] - A[1]

        # Variables to store dot product
    
        # Calculating the dot product
        AB_BE = AB[0] * BE[0] + AB[1] * BE[1]
        AB_AE = AB[0] * AE[0] + AB[1] * AE[1]
    
        # Minimum distance from
        # point E to the line segment
        reqAns = 0
    
        # Case 1
        if (AB_BE > 0) :
    
            # Finding the magnitude
            y = E[1] - B[1]
            x = E[0] - B[0]
            reqAns = np.sqrt(x * x + y * y)
    
        # Case 2
        elif (AB_AE < 0) :
            y = E[1] - A[1]
            x = E[0] - A[0]
            reqAns = np.sqrt(x * x + y * y)
    
        # Case 3
        else:
    
            # Finding the perpendicular distance
            x1 = AB[0]
            y1 = AB[1]
            x2 = AE[0]
            y2 = AE[1]
            mod = np.sqrt(x1 * x1 + y1 * y1)
            reqAns = abs(x1 * y2 - y1 * x2) / mod
        
        return reqAns

if __name__ == '__main__':
    try:
        rospy.init_node('track_detector')
        TrackDetector()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
