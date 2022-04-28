#!/usr/bin/env python

import numpy as np
import rospy

import cv2 as cv
from cv_bridge import CvBridge, CvBridgeError

from sensor_msgs.msg import Image
from final_challenge_2022.msg import Lookahead
from homography_transformer import HomographyTransformer

LEFT_COLOR = (184, 87, 0)
RIGHT_COLOR = (0, 221, 254)


class TrackDetector():
    ELEMENT = cv.getStructuringElement(cv.MORPH_CROSS, (3, 3))

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
        max_lookahead = rospy.get_param("~max_lookahead")
        self.homography_transformer = HomographyTransformer(pts_image_plane, pts_ground_plane)
        self.pt_left_uv = self.homography_transformer.transform_xy_to_uv(self.pt_left)
        self.pt_right_uv = self.homography_transformer.transform_xy_to_uv(self.pt_right)
        self.max_lookahead_uv = self.homography_transformer.transform_xy_to_uv((max_lookahead, 0))
        self.lp_factor = rospy.get_param("~lp_factor")

        self.lookahead_msg = Lookahead()

        # Subscribe to ZED camera RGB frames
        if self.send_debug:
            rospy.loginfo("Track detector with debug")
            self.debug_pub = rospy.Publisher(debug_topic, Image, queue_size=1)
        self.track_pub = rospy.Publisher(track_topic, Lookahead, queue_size=1)
        self.image_sub = rospy.Subscriber(image_topic, Image, self.image_callback)
        self.bridge = CvBridge() # Converts between ROS images and OpenCV Images

    def image_callback(self, image_msg):
        image = self.bridge.imgmsg_to_cv2(image_msg, "bgr8")
        height, width, channels = image.shape
        cv.rectangle(image, (0,height), (width, height-150), (0,0,0), -1)
        max_lookahead_line = [(0, self.max_lookahead_uv[1]),(width, self.max_lookahead_uv[1])]

        lines, mask = TrackDetector.__get_hough_lines(image, self.lower_bound, self.upper_bound, cv.COLOR_BGR2HLS)

        # if self.send_debug:
        #     image = cv.bitwise_and(image, image, mask=mask)

        left_line, right_line = self.__get_track(lines, image)

        left_lookahead_intersect = None
        right_lookahead_intersect = None
        left_right_intersect = None
        intersect_u = None
        intersect_v = None
        if left_line is not None:
#            rospy.loginfo("see left")
            left_lookahead_intersect = self.__get_intersect(left_line, max_lookahead_line)
            intersect_u, intersect_v = left_lookahead_intersect
            if self.send_debug:
                TrackDetector.__draw_point((intersect_u, intersect_v), image, LEFT_COLOR)
                TrackDetector.__draw_line(left_line, image, LEFT_COLOR)
        if right_line is not None:
#            rospy.loginfo("see right")
            right_lookahead_intersect = self.__get_intersect(right_line, max_lookahead_line)
            intersect_u, intersect_v = right_lookahead_intersect
            if self.send_debug:
                TrackDetector.__draw_point((intersect_u, intersect_v), image, RIGHT_COLOR)
                TrackDetector.__draw_line(right_line, image, RIGHT_COLOR)
        if left_line and right_line is not None:
            left_right_intersect = self.__get_intersect(right_line, left_line)
            if left_right_intersect[1] < left_lookahead_intersect[1]:
                intersect_u, intersect_v = left_right_intersect
            else:
                intersect_u = (left_lookahead_intersect[0] + right_lookahead_intersect[0])/2
                intersect_v = self.max_lookahead_uv[1]

        if intersect_u is not None:
            x, y = self.homography_transformer.transform_uv_to_xy((intersect_u, intersect_v))
            self.lookahead_msg.x = TrackDetector.__lp(self.lookahead_msg.x, x, self.lp_factor)
            self.lookahead_msg.y = TrackDetector.__lp(self.lookahead_msg.y, y, self.lp_factor)
            if self.send_debug:
                TrackDetector.__draw_point((intersect_u, intersect_v), image, (255,0,255))

        if self.send_debug:
            TrackDetector.__draw_point(self.pt_left_uv, image, LEFT_COLOR)
            TrackDetector.__draw_point(self.pt_right_uv, image, RIGHT_COLOR)
            TrackDetector.__draw_line(max_lookahead_line, image, (0,0,255))
            debug_msg = self.bridge.cv2_to_imgmsg(image, "bgr8")
            self.debug_pub.publish(debug_msg)

        self.track_pub.publish(self.lookahead_msg)
    
    @staticmethod
    def __lp(value, new_value, factor):
        return value * (1 - factor) + new_value * (factor)
    
    @staticmethod
    def __get_slope_intercept(line):
        delta = (np.array(line[1], dtype=float) - np.array(line[0], dtype=float))
        slope = delta[1]/delta[0]
        intercept = line[0][1] - line[0][0]*slope
        return slope, intercept

    @classmethod
    def __get_intersect(cls, line1, line2):
        slope_1, intercept_1 = cls.__get_slope_intercept(line1)
        slope_2, intercept_2 = cls.__get_slope_intercept(line2)
        u = (intercept_2-intercept_1)/(slope_1-slope_2)
        v = u*slope_1 + intercept_1
        return u, v

    @staticmethod
    def __draw_point(point, image, color):
        try:
            point = np.array(point, dtype=int)
        except:
            rospy.loginfo(point)
        cv.circle(image, tuple(point), 4, color, -1)
    
    @staticmethod
    def __draw_line(line, image, color):
        line = np.array(line, dtype=int)
        cv.line(image, tuple(line[0]), tuple(line[1]), color, 2, cv.LINE_AA)

    @staticmethod
    def __track_update(line, p, best_line, best_dist):
        p1_xy, p2_xy = line
        new_dist = TrackDetector.__min_distance(p1_xy, p2_xy, p)
        # rospy.loginfo("dist: %f" % new_dist)
        if best_dist is None or (new_dist < best_dist):
            return line, new_dist
        return best_line, best_dist
    
    def __get_track(self, lines, image=None):
        best_dist_left = None
        best_line_left = None

        best_dist_right = None
        best_line_right = None

        if lines is not None:
            colors = [(0,0,255), (0,255,0), (255,0,0)]
            j = 0
            for line in lines:
                p1, p2 = np.reshape(line[0], (2,2))
                delta = (np.array(p2, dtype = float) - np.array(p1, dtype = float))
                slope = delta[1]/delta[0]
                intercept = p1[1] - p1[0]*slope
                line = [p1, p2]
                rospy.loginfo("slope: %f, intercept: %f" % (slope, intercept))
                if np.abs(slope) > 0:
                    left_x = (self.pt_left_uv[1]-intercept)/slope
                    if left_x > self.pt_left_uv[0]:
#                        rospy.loginfo("do left")
                        best_line_left, best_dist_left = TrackDetector.__track_update(line, 
                                                                                      self.pt_left_uv, 
                                                                                      best_line_left, 
                                                                                      best_dist_left)
                    right_x = (self.pt_right_uv[1]-intercept)/slope
                    if right_x < self.pt_right_uv[0]:
#                        rospy.loginfo("do right")
                        best_line_right, best_dist_right = TrackDetector.__track_update(line, 
                                                                                        self.pt_right_uv, 
                                                                                        best_line_right, 
                                                                                        best_dist_right)
                if self.send_debug:
                    cv.line(image, tuple(p1), tuple(p2), colors[j%3], 3, cv.LINE_AA)
                    j += 1
        # rospy.loginfo("best: %f" % best_dist_left)
        # rospy.loginfo("-----------")
        return best_line_left, best_line_right

    # Adapted from http://felix.abecassis.me/2011/09/opencv-morphological-skeleton/
    @staticmethod
    def __get_outline_image(image, lower_bound, upper_bound, code=None):
        edge_image = image.copy()
        if code is not None:
            edge_image = cv.cvtColor(edge_image, code)
        
        mask = cv.inRange(edge_image, lower_bound, upper_bound)
        mask = cv.dilate(mask, TrackDetector.ELEMENT)

        skel = np.zeros(mask.shape, dtype=np.uint8)
        while(cv.countNonZero(mask) != 0):
            eroded = cv.erode(mask, TrackDetector.ELEMENT)
            temp = cv.dilate(eroded, TrackDetector.ELEMENT)
            temp = cv.subtract(mask, temp)
            skel = cv.bitwise_or(skel, temp)
            mask = eroded

        return skel

    @staticmethod
    def __get_hough_lines(image, lower_bound, upper_bound, code=None):
        dst = TrackDetector.__get_outline_image(image, lower_bound, upper_bound, code)
        return cv.HoughLinesP(dst, 1, np.pi / 720, 100, None, 20, 10), dst

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
