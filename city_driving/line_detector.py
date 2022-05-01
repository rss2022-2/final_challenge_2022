#!/usr/bin/env python

import numpy as np
import rospy

import cv2
from cv_bridge import CvBridge, CvBridgeError

from sensor_msgs.msg import Image
from geometry_msgs.msg import Point #geometry_msgs not in CMake file
from final_challenge_2022.msg import ConeLocationPixel

# import your color segmentation algorithm; call this function in ros_image_callback!
from computer_vision.color_segmentation import cd_color_segmentation


class LineDetector():
    """
    A class for applying your cone detection algorithms to the real robot.
    Subscribes to: /zed/zed_node/rgb/image_rect_color (Image) : the live RGB image from the onboard ZED camera.
    Publishes to: /relative_cone_px (ConeLocationPixel) : the coordinates of the cone in the image frame (units are pixels).
    """
    def __init__(self):
        # toggle line follower vs cone parker
        self.LineFollower = False

        # Subscribe to ZED camera RGB frames
        self.cone_pub = rospy.Publisher("/relative_cone_px", ConeLocationPixel, queue_size=10)
        self.debug_pub = rospy.Publisher("/cone_debug_img", Image, queue_size=10)
        self.image_sub = rospy.Subscriber("/zed/zed_node/rgb/image_rect_color", Image, self.image_callback)
        self.bridge = CvBridge() # Converts between ROS images and OpenCV Images

    def image_callback(self, image_msg):
        # Apply your imported color segmentation function (cd_color_segmentation) to the image msg here
        # From your bounding box, take the center pixel on the bottom
        # (We know this pixel corresponds to a point on the ground plane)
        # publish this pixel (u, v) to the /relative_cone_px topic; the homography transformer will
        # convert it to the car frame.

        #################################
        # YOUR CODE HERE
        # detect the cone and publish its
        # pixel location in the image.
        # vvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
        #################################

        # image = cv2.rotate(self.bridge.imgmsg_to_cv2(image_msg, "bgr8"),cv2.ROTATE_180)
        image = self.bridge.imgmsg_to_cv2(image_msg, "bgr8")

        height, width, channels = image.shape
	
        # image = cv2.rectangle(image, (0,0), (width, 100), (0,0,0), -1)
        


        try:
            bot_right, top_left = cd_color_segmentation(image, None)

            cone_loc_pixel_msg = ConeLocationPixel()

            cone_loc_pixel_msg.u = (bot_right[0] + top_left[0])/2
            cone_loc_pixel_msg.v = (bot_right[1] + top_left[1])/2
        
            self.cone_pub.publish(cone_loc_pixel_msg)
            
            cv2.rectangle(image, bot_right, top_left, (0,255,0),2)
            cv2.circle(image, (cone_loc_pixel_msg.u,cone_loc_pixel_msg.v), radius=5, color=(0, 0, 255), thickness=-1)
            debug_msg = self.bridge.cv2_to_imgmsg(image, "bgr8")
            self.debug_pub.publish(debug_msg)

        except:
            rospy.loginfo("No cone detected!")

if __name__ == '__main__':
    try:
        rospy.init_node('LineDetector', anonymous=True)
        LineDetector()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
