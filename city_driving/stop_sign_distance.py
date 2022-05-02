#!/usr/bin/env python

import rospy
from std_msgs.msg import Float32MultiArray
from final_challenge_2022.msg import StopSignDistance

class StopSignDistance():

    def __init__(self):
        self.distance_topic = "/stop_sign_distance"
        self.bbox_topic = "/stop_sign_bbox"

        self.bbox_sub = rospy.Subscriber(self.bbox_topic, Float32MultiArray, self.bbox_callback)
        self.distance_pub = rospy.Publisher(self.distance_topic, StopSignDistance, queue_size=10)

    def bbox_callback(self, msg):
        top_left_x = msg.data[0]
        top_left_y = msg.data[1]
        bot_right_x = msg.data[2]
        bot_right_y = msg.data[3]

        area = (bot_right_x - top_left_x)*(bot_right_y - top_left_y)
        see_stop_sign = StopSignDistance()
        if (3000 <= area <= 3500):
            see_stop_sign.distance = 1
        else:
            see_stop_sign.distance = -1
        
        self.distance_pub.publish(see_stop_sign)

if __name__=="__main__":
    rospy.init_node("stop_sign_distance")
    StopSignDistance = StopSignDistance()
    rospy.spin()