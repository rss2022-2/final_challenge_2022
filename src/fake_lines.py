#!/usr/bin/env python

from final_challenge_2022.msg import TrackLane
import rospy


class FakeLines():
    """
    publish fake line to test track pursuit
    """
    def __init__(self):
        self.track_topic    = rospy.get_param("~track_topic", "/track_line")
        self.lines_pub      = rospy.Publisher(self.track_topic, TrackLane, queue_size=10)

        rospy.Timer(rospy.Duration(1.0/20.0), self.send_lines)

    def send_lines(self, event):
        lines_msg = TrackLane()
        lines_msg.slope_left = -0.5
        lines_msg.intercept_left = 3
        lines_msg.slope_right = 0.5
        lines_msg.intercept_right = -1
        self.lines_pub.publish(lines_msg)

if __name__=="__main__":
    rospy.init_node("fake_lines")
    fake_lines = FakeLines()
    rospy.spin()