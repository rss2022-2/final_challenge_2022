import cv2
import rospy

import numpy as np
from sensor_msgs.msg import Image
from detector import StopSignDetector
from std_msgs.msg import Float32

class SignDetector:
    def __init__(self):
        self.detector = StopSignDetector(threshold=0)
        self.publisher = rospy.Publisher("/stop_sign_distance", Float32, queue_size=10)
        self.subscriber = rospy.Subscriber("/zed/zed_node/rgb/image_rect_color", Image, self.callback)
        self.bbox = [0, 0, 0, 0]

    def callback(self, img_msg):
        np_img = np.frombuffer(img_msg.data, dtype=np.uint8).reshape(img_msg.height, img_msg.width, -1)
        bgr_img = np_img[:,:,:-1]
        rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)

        present, bbox = self.detector.predict(rgb_img)
        if not present:
            self.bbox = [0, 0, 0, 0]
            self.publisher.publish(100)
        else:
            self.bbox = bbox

if __name__=="__main__":
    rospy.init_node("stop_sign_detector")
    detect = SignDetector()
    rospy.spin()
