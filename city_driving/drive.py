#!/usr/bin/env python2
import rospy
import numpy as np
from visual_servoing.msg import ConeLocation, ParkingError
from ackermann_msgs.msg import AckermannDriveStamped
from std_msgs.msg import Float32
import time

class Drive:
    def __init__(self):
        DRIVE_TOPIC = rospy.get_param("~drive_topic") # set in launch file; different for simulator vs racecar
        self.drive_pub = rospy.Publisher(DRIVE_TOPIC, AckermannDriveStamped, queue_size=10)
        self.error_pub = rospy.Publisher("/parking_error", ParkingError, queue_size=10)
        self.stop_sub = rospy.Subscriber("/stop_sign_distance", Float32, self.stop_callback)
        self.cone_sub = rospy.Subscriber("/relative_cone", ConeLocation, self.relative_cone_callback)
        self.drive_message = AckermannDriveStamped()

        self.parking_distance = .3 # meters; try playing with this number!
        self.relative_x = 0
        self.relative_y = 0

        self.direction = 1
        self.velocity = 1

        self.angle_tolerance = 5.0*np.pi/180.0 #5 degrees in rad 
        self.distance_tolerance = 0.1

        self.last_time = None
        self.prev_dist_err = 0
        self.prev_ang = 0

        self.P = 1.2
        self.I = 0.5
        self.D = 0.3
        self.I_sum = 0

        self.P_ang = 1
        self.D_ang = 2

        self.slow = 0.2
        self.avg = 0.5
        self.fast = 1

        self.stop_signal = 0
        self.prev_time = time.time() # For PID controller
        self.stop_time = time.time() # Measures time to stop at sign

        # pure pursuit params:
        self.speed              = rospy.get_param("~speed", 1.0)
        self.wheelbase_length   = rospy.get_param("~wheelbase_length", 0.3)
        self.small_angle        = rospy.get_param("~small_steering_angle", 0.01)
        self.point_car          = np.array([0, 0])
        self.car_unit_vec       = np.array([1, 0])

    def stop_callback(self, msg):
        """ Classes:
            0: Keep going
            1: Stop 
            2: Resume driving
        """
        distance = msg.data
        if self.stop_signal == 1:
            curr_time = time.time()
            if curr_time - self.stop_time > 1:
                self.stop_signal = 2
        else:
            if distance == -1:
                self.stop_signal = 0
            else:
                if self.stop_signal != 2:
                    self.stop_signal = 1
                    self.stop_time = time.time()    
                               
            # if distance > 1:
            #     self.stop_signal = 0
            # elif (distance > 0.75 or distance < 1):
            #     if self.stop_signal != 2:
            #         self.stop_signal = 1
            #         self.stop_time = time.time()
            # elif distance < 0.75:
            #     self.stop_signal = 2

    def relative_cone_callback(self, msg):
        self.relative_x = msg.x_pos
        self.relative_y = msg.y_pos
        levi = AckermannDriveStamped()

        #calculate distance and angle to cone
        dist = np.sqrt( np.square(self.relative_x) + np.square(self.relative_y))
        angle = np.arctan2(self.relative_y, self.relative_x)

        #calculate and filter distance error
        dist_err = self.low_pass_filter(self.prev_dist_err, dist - self.parking_distance)
        
        #timing stuff
        current_time = rospy.get_time()
        if self.last_time is None: self.last_time = current_time
        delta_t = current_time - self.last_time

        #update net error
        if abs(dist_err) > self.distance_tolerance:
            self.I_sum += dist_err*delta_t 
        if self.I_sum > self.velocity:
            self.I_sum = self.velocity
        elif self.I_sum < -self.velocity:
            self.I_sum = -self.velocity

        #update drive message
        # use PID
        #levi.drive.speed, levi.drive.steering_angle = self.controller(dist_err, angle, delta_t)
        
        # use pure pursuit
        if self.stop_signal != 1:
            levi.drive.speed, levi.drive.steering_angle = self.pure_pursuit(np.array([self.relative_x, self.relative_y]))
        else:
            rospy.loginfo("stop for stop sign!")
            levi.drive.speed, levi.drive.steering_angle = (0, 0) #stop for stop sign

        self.prev_dist_err = dist_err
        self.last_time = current_time
        levi.drive.acceleration = 0
        # levi.drive.steering_angle_velocity = 0.5
        levi.drive.jerk = 0.1
        levi.header.stamp = rospy.Time.now()
        self.drive_pub.publish(levi)
        self.error_publisher()

    def error_publisher(self):
        """
        Publish the error between the car and the cone. We will view this
        with rqt_plot to plot the success of the controller
        """
        error_msg = ParkingError()
        error_msg.y_error = self.relative_y
        error_msg.x_error = self.relative_x - self.parking_distance
        error_msg.distance_error = np.sqrt(np.square(self.relative_y) + np.square(self.relative_x)) - self.parking_distance
        self.error_pub.publish(error_msg)

    def low_pass_filter(self, prev_val, cur_val, alpha=0.5):
        return alpha*cur_val + (1-alpha)*prev_val

    def controller(self, dist_err, angle, delta_t):
        P_ang = self.P_ang*angle
        D_ang = 0
        #case where we're near right distance but wrong angle
        if abs(dist_err) < self.distance_tolerance + 0.75  and abs(angle) > self.angle_tolerance:

            #if we're too close or angle is way off, reverse
            if dist_err < 0 or (dist_err < self.distance_tolerance + 0.25 and abs(angle) > 20.0*np.pi/180.0): 
                self.direction = -1

            #if cone is behind car, go forward
            if abs(angle) > np.pi*2./3.:
                self.direction = 1

            if delta_t > 0:
                D_ang = self.D_ang*(angle - self.prev_ang)/delta_t

            return (self.direction*0.5*self.velocity, self.direction*(P_ang + D_ang))    

        #case where we are parked
        elif abs(dist_err) < self.distance_tolerance:
            return (0, 0)

        #case where distance and/or angle are off
        else:
            self.direction = 1 if dist_err > 0 else -1
            P_err = self.P*dist_err
            I_err = self.I*self.I_sum
            D_err = 0
            if delta_t > 0:
                D_err = self.D*(dist_err - self.prev_dist_err)/delta_t
                D_ang = self.D_ang*(angle - self.prev_ang)/delta_t
            
            speed = self.direction*abs(P_err + I_err + D_err)
            if speed > 1: speed = 1
            elif speed < -1: speed = -1

            steering_angle = self.direction*(P_ang + D_ang) if abs(angle) > self.angle_tolerance else 0

            return (speed, steering_angle)

    def pure_pursuit(self, lookahead):
        if lookahead[0] == -1 and lookahead[1] == -1:
            rospy.loginfo("go back")
            return (-self.speed, 0.0)

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
        steer_ang = alpha
        if steer_ang > 0.25: steer_ang = np.pi
        steer_ang = abs(steer_ang) if lookahead[1] >= 0 else -abs(steer_ang)
        # rospy.loginfo(steer_ang)
        
        return (self.speed, steer_ang)

    def drive_controller(self):
        if self.stop_signal == 0:
            self.create_message(self.avg, self.steering_angle)
            self.drive_pub.publish(self.drive_message)
        elif self.stop_signal == 1:
            self.create_message(0, 0)
            self.drive_pub.publish(self.drive_message)
        elif self.stop_signal == 2:
            self.create_message(self.slow, self.steering_angle)
            self.drive_pub.publish(self.drive_message)

    def create_message(self, velocity, steering_angle):
        self.drive_message.header.stamp = rospy.Time.now()
        self.drive_message.header.frame_id = 'map'
        self.drive_message.drive.steering_angle = steering_angle
        self.drive_message.drive.steering_angle_velocity = 0
        self.drive_message.drive.speed = velocity
        self.drive_message.drive.acceleration = 0
        self.drive_message.drive.jerk = 0

if __name__=="__main__":
    rospy.init_node("driver")
    driver = Drive()
    rospy.spin()
