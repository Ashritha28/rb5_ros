#!/usr/bin/env python
import sys
import rospy
from sensor_msgs.msg import Joy
from key_parser import get_key, save_terminal_settings, restore_terminal_settings
import math
import time
import csv

class KeyJoyNode:
    def __init__(self):
        self.pub_joy = rospy.Publisher("/joy", Joy, queue_size=1)
        self.settings = save_terminal_settings()
        self.current_pos = (0, 0, 0)
        self.current_theta = self.current_pos[-1]

        self.timestep = 0.02

        self.linear_speed = 0.80

        self.veh_lin_speed = 0.25  # m/s was 0.35
        self.veh_ang_speed = 1.20  # rad/s was 3.6

        # self.left_to_right_linear_ratio = 0.96

        self.angular_speed = 0.75

    def run(self, waypoints):
        # while True:
        #     # parse keyboard control
        #     key = get_key(self.settings, timeout=0.1)

        #     # interpret keyboard control as joy
        #     joy_msg, flag = self.key_to_joy(key)
        #     if flag is False:
        #         break
        #
        #     # publish joy
        #     self.pub_joy.publish(joy_msg)
        #waypoints = [(-1, 0, 0), (-1,1,1.57), (-2,1,0), (-2,2,-1.57), (-1,1,-0.78), (0,0,0)]
        for point in waypoints[1:]:
            key, angle = self.calculate_key_rot_initial(point)
            print(angle)
            if (key == 'q' or key == 'e'):
                duration = abs(angle) / self.veh_ang_speed
                print("ANGULAR DURATION ", duration)

                joy_msg, flag = self.key_to_joy(key)
                self.pub_joy.publish(joy_msg)
                # print(joy_msg)
                time.sleep(duration)
            #stop
            print("HELLO")
            joy_msg, flag = self.key_to_joy('u')
            self.pub_joy.publish(joy_msg)
            time.sleep(2.0)

            # self.stop()
            key, distance = self.calculate_key_trans(point)
            duration = distance / self.veh_lin_speed
            print("LINEAR DURATION : ", duration)

            joy_msg, flag = self.key_to_joy(key)
            self.pub_joy.publish(joy_msg)
            time.sleep(duration)

            joy_msg, flag = self.key_to_joy('u')
            self.pub_joy.publish(joy_msg)
            time.sleep(2.0)
            #
            key, angle = self.calculate_key_rot_final(point)
            print(angle)
            if (key == 'q' or key == 'e'):
                print("In the FINAL ROTATION")
                # duration = 0.07 + abs(angle) / self.veh_ang_speed
                duration = abs(angle) / self.veh_ang_speed
                print("ANGULAR DURATION ", duration)

                # # for i in range(steps):
                joy_msg, flag = self.key_to_joy(key)
                self.pub_joy.publish(joy_msg)
                time.sleep(duration)
            # self.stop()
            # stop
            joy_msg, flag = self.key_to_joy('u')
            self.pub_joy.publish(joy_msg)
            time.sleep(2.0)

            # self.stop()

    def calculate_key_rot_initial(self, dest_pos):
        pos = (dest_pos[0], dest_pos[1])
        theta_heading = math.atan2(pos[1] - self.current_pos[1], pos[0] - self.current_pos[0]) - self.current_pos[
            -1]
        print(theta_heading)
        if abs(theta_heading) > 0.01:
            self.current_theta += theta_heading
        print("Current Theta:", self.current_theta)
        #self.current_theta = dest_pos[-1]
        if theta_heading > 0.01:
            return 'e', theta_heading
        elif theta_heading < 0.01:
            return 'q', theta_heading
        else:
            return 'u', theta_heading

    def calculate_key_trans(self, dest_pos):
        dist = math.sqrt((dest_pos[1] - self.current_pos[1]) ** 2 + (dest_pos[0] - self.current_pos[0]) ** 2)
        return 'w', dist


    def calculate_key_rot_final(self, dest_pos):
        rel_angle = dest_pos[-1] - self.current_theta
        print("Current Theta:", self.current_theta)
        print(rel_angle)
        self.current_theta = dest_pos[-1]
        self.current_pos = dest_pos

        if rel_angle > 0.01:
            return 'e', rel_angle
        elif rel_angle < 0.01:
            return 'q', rel_angle
        else:
            return 'u', rel_angle

    def key_to_joy(self, key):
        flag = True
        joy_msg = Joy()
        joy_msg.axes = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        joy_msg.buttons = [0, 0, 0, 0, 0, 0, 0, 0]
        if key == 'w':
            joy_msg.axes[1] = 1.0
        elif key == 's':
            joy_msg.axes[1] = -1.0
        elif key == 'a':
            joy_msg.axes[0] = -1.0
        elif key == 'd':
            joy_msg.axes[0] = 1.0
        elif key == 'q':
            joy_msg.axes[2] = -1.0
        elif key == 'e':
            joy_msg.axes[2] = 1.0
        elif (len(key) > 0 and ord(key) == 27) or (key == '\x03'):
            flag = False
        return joy_msg, flag

    def stop(self):
        restore_terminal_settings(self.settings)
        
    def read_file(self):
        waypoints = []
        with open('waypoints.txt', 'r') as f:
            reader = csv.reader(f, quoting=csv.QUOTE_NONNUMERIC)
            for row in reader:
                waypoints.append(row)

        return waypoints

if __name__ == "__main__":
    key_joy_node = KeyJoyNode()
    rospy.init_node("key_joy")
    waypoints = key_joy_node.read_file()
    key_joy_node.run(waypoints)
