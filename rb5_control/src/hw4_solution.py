#!/usr/bin/env python
import sys
import roslib
import rospy
import geometry_msgs.msg
from geometry_msgs.msg import Twist
import numpy as np
import math
import tf
import tf2_ros
from tf.transformations import quaternion_matrix

"""
The class of the pid controller.
"""

rTc = np.asarray([[0, 0, 1, 0.05], [-1, 0, 0, 0.015], [0,-1,0, 0.15], [0,0,0,1]])
pose_ma = {8: np.asarray([[0, 0, 1, 2.05],[-1, 0, 0, 0.015], [0, -1, 0, 0.15], [0,0,0,1]]),
5: np.asarray([[0, 0, 1, 2.05],[-1, 0, 0, 0.015], [0, -1, 0, 0.15], [0,0,0,1]]),
0: np.asarray([[0, 0, 1, 2.05],[-1, 0, 0, 0.015], [0, -1, 0, 0.15], [0,0,0,1]]),
1: np.asarray([[0, 0, 1, 2.05],[-1, 0, 0, 0.015], [0, -1, 0, 0.15], [0,0,0,1]]),
2: np.asarray([[0, 0, 1, 2.05],[-1, 0, 0, 0.015], [0, -1, 0, 0.15], [0,0,0,1]]),
3: np.asarray([[0, 0, 1, 2.05], [-1, 0, 0, 0.015], [0, -1, 0, 0.15], [0, 0, 0, 1]]),
4: np.asarray([[0, 0, 1, 2.05], [-1, 0, 0, 0.015], [0, -1, 0, 0.15], [0, 0, 0, 1]]),
6: np.asarray([[0, 0, 1, 2.05],[-1, 0, 0, 0.015], [0, -1, 0, 0.15], [0,0,0,1]]),
7: np.asarray([[0, 0, 1, 2.05],[-1, 0, 0, 0.015], [0, -1, 0, 0.15], [0,0,0,1]]),
9: np.asarray([[0, 0, 1, 2.05],[-1, 0, 0, 0.015], [0, -1, 0, 0.15], [0,0,0,1]])}

class PIDcontroller:
    def __init__(self, Kp, Ki, Kd):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.target = None
        self.I = np.array([0.0, 0.0, 0.0])
        self.lastError = np.array([0.0, 0.0, 0.0])
        self.timestep = 0.1
        self.maximumValue = 0.02

    def setTarget(self, targetx, targety, targetw):
        """
        set the target pose.
        """
        self.I = np.array([0.0, 0.0, 0.0])
        self.lastError = np.array([0.0, 0.0, 0.0])
        self.target = np.array([targetx, targety, targetw])

    def setTarget(self, state):
        """
        set the target pose.
        """
        self.I = np.array([0.0, 0.0, 0.0])
        self.lastError = np.array([0.0, 0.0, 0.0])
        self.target = np.array(state)

    def getError(self, currentState, targetState):
        """
        return the different between two states
        """
        result = targetState - currentState
        result[2] = (result[2] + np.pi) % (2 * np.pi) - np.pi
        return result

    def setMaximumUpdate(self, mv):
        """
        set maximum velocity for stability.
        """
        self.maximumValue = mv

    def update(self, currentState):
        """
        calculate the update value on the state based on the error between current state and target state with PID.
        """
        e = self.getError(currentState, self.target)

        P = self.Kp * e
        self.I = self.I + self.Ki * e * self.timestep
        I = self.I
        D = self.Kd * (e - self.lastError)
        result = P + I + D

        self.lastError = e

        # scale down the twist if its norm is more than the maximum value.
        resultNorm = np.linalg.norm(result)
        if (resultNorm > self.maximumValue):
            result = (result / resultNorm) * self.maximumValue
            self.I = 0.0

        return result


def rotationMatrixToEulerAngles(R):
    assert (isRotationMatrix(R))

    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0
    # print(np.array([x, y, z]))
    return np.array([x, y, z])

def getCurrentPos(l):
    """
    Given the tf listener, we consider the camera's z-axis is the header of the car
    """
    br = tf.TransformBroadcaster()
    result = None
    foundSolution = False

    for i in range(0, 9):
        camera_name = "camera_" + str(i)
        if l.frameExists(camera_name):
            print("Trying camera", camera_name)
            try:
                now = rospy.Time()
                # wait for the transform ready from the map to the camera for 1 second.
                #l.waitForTransform("map", camera_name, now, rospy.Duration(1.0))
                print("Waiting for transform")
                l.waitForTransform(camera_name, "marker_" + str(i), now, rospy.Duration(1))
                # extract the transform camera pose in the map coordinate.
                (trans, rot) = l.lookupTransform(camera_name, "marker_"+str(i) , now)
                # convert the rotate matrix to theta angle in 2d
                matrix = quaternion_matrix(rot)[:3,:3]
                # print("Rotation matrix cTa in control node: \n", matrix)
                # print("Translation cTa in control node: \n", trans)
                position = np.array([[trans.x], [trans.y], [trans.z]])
                cTa = np.append(np.append(matrix, position, axis=1), [[0, 0, 0, 1]], axis=0)
                print("Rotation matrix cTa in control node: \n", matrix)
                print("Translation cTa in control node: \n", position)
                rTa = np.matmul(rTc, cTa)
                aTr = np.linalg.inv(rTa)
                wTa = pose_ma[i]
                wTr = np.matmul(wTa, aTr)
                # angle = math.atan2(matrix[1][2], matrix[0][2])
                # this is not required, I just used this for debug in RVIZ
                # br.sendTransform((trans[0], trans[1], 0), tf.transformations.quaternion_from_euler(0, 0, angle),
                #                  rospy.Time.now(), "base_link", "map")
                print("Robot in world coordinates wTr in control node: \n", wTr)
                transwTr = wTr[:3, 3]
                rotwTr = wTr[:3, :3]
                eulerangles = rotationMatrixToEulerAngles(rot)
                yaw = eulerangles[2]
                result = np.array([trans[0], trans[1], yaw])
                foundSolution = True
                break
            except (
            tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException, tf2_ros.TransformException):
                print("meet error")
    listener.clear()
    return foundSolution, result


def genTwistMsg(desired_twist):
    """
    Convert the twist to twist msg.
    """
    twist_msg = Twist()
    twist_msg.linear.x = desired_twist[0]
    twist_msg.linear.y = desired_twist[1]
    twist_msg.linear.z = 0
    twist_msg.angular.x = 0
    twist_msg.angular.y = 0
    twist_msg.angular.z = desired_twist[2]
    return twist_msg


def coord(twist, current_state):
    """
    Convert the twist into the car coordinate
    """
    J = np.array([[np.cos(current_state[2]), np.sin(current_state[2]), 0.0],
                  [-np.sin(current_state[2]), np.cos(current_state[2]), 0.0],
                  [0.0, 0.0, 1.0]])
    return np.dot(J, twist)


if __name__ == "__main__":
    import time

    rospy.init_node("hw4")
    pub_twist = rospy.Publisher("/twist", Twist, queue_size=1)

    listener = tf.TransformListener()

    waypoint = np.array([[0.0, 0.0, 0.0],
                         [1.0, 0.0, 0.0],
                         [1.0, 2.0, np.pi],
                         [0.0, 0.0, 0.0]])

    # init pid controller
    pid = PIDcontroller(0.1, 0.005, 0.005)

    # init current state
    current_state = np.array([0.0, 0.0, 0.0])
    count = 0
    while count < 100:
        found_state, estimated_state = getCurrentPos(listener)
        print("Found state indicator: ", found_state)
        count += 1

    # in this loop we will go through each way point.
    # once error between the current state and the current way point is small enough,
    # the current way point will be updated with a new point.
    # for wp in waypoint:
    #     print("move to way point", wp)
    #     # set wp as the target point
    #     pid.setTarget(wp)
    #
    #     # calculate the current twist
    #     update_value = pid.update(current_state)
    #     # publish the twist
    #     pub_twist.publish(genTwistMsg(coord(update_value, current_state)))
    #     # print(coord(update_value, current_state))
    #     time.sleep(0.05)
    #     # update the current state
    #     current_state += update_value
    #     found_state, estimated_state = getCurrentPos(listener)
    #     if found_state:  # if the tag is detected, we can use it to update current state.
    #         current_state = estimated_state
    #     while (np.linalg.norm(
    #             pid.getError(current_state, wp)) > 0.05):  # check the error between current state and current way point
    #         # calculate the current twist
    #         update_value = pid.update(current_state)
    #         # publish the twist
    #         pub_twist.publish(genTwistMsg(coord(update_value, current_state)))
    #         # print(coord(update_value, current_state))
    #         time.sleep(0.05)
    #         # update the current state
    #         current_state += update_value
    #         found_state, estimated_state = getCurrentPos(listener)
    #         if found_state:
    #             current_state = estimated_state
    # # stop the car and exit
    # pub_twist.publish(genTwistMsg(np.array([0.0, 0.0, 0.0])))

