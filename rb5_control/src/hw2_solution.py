#!/usr/bin/env python
import sys
import rospy
from geometry_msgs.msg import Twist
from april_detection.msg import Pose
import numpy as np
import math
from std_msgs.msg import Float64MultiArray

"""
The class of the pid controller.
"""

def isRotationMatrix(R) :
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype = R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6
 
# Calculates rotation matrix to euler angles
# The result is the same as MATLAB except the order
# of the euler angles ( x and z are swapped ).
def rotationMatrixToEulerAngles(R) :
 
    assert(isRotationMatrix(R))
 
    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
 
    singular = sy < 1e-6
 
    if not singular :
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else :
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0
    #print(np.array([x, y, z]))
    return np.array([x, y, z])

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
    #print("Twist message:",twist_msg)
    return twist_msg

def coord(twist, current_state):
    J = np.array([[np.cos(current_state[2]), np.sin(current_state[2]), 0.0],
                  [-np.sin(current_state[2]), np.cos(current_state[2]), 0.0],
                  [0.0,0.0,1.0]])
    return np.dot(J, twist)

class PIDcontroller:
    def __init__(self, Kp, Ki, Kd, waypoints):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.target = None
        self.I = np.array([0.0,0.0,0.0])
        self.lastError = np.array([0.0,0.0,0.0])
        self.timestep = 0.1
        self.maximumValue = 0.1
        self.waypoints = waypoints
        self.pub_twist = rospy.Publisher("/twist", Twist, queue_size=1)
        self.current_state =  np.array([0.0,0.0,0.0])
        self.current_state_before_update =  np.array([0.0,0.0,0.0])
        self.sub = rospy.Subscriber('/current_pose', Pose, self.pose_callback) 
        self.message_state = np.array([0.0,0.0,0.0])
        self.flag = False

    def setTarget(self, targetx, targety, targetw):
        """
        set the target pose.
        """
        self.I = np.array([0.0,0.0,0.0]) 
        self.lastError = np.array([0.0,0.0,0.0])
        self.target = np.array([targetx, targety, targetw])

    def setTarget(self, state):
        """
        set the target pose.
        """
        self.I = np.array([0.0,0.0,0.0]) 
        self.lastError = np.array([0.0,0.0,0.0])
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
        if(resultNorm > self.maximumValue):
            result = (result / resultNorm) * self.maximumValue
            self.I = 0.0

        return result

    def planner(self):
        print("Planner")
        total_dist = 0
        total_orientation = 0
        traj_points = []
        errors = []
        # init current state
        

        # in this loop we will go through each way point.
        # once error between the current state and the current way point is small enough, 
        # the current way point will be updated with a new point.
        for wp in self.waypoints:
            error_wp = []
            print("move to way point", wp)
            # set wp as the target point
            self.setTarget(wp)

            # calculate the current twist
            update_value = self.update(self.current_state)
            # publish the twist
            self.pub_twist.publish(genTwistMsg(coord(update_value, self.current_state)))
            #print(coord(update_value, current_state))
            time.sleep(0.05)
            # update the current state
            self.current_state_before_update = self.current_state
            self.current_state += update_value
            total_dist += np.linalg.norm(self.current_state[:2]-self.current_state_before_update[:2])
            traj_points.append([self.current_state[0], self.current_state[1]])
            
            while(np.linalg.norm(self.getError(self.current_state, wp)[:2]) > 0.1): # check the error between current state and current way point
                # calculate the current twist
                #print("Error:", np.linalg.norm(self.getError(self.current_state, wp)))
                error_wp.append(self.getError(self.current_state, wp)[:2])
                update_value = self.update(self.current_state)
                # publish the twist
                self.pub_twist.publish(genTwistMsg(coord(update_value, self.current_state)))
                #print(coord(update_value, current_state))
                time.sleep(0.05)
                self.current_state_before_update = self.current_state
                if self.flag == True:
                    #print("True")
                    self.current_state = self.message_state
                    #print("Current State in True:", self.current_state)
                    self.flag = False
                else:
                    #print("False")
                    # update the current state similar to open loop
                    self.current_state += update_value
                    #print("Current State in False:",self.current_state)
                    # update_value = pid.update(cur_pose)
                traj_points.append([self.current_state[0], self.current_state[1]])
                total_dist +=  np.linalg.norm(self.current_state[:2]-self.current_state_before_update[:2])
                total_orientation += abs(self.current_state[2]-self.current_state_before_update[2])
            print("Translation Error at waypoint:", np.linalg.norm(self.getError(self.current_state, wp)[:2]))
            print("Rotation Error at waypoint:", self.current_state[2]-wp[2]) 
            errors.append(error_wp)   
            time.sleep(0.05)
        #print("2D Trajectory points:", traj_points)
        with open(r'traj.txt', 'w') as fp:
            fp.write(','.join(str(v) for v in traj_points))
        with open(r'errors.txt', 'w') as fp:
            fp.write(','.join(str(v) for v in errors))
        # Total distance
        print(total_dist)
        # Total Orientation
        print(total_orientation)    
        # stop the car and exit
        self.pub_twist.publish(genTwistMsg(np.array([0.0,0.0,0.0])))


    def pose_callback(self, msg):
        #print("Pose Callback")
        if msg.pose:
            cur_pose_arr = np.asarray(msg.pose)
            #print("Current pose shape:",cur_pose_arr.shape)
            #print("Current Pose:",cur_pose_arr)
            cur_pose_matrix = cur_pose_arr.reshape(4,4)
            #print("Current Pose:", cur_pose_matrix)
            trans = cur_pose_matrix[:3, 3]
            #print("Translation:", trans)
            rot = cur_pose_matrix[:3, :3]
            #print("Rotation part of pose:", rot)
            eulerangles = rotationMatrixToEulerAngles(rot)
            yaw = eulerangles[2]
            #print("Angle with z-axis:", yaw)
            #print("Angle with y-axis:", eulerangles[1])
            #print("Angle with x-axis:", eulerangles[0])
            # update current state based on visual feedback
            self.message_state = np.asarray([trans[0], trans[1], yaw])
            #print("Message state:", self.message_state)
            self.flag = True
        else:
            #print("Empty")
            self.flag = False



if __name__ == "__main__":
    import time
    rospy.init_node("hw2")
    waypoints = [[1.0,0.0,0.0],[1.0,2.0,math.pi],[0,0,0]]
    # waypoints = [[1.0,0.0,0.0], [1.0,2.0,math.pi]]
    # pid = PIDcontroller(0.05,0.005,0.005, waypoints)
    # pid = PIDcontroller(0.05,0.0001,0.08, waypoints)
    #pid = PIDcontroller(0.0185,0.0015,0.01, waypoints) -- working approximately fine
    # pid = PIDcontroller(0.0185,0.0015,0.01, waypoints) -- works best for now
    # pid = PIDcontroller(0.0125,0.0015,0.1, waypoints) -- worked perfectly for 2 points
    # pid = PIDcontroller(0.0125,0.0015,0.05, waypoints) -- worked perfectly for 2 points
    
    pid = PIDcontroller(0.0185,0.0015,0.09, waypoints)
    time.sleep(1.0)
    pid.planner()
    rospy.spin()
    

    

