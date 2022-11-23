#!/usr/bin/env python

import rospy
import cv2
import apriltag
from april_detection.msg import AprilTagDetectionArray
from april_detection.msg import Pose
# from std_msgs.msg import Float64MultiArray
import numpy as np
import time
import tf

# pose_pub = rospy.Publisher('/current_pose', Float64MultiArray, queue_size=1)
pose_pub = rospy.Publisher('/current_pose', Pose, queue_size=1)

# Location of the marker AprilTag
#pose_ma = {8: np.asarray([[0, -1, 0, 2.05],[0, 0, -1, 0.015], [1, 0, 0, 0.15], [0,0,0,1]])}
# pose_ma = {8: np.asarray([[0, 0, 1, 2.05],[-1, 0, 0, 0.015], [0, -1, 0, 0.15], [0,0,0,1]]),
# 5: np.asarray([[0, 0, -1, -0.2],[-1, 0, 0, 2.3], [0, -1, 0, 0.15], [0,0,0,1]])}
# rTc = np.asarray([[0, 0, 1, 0.05], [-1, 0, 0, 0.015], [0,-1,0, 0.15], [0,0,0,1]])
# pose_ma = {3: np.asarray([[-1, 0, 0, 1],[0, 0, -1, 0], [0, -1,  0, 0.15], [0,0,0,1]]),
# 2: np.asarray([[-1, 0, 0, 2],[0, 0, -1, 0], [0, -1, 0, 0.15], [0,0,0,1]]),
# 7: np.asarray([[0, 0, 1, 3.0],[-1, 0, 0, 1.0], [0, -1, 0, 0.15], [0,0,0,1]]),
# 8: np.asarray([[0, 0, 1, 3.0],[-1, 0, 0, 2.0], [0, -1, 0, 0.15], [0,0,0,1]]),
# 4: np.asarray([[1, 0, 0, 2],[0, 0, 1, 3], [0, -1, 0, 0.15], [0,0,0,1]]),
# 1: np.asarray([[1, 0, 0, 1], [0, 0, 1, 3], [0, -1, 0, 0.15], [0, 0, 0, 1]]),
# 5: np.asarray([[1, 0, 0, 1.5], [0, 0, 1, 1.3], [0, -1, 0, 0.15], [0, 0, 0, 1]]),
# 0: np.asarray([[0, 0, 1, 1.3],[-1, 0, 0, 1.43], [0, -1, 0, 0.15], [0,0,0,1]])}


# For Voronoi
# rTc = np.asarray([[0, 0, 1, 0.05], [-1, 0, 0, 0.015], [0,-1,0, 0.15], [0,0,0,1]])
# pose_ma = {
#     # 3: np.asarray([[-1, 0, 0, 0.5],[0, 0, -1, 0.0], [0, -1,  0, 0.15], [0,0,0,1]]),
# 2: np.asarray([[-1, 0, 0, 1.5],[0, 0, -1, 0.0], [0, -1, 0, 0.15], [0,0,0,1]]),
# 1: np.asarray([[0, 0, 1, 2.0],[-1, 0, 0, 0.5], [0, -1, 0, 0.15], [0,0,0,1]]),
# 8: np.asarray([[0, 0, 1, 2.0],[-1, 0, 0, 1.5], [0, -1, 0, 0.15], [0,0,0,1]]),
# # 4: np.asarray([[1, 0, 0, 1.5],[0, 0, 1, 2.0], [0, -1, 0, 0.15], [0,0,0,1]]),
# # 1: np.asarray([[1, 0, 0, 0.5], [0, 0, 1, 2.0], [0, -1, 0, 0.15], [0, 0, 0, 1]]),
# 5: np.asarray([[1, 0, 0, 1], [0, 0, 1, 0.87], [0, -1, 0, 0.15], [0, 0, 0, 1]]),
# # 0: np.asarray([[0, 0, 1, 1.3],[-1, 0, 0, 1.43], [0, -1, 0, 0.15], [0,0,0,1]])
# }

# For A*
pose_ma = {
    # 3: np.asarray([[-1, 0, 0, 0.5],[0, 0, -1, 0.0], [0, -1,  0, 0.15], [0,0,0,1]]),
2: np.asarray([[-1, 0, 0, 1.5],[0, 0, -1, 0.0], [0, -1, 0, 0.15], [0,0,0,1]]),
1: np.asarray([[0, 0, 1, 2.0],[-1, 0, 0, 0.5], [0, -1, 0, 0.15], [0,0,0,1]]),
8: np.asarray([[0, 0, 1, 0.76],[-1, 0, 0, 0.68], [0, -1, 0, 0], [0,0,0,1]]), 
# np.asarray([[1, 0, 0, 1], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]])]
# 8: np.asarray([[0, 0, 1, 2.0],[-1, 0, 0, 1.5], [0, -1, 0, 0.15], [0,0,0,1]]),
# 4: np.asarray([[1, 0, 0, 1.5],[0, 0, 1, 2.0], [0, -1, 0, 0.15], [0,0,0,1]]),
# 1: np.asarray([[1, 0, 0, 0.5], [0, 0, 1, 2.0], [0, -1, 0, 0.15], [0, 0, 0, 1]]),
# 0: np.asarray([[1, 0, 0, 1], [0, 0, 1, 0.83], [0, -1, 0, 0.15], [0, 0, 0, 1]]),
# 0: np.asarray([[0, 0, 1, 1.3],[-1, 0, 0, 1.43], [0, -1, 0, 0.15], [0,0,0,1]])
}

# Camera in robot frame
#rTc = np.asarray([[0, -1, 0, 0.05], [0, 0, -1, 0.015], [1,0,0, 0.15], [0,0,0,1]])
rTc = np.asarray([[0, 0, 1, 0.05], [-1, 0, 0, 0.015], [0,-1,0, 0.15], [0,0,0,1]])

def tag_callback(msg):
    pose_msg = Pose()
    pose_msg.header.stamp = msg.header.stamp
    for detection in msg.detections:

        #print("Type:",type(detection))
        apriltag_id = detection.id
        position = detection.pose.position
        position = np.array([[position.x], [position.y], [position.z]])
        #print("ID:", apriltag_id)
        #print("Position:", position)
        tag_orientation = detection.pose.orientation 
        #print("Tag oreintation quaternion:", tag_orientation)
        
        # r = np.array(Rotation.from_quat(tag_orientation))
        # print("Rotation matrix using scipy:", r)
        r = tf.transformations.quaternion_matrix([tag_orientation.x, tag_orientation.y, 
        tag_orientation.z, tag_orientation.w])[:3,:3]
        #print("Rotation Matrix using tf: \n", r)
        #print("Size of rotation matrix:", r.shape)
        #print("Size of position matrix:",position.shape)
        cTa = np.append(np.append(r, position,axis=1), [[0,0,0,1]], axis=0)
        print("Rotation matrix cTa in localization node: \n",r)
        print("Translation cTa in localization node: \n", position)
        rTa = np.matmul(rTc, cTa)
        #print("AprilTag in robot coordinates rTa: \n",rTa)
        aTr = np.linalg.inv(rTa)
        wTa = pose_ma[apriltag_id]
        wTr = np.matmul(wTa,aTr)
        # aTc = np.linalg.inv(cTa)
        # cTr = np.linalg.inv(rTc)
        # rTa = np.matmul(cTr, cTa)
        # print("AprilTag in robot coordinates rTa: \n",rTa)
        # AprilTag in robot coordinates
        # rTa = np.matmul(aTc, rTc) -- v
        # rTa = np.matmul(rTc, cTa) -- first attempt
        # aTr = np.matmul(aTc, rTc) -- second attempt
        
        # aTr = np.matmul(cTa, cTr)
        # print("AprilTag in robot coordinates rTa: \n",rTa) --v
        # print("AprilTag in robot coordinates rTa: \n",aTr)
        # Robot in world coordinates
        # rTa = np.linalg.inv(aTr)
        # wTa = pose_ma[apriltag_id]
        # aTw = np.linalg.inv(wTa)
        # aTr = np.linalg.inv(rTa)
        # wTr = np.matmul(pose_ma[apriltag_id], aTr)
        print("Robot in world coordinates wTr: \n",wTr)
        # new[apriltag_id] = wTr
        pose_msg.pose = list(wTr.flatten())
        pose_pub.publish(pose_msg)
        # pose_pub.publish(wTr)

if __name__ == "__main__":
    rospy.init_node('localization_node')
    rospy.Subscriber("/apriltag_detection_array", AprilTagDetectionArray, tag_callback)
    rospy.spin()