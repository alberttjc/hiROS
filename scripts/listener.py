#!/usr/bin/env python

import rospy
from HiROS.msg import Gestures

def callback(data):
    rospy.loginfo('[Action %s] %s ', data.action_index, data.action)
    rospy.loginfo('[P-value] %s', data.prob)

def listener():
    rospy.init_node('listener', anonymous=True)
    rospy.Subscriber('/HiROS/gestures', Gestures, callback)
    rospy.spin()

if __name__ == '__main__':
    listener()
