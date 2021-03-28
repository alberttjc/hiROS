#!/usr/bin/env python

''' Imports '''
import os
import sys
import time
import rospy
import signal
import cv2 as cv
import numpy as np
import json
from threading import Lock

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable as V
from torchvision.transforms import Compose, CenterCrop, ToPILImage, ToTensor, Normalize
from collections import OrderedDict, deque
from cv_bridge import CvBridge, CvBridgeError
from imutils.video import VideoStream, FileVideoStream, WebcamVideoStream, FPS

import imutils
from sensor_msgs.msg import Image
from Hand_Recognition_ROS.msg import Persons, Person, BodyPartElm
from Hand_Recognition_ROS.msg import OpenPoseHumanList, OpenPoseHuman, BoundingBox, PointWithProb
from model import ConvColumn
import argparse
import pyautogui
import configparser

# from train_data.classes_dict in train.py
gesture_dict = {
    'Doing other things': 0, 0: 'Doing other things',
    'No gesture': 1, 1: 'No gesture',
    'Stop Sign': 2, 2: 'Stop Sign',
    'Swiping Down': 3, 3: 'Swiping Down',
    'Swiping Left': 4, 4: 'Swiping Left',
    'Swiping Right': 5, 5: 'Swiping Right',
    'Swiping Up': 6, 6: 'Swiping Up',
    'Turning Hand Clockwise': 7, 7: 'Turning Hand Clockwise',
    'Turning Hand Counterclockwise': 8, 8: 'Turning Hand Counterclockwise'
}

# initialise some variables
qsize = 20  # size of queue to retain for 3D conv input
sqsize = 10 # size of queue for prediction stabilisation
num_classes = 9
threshold = 0.75

verbose = 2

transform = Compose([
        ToPILImage(),
        CenterCrop(84),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
    ])


def callback_image(data):
    global Q
    global SQ
    global act
    global image_w, image_h
    global current_center
    global boundary_center

    try:
        frame = cv_bridge.imgmsg_to_cv2(data, 'bgr8')
    except CvBridgeError as e:
        rospy.logerr('Converting Image Error. ' + str(e))
        return

    image_w = frame.shape[0]
    image_h = frame.shape[1]
    #print(image_w,image_h)

    #frame = imutils.resize(frame, height=100)

    #cv.imshow("Frame",frame)
    y1 = 201
    y2 = 440
    x1 = 100
    x2 = 440
    #frame = frame[current_center[0]:current_center[1],current_center[2]:current_center[3]]
    #frame = frame[current_center[2]:current_center[3],current_center[0]:current_center[1]]
    #frame = frame[x1:x2,y1:y2]

    #frame = frame[boundary_center[2]:boundary_center[3],boundary_center[0]:boundary_center[1]]
    #frame = frame[boundary_center[0]:boundary_center[1],boundary_center[2]:boundary_center[3]]
    frame = imutils.resize(frame, height=100)
    #pub_image.publish(cv_bridge.cv2_to_imgmsg(frame,'bgr8'))

    Q.append(frame)

    imgs = []
    for img in Q:
        img = transform(img)
        imgs.append(torch.unsqueeze(img,0))

    data = torch.cat(imgs)
    data = data.permute(1, 0, 2, 3)
    data = data[None, :, :, :, :]
    target = [2]
    target = torch.tensor(target)
    data = data.to(device)

    if (data.shape[2] == qsize):
        output = model(data)
        output = torch.nn.functional.softmax(output, dim=1)

        k = 5
        ts, pred = output.detach().cpu().topk(k, 1, True, True)
        top5 = [gesture_dict[pred[0][i].item()] for i in range(k)]

        pi = [pred[0][i].item() for i in range(k)]
        ps = [ts[0][i].item() for i in range(k)]
        top1 = top5[0] if ps[0] > threshold else gesture_dict[0]

        hist = {}

        for i in range(num_classes):
            hist[i] = 0

        for i in range(len(pi)):
            hist[pi[i]] = ps[i]

        SQ.append(list(hist.values()))
        ave_pred = np.array(SQ).mean(axis=0)
        top1 = gesture_dict[np.argmax(ave_pred)] if max(ave_pred) > threshold else gesture_dict[0]

        top1 = top1.lower()
        act.append(top1)

        print(top1)


def cb_pose(data):
    global current_center
    global boundary_center
    global image_w, image_h
    # get image with pose time
    #t = data.header.stamp
    # 3,4,6,7,8,9,19,11,12,13,

    current_center = [500,0,500,0]
    #current_center = [0,0]
    #boundary_center = [0,0,0,0]

    body_part_number = [0,1,14,15,16,17]
    threshold = 100
    boundary = 50

    #for human in range(data.num_humans):
        #print(data.human_list)

    for BoundingBox in (data.human_list):
        #print(BoundingBox.face_bounding_box.width)
        center_width = int(BoundingBox.face_bounding_box.width)
        center_height = int(BoundingBox.face_bounding_box.height)
        center_x = BoundingBox.face_bounding_box.x
        center_y = BoundingBox.face_bounding_box.y

        center = (int(center_x * image_h + 0.5), int(center_y *image_w + 0.5))

        #print(center_x)
        #print(center_y)

        current_center[0] = int(center_x - 50) if abs(current_center[0] - center_x) > threshold  else current_center[0]
        current_center[1] = int(center_x + center_width + 50) if abs(current_center[1] - center_x) > threshold else current_center[1]
        current_center[2] = int(center_y - 50) if abs(current_center[2] - center_y) > threshold  else current_center[2]
        current_center[3] = int(center_y + center_height + 50) if abs(current_center[3] - center_y) > threshold else current_center[3]

        #current_center[0] = int(center_width) if abs(current_center[0] - center_width) > threshold  else current_center[0]
        #current_center[1] = int(center_width) if abs(current_center[1] - center_width) > threshold else current_center[1]
        #print(current_center)


    """

    for p_idx, person in enumerate(data.persons):
        for body_part in person.body_part:
            if body_part.part_id in body_part_number:
                print(body_part.part_id)
                center = (int(body_part.x * image_h + 0.5), int(body_part.y *image_w + 0.5))


            current_center[0] = center[0] if (current_center[0] > center[0]) else current_center[0]
            current_center[1] = center[0] if (current_center[1] < center[0]) else current_center[1]
            current_center[2] = center[1] if (current_center[2] > center[1]) else current_center[2]
            current_center[3] = center[1] if (current_center[3] < center[1]) else current_center[3]

            #boundary_center[0] = current_center[0] if boundary_center[0] > current_center[0] else boundary_center[0]
            #boundary_center[1] = current_center[1] if boundary_center[1] < current_center[1] else boundary_center[1]
            #boundary_center[2] = current_center[2] if boundary_center[2] > current_center[2] else boundary_center[2]
            #boundary_center[3] = current_center[3] if boundary_center[3] < current_center[3] else boundary_center[3]

            for i in range(len(boundary_center)):
                if boundary_center[i] is 0 and i in [0,2]:
                    boundary_center[i] =  current_center[i] - boundary
                elif boundary_center[i] is 0 and i in [1,3]:
                    boundary_center[i] =  current_center[i] + boundary

                #print(boundary_center)


                #print(abs(boundary_center[i]-current_center[i]))
                if threshold <= abs(boundary_center[i]-current_center[i]) <= (2*threshold):
                    if i in [0,2]:
                        boundary_center[i]=current_center[i]-boundary
                    elif i in [1,3]:
                        boundary_center[i]=current_center[i]+boundary

                if abs(boundary_center[i]-old_boundary_center[i]) > threshold:
                    boundary_center[i]=old_boundary_center[i]

            old_boundary_center = boundary_center




            #boundary_center[0] = current_center[0] - boundary  if (boundary_center[0] - threshold) > current_center[0] else boundary_center[0]
            #boundary_center[1] = current_center[1] + boundary  if (boundary_center[1] + threshold) < current_center[1] else boundary_center[1]
            #boundary_center[2] = current_center[2] - boundary  if (boundary_center[2] - threshold) > current_center[2] else boundary_center[2]
            #boundary_center[3] = current_center[3] + boundary  if (boundary_center[3] + threshold) < current_center[3] else boundary_center[3]



            #print(body_part.part_id)
            #print(boundary_center)

        """





def sigint_handler(sig,iteration):
    sys.exit(0)

if __name__ == '__main__':

    global Q
    global SQ
    global act

    signal.signal(signal.SIGINT, sigint_handler)

    ''' Initialize node '''
    rospy.loginfo('Initialization+')
    rospy.init_node("Gesture_inference", anonymous=True)

    ''' Initialize parameters '''
    debug           =   rospy.get_param('~debug', 'True')
    map_fn          =   rospy.get_param('~map', "/home/caris/catkin_ws/src/Hand_Recognition_ROS/scripts/mapping.ini")
    ckpt_fn         =   rospy.get_param('~ckpt', "/home/caris/catkin_ws/src/Hand_Recognition_ROS/model/model_best.pth.tar")
    image_topic     =   rospy.get_param('~camera', "/kinect2/qhd/image_color_rect")

    if not image_topic:
        rospy.logerr('Parameter \'camera\' is not provided')
        sys.exit(-1)

    ''' Initialize Constant '''
    #time.sleep(2.0)
    Q   = deque(maxlen=qsize)
    SQ  = deque(maxlen=sqsize)
    act = deque(['No gesture','No gesture'], maxlen=3)

    cv_bridge   =   CvBridge()
    #rospy.wait_for_message(image_topic, Image, timeout=30)
    sub_image   =   rospy.Subscriber(image_topic, Image, callback_image, queue_size=1, buff_size=2**24)
    sub_pose    =   rospy.Subscriber("/openpose_ros/human_list", OpenPoseHumanList, cb_pose, queue_size=1)
    pub_image   =   rospy.Publisher("frame", Image, queue_size=1)

    ''' Initialize inference layer '''
    use_cuda    =   torch.cuda.is_available()
    device      =   torch.device('cuda')
    model       =   ConvColumn(num_classes)
    model       =   model.cuda()

    mapping     =   configparser.ConfigParser()
    action      =   {}

    mapping.read(map_fn)

    for m in mapping['MAPPING']:
        val         =   mapping['MAPPING'][m].split(',')
        action[m]   =   {'fn': val[0], 'keys': val[1:]} # fn: hotkey, press, typewrite

    ''' Loading checkpoint file '''
    checkpoint      =   torch.load(ckpt_fn)
    new_state_dict  =   OrderedDict()

    for k, v in checkpoint.items():
        if (k== 'state_dict'):
            del checkpoint['state_dict']
            for j, val in v.items():
                name = j[7:] # remove module
                new_state_dict[name] = val
            checkpoint['state_dict'] = new_state_dict
            break
    model.load_state_dict(checkpoint['state_dict'])

    if (verbose>0): print("=> loaded checkpoint '{}' (epoch {})"
                .format(ckpt_fn, checkpoint['epoch']))

    model.eval()

    if (verbose>0): print("[INFO] Attempting to start video stream ...")

    rospy.loginfo('start+')
    rospy.spin()
    rospy.loginfo('finished')
