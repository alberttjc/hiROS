# hiROS 

## Introduction

hiROS (Human Interaction Robot Operating System) is a repo constructed using ROS Melodic (Robot Operating System) and [sense](https://github.com/TwentyBN/sense) as the main backbone of this repo.

The custom dataset used to train this repo uses communicative body and hand gestures, with the aim to improve human-robot interaction. Based on current literature, there is no standardised set of gestures used for human-robot interaction and this repo aims to address it.

## Dependencies
- ROS Melodic
- [sense](https://github.com/TwentyBN/sense)
- Python 3

## Environment
Before we are able to execute anything, ensure that this repo is in the right environment. Since ROS only uses Python2.7, you need to install Python3 in order to run this repo. Otherwise, you can use install a virtual environemtn of Python3

```
$ source py36env/bin/activate
```

If you are using ROS to run this repo, remember to build [opencv_vision](https://github.com/ros-perception/vision_opencv) in Python3 and import the **build** and **devel** files and merge it into your main workspace (i.e. catkin_ws).

## Tools

To run this gesture recognition inference, use the following command below. Run <kbd> rqt </kbd> to view the result

```
$ roslaunch hiROS inference.launch

or

# If you are using a kinect2_bridge 
$ roslaunch hiROS interface.launch 
```
