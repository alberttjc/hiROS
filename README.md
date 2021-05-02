# hiROS 

## Introduction

hiROS (Human Interaction Robot Operating System) is a repo constructed using ROS Melodic (Robot Operating System) and [sense](https://github.com/TwentyBN/sense) as the main backbone of this repo.

The custom dataset used to train this repo uses communicative body and hand gestures, with the aim to improve human-robot interaction. Based on current literature, there is no standardised set of gestures used for human-robot interaction and this repo aims to address it.

## Dependencies
- ROS Melodic
- [sense](https://github.com/TwentyBN/sense)
- Python 3


## Usage

To use this repository, you need to build [opencv_vision](https://github.com/ros-perception/vision_opencv) in Python3 and extend your main workspace (i.e. catkin_ws). More details can be found [here](#Setup)

First, you need to set up the environment.

```
# In this example, the package was built in ~/build_ws/
# In your main workspace (i.e. catkin_ws), you need to extend the environment
$ cd ~/catkin_ws/
$ catkin_make
$ source devel/setup.bash
$ source ~/build_ws/devel/setup.bash --extend
```

Then, you can run the inference layer. Run <kbd> rqt </kbd> to view the result.

```
# kinect = true (if you are using kinect)
# camera = 0 (change camera id if you have multiple cameras)

$ roslaunch hiROS inference.launch
$ roslaunch hiROS interface.launch kinect:=false camera:=0

# To just use the camera, you can run this two launch files
$ roslaunch video_stream_opencv webcam.launch
$ roslaunch video_stream_opencv camera.launch
```

To work on the Fetch robot, you need to enable SSH to communicate with it. In this case, the master should be the Fetch and the Follower should be your device.

```
# On personal device, run
$ export ROS_MASTER_URI=http://[Fetch robot IP]:11311/
$ export ROS_IP=[personal device IP]
```

For my project, I used these commands below:
```
# On personal device
$ export ROS_MASTER_URI=http://160.69.69.80:11311/
$ export ROS_IP=160.69.69.129
$ roslaunch hiROS interface.launch kinect:=false camera:=0

# On Fetch
$ ssh hrigroup@160.69.69.80
$ roslaunch hiros_smach robot.launch
$ rosrun hiros_smach main.py
```

## Setup


Once you have installed all the right dependancies for [opencv_vision](https://github.com/ros-perception/vision_opencv), you have to build your workspace. More details can be found from this [site](https://medium.com/@beta_b0t/how-to-setup-ros-with-python-3-44a69ca36674) for reference.

```
$ mkdir -p ~/build_ws/src
$ cd ~/build_ws/src
$ git clone -b melodic https://github.com/ros-perception/vision_opencv.git
$ cd ~/build_ws
$ catkin build cv_bridge

# Once you build it, you will need to extend it into your main working environment 
$ source install/setup.bash --extend
```