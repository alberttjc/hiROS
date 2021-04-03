#!/usr/bin/env python3

import os
import json
import rospy
import sys
import time
from cv_bridge import CvBridge, CvBridgeError

# ROS Messages
from sensor_msgs.msg import Image
from std_msgs.msg import String, Float64, Int16
from hiROS.msg import Gestures

# Dependncies for Sense
import torch
import sense.display
from sense.downstream_tasks.nn_utils import LogisticRegression
from sense.downstream_tasks.nn_utils import Pipe
from sense.downstream_tasks.postprocess import PostprocessClassificationOutput
from sense.loading import build_backbone_network
from sense.loading import load_backbone_model_from_config
from sense.loading import update_backbone_weights, load_backbone_weights
from sense.thresholds import GESTURE_THRESHOLDS

#Dependencies for HiROS
from collections import Callable
from typing import List
from typing import Optional
from typing import Union

from sense.display import DisplayResults
from sense.engine import InferenceEngine
from sense.downstream_tasks.nn_utils import RealtimeNeuralNet
from sense.downstream_tasks.postprocess import PostProcessor
from typing import Dict
import numpy as np
import cv2

# Dependncies to detect package path
from rospkg import RosPack
package = RosPack()
package_path = package.get_path('hiROS')

# To shutdown
import signal
def sigint_handler(sig, frame):
    main._stop_inference()
    exit(0)

class hiROS():
    def __init__(
            self,
            neural_network: RealtimeNeuralNet,
            post_processors: Union[PostProcessor, List[PostProcessor]],
            thresholds: Dict[str, float],
            callbacks: Optional[List[Callable]] = None,
            path_in: str = Optional[None],
            path_out: str = Optional[None],
            use_gpu: bool = True):
        
        self.inference_engine = InferenceEngine(neural_network, use_gpu=use_gpu)

        if isinstance(post_processors, list):
            self.postprocessors = post_processors
        else:
            self.postprocessors = [post_processors]
        
        self.callbacks = callbacks or []
        self.frame_index = None
        self.clip = None
        self.aspect_ratio = True

        # ROS Subscribers
        self.bridge = CvBridge()
        self.pub_image = rospy.Publisher("/hiROS/interface", Image, queue_size=10)
        self.pub_gestures = rospy.Publisher("/hiROS/gestures", Gestures, queue_size=1)
        self.sub_image = rospy.Subscriber(image_topic, Image, self.run_inference, queue_size=1, buff_size=2**24)
        self._start_inference()

        # Display Overlay
        self.thresholds = thresholds
        self._start_time = None
        self._class_prediction = None


    def run_inference(self, data):
        try:
            self.frame = self.bridge.imgmsg_to_cv2(data, "rgb8")
        except CvBridgeError as e:
            rospy.logerr('Converting Image Error. '+ str(e))

        _frame = self.process_image(self.frame)
        self.frame_index += 1
        self.clip = np.roll(self.clip, -1, 1)
        self.clip[:, -1, :, :, :] = _frame

        if self.frame_index == self.inference_engine.step_size:
            # A new clip is ready
            self.inference_engine.put_nowait(self.clip)
        
        self.frame_index = self.frame_index % self.inference_engine.step_size

        # Get predictions
        prediction = self.inference_engine.get_nowait()
        prediction_postprocessed = self.postprocess_prediction(prediction)
        self.display_prediction(self.frame, prediction_postprocessed)

        # Apply callbacks
        if not all(callback(prediction_postprocessed) for callback in self.callbacks):
            print('Error')  
    
    def process_image(self, img):
        #self.size = [img.shape[0], img.shape[1]]
        self.size = (self.inference_engine.expected_frame_size[0],
            self.inference_engine.expected_frame_size[1],)
        #print(self.size)
        if self.aspect_ratio:
            square_size = max(img.shape[0:2])
            pad_top = int((square_size - img.shape[0]) / 2)
            pad_bottom = square_size - img.shape[0] - pad_top
            pad_left = int((square_size - img.shape[1]) / 2)
            pad_right = square_size - img.shape[1] - pad_left
            pad_img = cv2.copyMakeBorder(img, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT)
        # size of the image is (256,256,3)
        return cv2.resize(pad_img, self.size) #if self.size else img

    def postprocess_prediction(self, prediction):
        post_processed_data = {}
        for post_processor in self.postprocessors:
            post_processed_data.update(post_processor(prediction))
        return {'prediction': prediction, **post_processed_data}

    def display_prediction(self, img: np.ndarray, prediction_postprocessed: dict):
        # Live display
        sorted_predictions = prediction_postprocessed['sorted_predictions']

        # Display Top 1 result from the inference layer
        for index in range(1):
            activity, proba = sorted_predictions[index]
            y_pos = 20* index + 40
        x_offset = int(self.frame.shape[1]/2 + y_pos)
        cv2.putText(img, 'Activity: {}'.format(activity[0:50]), (10,y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)
        cv2.putText(img, 'Proba: {}'.format("{:.2f}".format(proba)), (10 + x_offset,y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)

        # Display the Top 1 result after comparing the hard-coded thresholds
        now = self._get_current_time()

        if self._class_prediction and now - self._start_time < 2.0:
            textsize = cv2.getTextSize(self._class_prediction, cv2.FONT_HERSHEY_PLAIN, 2.5, 2)[0]
            w_middle = int((self.frame.shape[1]/2) - textsize[0])
            h_middle = int((self.frame.shape[0] + textsize[1]) / 2)
            cv2.putText(img, self._class_prediction, (w_middle,h_middle), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (255,255,255), 2, cv2.LINE_AA)
        else:
            self._class_prediction = None
            for class_name, proba in sorted_predictions:
                if class_name in self.thresholds and proba > self.thresholds[class_name]:
                    textsize = cv2.getTextSize(self._class_prediction, cv2.FONT_HERSHEY_PLAIN, 2.5, 2)[0]
                    w_middle = int((self.frame.shape[1]/2) - textsize[0])
                    h_middle = int((self.frame.shape[0] + textsize[1]) / 2)
                    cv2.putText(img, self._class_prediction, (w_middle,h_middle), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (255,255,255), 2, cv2.LINE_AA)
                    self._class_prediction = class_name
                    self._start_time = now

                    # Message 
                    gesture = Gestures()
                    gesture.action = class_name
                    gesture.action_index = class2int[class_name]
                    gesture.prob = proba
                    self.pub_gestures.publish(gesture)
                    break

        image_msg = self.bridge.cv2_to_imgmsg(img, "rgb8")
        self.pub_image.publish(image_msg)

    @staticmethod
    def _get_current_time() -> float:
        """
        Wrapper method to get the current time.
        Extracted for ease of testing.
        """
        return time.perf_counter()


    def _start_inference(self):
        rospy.loginfo("Starting inference")
        self.clip = np.random.randn(
            1,
            self.inference_engine.step_size,
            self.inference_engine.expected_frame_size[0],
            self.inference_engine.expected_frame_size[1],
            3
        )
        self.frame_index = 0
        self.inference_engine.start()

    def _stop_inference(self):
        rospy.loginfo("Stopping inference")
        self.inference_engine.stop()


if __name__ == '__main__':

    # Initialize sigint handler
    signal.signal(signal.SIGINT, sigint_handler)

    # Initialise node 
    rospy.loginfo('Initialise h.i.R.O.S node (human interaction Robot Operating System')
    rospy.init_node('hiROS', anonymous=True)
    rospy.loginfo(package_path)

    # Initialise parameters 
    image_topic     =   rospy.get_param('~image_topic',     '/kinect2/qhd/image_color_rect')
    ckpt_fn         =   rospy.get_param('~checkpoint_file', 'fyp1106.checkpoint')
    backbone_fn     =   rospy.get_param('~backbone_file',   'strided_inflated_efficientnet.ckpt')
    thresholds_fn   =   rospy.get_param('~thresholds',      'thresholds.py')
    use_gpu         =   rospy.get_param('~use_gpu',         'true')
    title           =   rospy.get_param('~title,',           None)

    if not image_topic:
        rospy.logerr('Parameter \'camera\' is not provided')
        sys.exit(-1)

    # Constants
    models_dir      =   os.path.join(package_path, 'models')
    ckpt_path       =   os.path.join(models_dir, ckpt_fn)
    backbone_path   =   os.path.join(models_dir, backbone_fn)

    # Raise error if it cannot find the model
    if not os.path.isfile(ckpt_path):
        raise IOError(('{:s} not found.').format(ckpt_path))

    # Load backbone network accoding to config file
    backbone_model_config = load_backbone_model_from_config(models_dir)
    backbone_weights = load_backbone_weights(backbone_path)
    # Load custom classifier
    checkpoint_classifier = torch.load(ckpt_path)
    # Update original weights in case some intermediate layers have been finetuned
    update_backbone_weights(backbone_weights,checkpoint_classifier)
    # Create backbone network
    backbone_network = build_backbone_network(backbone_model_config, backbone_weights)

    with open(os.path.join(models_dir, 'label2int.json')) as file:
        class2int = json.load(file)
    INT2LAB = {value: key for key, value in class2int.items()}

    gesture_classifier = LogisticRegression(num_in=backbone_network.feature_dim,
                                            num_out=len(INT2LAB))
    gesture_classifier.load_state_dict(checkpoint_classifier)
    gesture_classifier.eval()

    # Concatenate feature extractor and met converter
    net = Pipe(backbone_network, gesture_classifier)

    postprocessor = [
        PostprocessClassificationOutput(INT2LAB, smoothing=4)
    ]

    main = hiROS(
        neural_network=net,
        post_processors=postprocessor,
        thresholds=GESTURE_THRESHOLDS,
        callbacks=[],
        use_gpu=use_gpu
    )

    rospy.spin()










    



