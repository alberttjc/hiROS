import argparse
import logging
import time

import cv2
import numpy as np

from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh

#from utils.lib_tracker import Tracker
from collections import OrderedDict, deque
import torch
from torchvision.transforms import Compose, CenterCrop, ToPILImage, ToTensor, Normalize

transform = Compose([
        ToTensor()
    ])

logger = logging.getLogger('TfPoseEstimator-WebCam')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

fps_time = 0

def humans_to_skels_list(humans, scale_h):
    ''' Get skeleton data of (x, y * scale_h) from humans.
    Arguments:
        humans {a class returned by self.detect}
        scale_h {float}: scale each skeleton's y coordinate (height) value.
            Default: (image_height / image_widht).
    Returns:
        skeletons {list of list}: a list of skeleton.
            Each skeleton is also a list with a length of 36 (18 joints * 2 coord values).
        scale_h {float}: The resultant height(y coordinate) range.
            The x coordinate is between [0, 1].
            The y coordinate is between [0, scale_h]
    '''
    skeletons = []
    NaN = 0
    for human in humans:
        skeleton = [NaN]*(18*2)
        for i, body_part in human.body_parts.items(): # iterate dict
            idx = body_part.part_idx
            skeleton[2*idx]=body_part.x
            skeleton[2*idx+1]=body_part.y * scale_h
        skeletons.append(skeleton)
    return skeletons, scale_h

def remove_skeletons_with_few_joints(skeletons):
    ''' Remove bad skeletons before sending to the tracker '''
    good_skeletons = []
    for skeleton in skeletons:
        px = skeleton[2:2+13*2:2]
        py = skeleton[3:2+13*2:2]
        num_valid_joints = len([x for x in px if x != 0])
        num_leg_joints = len([x for x in px[-6:] if x != 0])
        total_size = max(py) - min(py)
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # IF JOINTS ARE MISSING, TRY CHANGING THESE VALUES:
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        if num_valid_joints >= 5 and total_size >= 0.1 and num_leg_joints >= 0:
            # add this skeleton only when all requirements are satisfied
            good_skeletons.append(skeleton)
    return good_skeletons

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='tf-pose-estimation realtime webcam')

    parser.add_argument('--resize', type=str, default='0x0',
                        help='if provided, resize images before they are processed. default=0x0, Recommends : 432x368 or 656x368 or 1312x736 ')
    parser.add_argument('--resize-out-ratio', type=float, default=4.0,
                        help='if provided, resize heatmaps before they are post-processed. default=1.0')

    parser.add_argument('--model', type=str, default='mobilenet_thin', help='cmu / mobilenet_thin / mobilenet_v2_large / mobilenet_v2_small')
    parser.add_argument('--show-process', type=bool, default=False,
                        help='for debug purpose, if enabled, speed for inference is dropped.')

    parser.add_argument('--tensorrt', type=str, default="False",
                        help='for tensorrt process.')
    args = parser.parse_args()

    logger.debug('initialization %s : %s' % (args.model, get_graph_path(args.model)))
    w, h = model_wh(args.resize)

    if w > 0 and h > 0:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h), trt_bool=str2bool(args.tensorrt))
    else:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(432, 368), trt_bool=str2bool(args.tensorrt))

    # Camera initialization
    cam = cv2.VideoCapture(0)
    ret_val, image = cam.read()
    logger.info('cam image=%dx%d' % (image.shape[1], image.shape[0]))
    _scale_h = 1.0 * image.shape[0] / image.shape[1]

    # Function initialization
    #multiperson_tracker = Tracker()

    # Frame initialization
    #global Q
    Q = deque(maxlen=30)

    while True:
        ret_val, image = cam.read()

        logger.debug('image process+')

        # Detect skeletons
        humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=args.resize_out_ratio)
        skeletons, scale_h = humans_to_skels_list(humans,_scale_h)
        #skeletons = remove_skeletons_with_few_joints(skeletons)

        # -- Track people
        #dict_id2skeleton = multiperson_tracker.track(skeletons)

        """
        frames = []
        if skeletons:
            Q.append(skeletons)
            if len(Q) is 30:
                for frame in Q:
                    frames.append(frame)
                frames = torch.tensor(frames)
                #torch.unsqueeze(frames,0)
        #else:
        #    Q.append([[0]*36])

        print(frames)
        """
        Q.append(skeletons)
        frames  = []
        for frame_idx in Q:
            frame_idx = torch.tensor(frame_idx)
            frames.append(frame_idx)

        data = torch.cat(frames)
        #data = data.permute(1, 0, 2, 3)
        #data = data[None, :, :, :, :]

        print(data.size())

        """
        logger.debug('postprocess+')
        image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)

        logger.debug('show+')
        cv2.putText(image,
                    "FPS: %f" % (1.0 / (time.time() - fps_time)),
                    (10, 10),  cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 0), 2)
        cv2.imshow('tf-pose-estimation result', image)
        fps_time = time.time()
        """
        if cv2.waitKey(1) == 27:
            break

    cv2.destroyAllWindows()
logger.debug('finished+')


def process_frame(skeletons):
    return 0
