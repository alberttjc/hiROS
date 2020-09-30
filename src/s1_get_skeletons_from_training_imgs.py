#!/usr/bin/env python
# coding: utf-8

import cv2
import numpy as np
import yaml
from tf_pose import common

if True:  # Include project path
    import sys
    import os
    ROOT = os.path.dirname(os.path.abspath(__file__))+"/../"
    CURR_PATH = os.path.dirname(os.path.abspath(__file__))+"/"
    sys.path.append(ROOT)

    from utils.lib_openpose import SkeletonDetector
    from utils.lib_tracker import Tracker
    from utils.lib_skeletons_io import ReadValidImagesAndActionTypesByTxt
    import utils.lib_commons as lib_commons


def par(path):  # Pre-Append ROOT to the path if it's not absolute
    return ROOT + path if (path and path[0] != "/") else path

# -- Settings


cfg_all = lib_commons.read_yaml(ROOT + "config/config.yaml")
cfg = cfg_all["s1_get_skeletons_from_training_imgs.py"]
CLASSES = np.array(cfg_all["classes"])

IMG_FILENAME_FORMAT = cfg_all["image_filename_format"]
SKELETON_FILENAME_FORMAT = cfg_all["skeleton_filename_format"]


# Input
if True:
    #SRC_IMAGES_DESCRIPTION_TXT = par(cfg["input"]["images_description_txt"])
    SRC_IMAGES_FOLDER = par(cfg["input"]["images_folder"])

# Output
if True:
    # This txt will store image info, such as index, action label, filename, etc.
    # This file is saved but not used.
    #DST_IMAGES_INFO_TXT = par(cfg["output"]["images_info_txt"])

    # Each txt will store the skeleton of each image
    DST_DETECTED_SKELETONS_FOLDER = par(
        cfg["output"]["detected_skeletons_folder"])

    # Each image is drawn with the detected skeleton
    DST_VIZ_IMGS_FOLDER = par(cfg["output"]["viz_imgs_folders"])

# Openpose
if True:
    OPENPOSE_MODEL = cfg["openpose"]["model"]
    OPENPOSE_IMG_SIZE = cfg["openpose"]["img_size"]

# -- Functions

class ImageDisplayer(object):
    ''' A simple wrapper of using cv2.imshow to display image '''

    def __init__(self):
        self._window_name = "cv2_display_window"
        cv2.namedWindow(self._window_name)

    def display(self, image, wait_key_ms=1):
        cv2.imshow(self._window_name, image)
        cv2.waitKey(wait_key_ms)

    def __del__(self):
        cv2.destroyWindow(self._window_name)

# -- Directory Count

files = folders = 0
for _, dirnames, filenames in os.walk(SRC_IMAGES_FOLDER):
    files += len(filenames)
    folders += len(dirnames)
#print "{:,} files, {:,} folders".format(files, folders)


def main():
    ith_img = 7831#0
    action_idx = None
    for folder_idx in range(folders):
        folder_idx += 1
        last_file_idx = 1

        #test
        folder_test = folder_idx + 160
        INPUT_DIR = SRC_IMAGES_FOLDER + ("%s" % (folder_test)) + "/"
        _, _, files = next(os.walk(INPUT_DIR))

        """

        INPUT_DIR = SRC_IMAGES_FOLDER + ("%s" % folder_idx) + "/"
        #print("Processing current folder: %s", INPUT_DIR)

        _, _, files = next(os.walk(INPUT_DIR))
        #print(len(files))

        if folder_idx < (1*32+1):   action_idx = 0
        elif folder_idx < (2*32+1):   action_idx = 1
        elif folder_idx < (3*32+1):   action_idx = 2
        elif folder_idx < (4*32+1):   action_idx = 3
        """

        if folder_test < (1*32+1):   action_idx = 0
        elif folder_test < (2*32+1):   action_idx = 1
        elif folder_test < (3*32+1):   action_idx = 2
        elif folder_test < (4*32+1):   action_idx = 3
        elif folder_test < (5*32+1):   action_idx = 4
        elif folder_test < (6*32+1):   action_idx = 5
        elif folder_test < (7*32+1):   action_idx = 6


        for frame_idx in range(len(files)):

            name = 'frame' + str(last_file_idx) + '.jpg'
            current_frame = INPUT_DIR + name
            print(current_frame)

            img = common.read_imgfile(current_frame, None, None)

            if img is None:
                logger.error('Image cannot be read')
                sys.exit(-1)

            # -- Detect
            humans = skeleton_detector.detect(img)

            # -- Draw
            #img_disp = img.copy()
            #skeleton_detector.draw(img_disp, humans)
            #img_displayer.display(img_disp, wait_key_ms=1)

            # -- Get skeleton data and save to file
            skeletons, scale_h = skeleton_detector.humans_to_skels_list(humans)
            dict_id2skeleton = multiperson_tracker.track(
                skeletons)  # dict: (int human id) -> (np.array() skeleton)
            skels_to_save = [[action_idx+1]+[folder_test]+[ith_img+1]+[CLASSES[action_idx]]+skeleton.tolist()
                         for skeleton in dict_id2skeleton.values()]

            # -- Save result

            # Save the visualized image for debug
            #filename = IMG_FILENAME_FORMAT.format(ith_img)
            #cv2.imwrite(
            #    DST_VIZ_IMGS_FOLDER + filename,
            #    img_disp)

            # Save skeleton data for training
            filename = SKELETON_FILENAME_FORMAT.format(ith_img)
            lib_commons.save_listlist(
                DST_DETECTED_SKELETONS_FOLDER + filename,
                skels_to_save)

            ith_img += 1
            last_file_idx += 1

# -- Main
if __name__ == "__main__":

    # -- Detector
    skeleton_detector = SkeletonDetector(OPENPOSE_MODEL, OPENPOSE_IMG_SIZE)
    multiperson_tracker = Tracker()

    #img_displayer = ImageDisplayer()

    # -- Init output path

    #if not os.path.exists(os.path.dirname(DST_IMAGES_INFO_TXT)):
    #    os.makedirs(os.path.dirname(DST_IMAGES_INFO_TXT))
    if not os.path.exists(DST_DETECTED_SKELETONS_FOLDER):
        os.makedirs(DST_DETECTED_SKELETONS_FOLDER)
    if not os.path.exists(DST_VIZ_IMGS_FOLDER):
        os.makedirs(DST_VIZ_IMGS_FOLDER)

    main()
    print("Program ends")
