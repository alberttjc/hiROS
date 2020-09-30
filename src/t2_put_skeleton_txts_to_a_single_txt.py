#!/usr/bin/env python
# coding: utf-8

'''
Read multiple skeletons txts and saved them into a single txt.
If an image doesn't have skeleton, discard it.
If an image label is not `CLASSES`, discard it.

Input:
    `skeletons/00001.txt` ~ `skeletons/xxxxx.txt` from `SRC_DETECTED_SKELETONS_FOLDER`.
Output:
    `skeletons_info.txt`. The filepath is `DST_ALL_SKELETONS_TXT`.
'''

import numpy as np
import simplejson
import collections

if True:  # Include project path
    import sys
    import os
    ROOT = os.path.dirname(os.path.abspath(__file__))+"/../"
    CURR_PATH = os.path.dirname(os.path.abspath(__file__))+"/"
    sys.path.append(ROOT)

    # import utils.lib_feature_proc # This is no needed,
    #   because this script only transfer (part of) the data from many txts to a single txt,
    #   without doing any data analsysis.

    import utils.lib_commons as lib_commons


def par(path):  # Pre-Append ROOT to the path if it's not absolute
    return ROOT + path if (path and path[0] != "/") else path

# -- Settings

cfg_all = lib_commons.read_yaml(ROOT + "config/config.yaml")
cfg = cfg_all["s2_put_skeleton_txts_to_a_single_txt.py"]

CLASSES = np.array(cfg_all["classes"])

SKELETON_FILENAME_FORMAT = cfg_all["skeleton_filename_format"]

SRC_DETECTED_SKELETONS_FOLDER = par(cfg["input"]["detected_skeletons_folder"])

DST_ALL_SKELETONS_TXT = "/home/albie/anxious_popcorn/data_proc/UTD-MHAD/test_data.txt"
SRC_IMAGES_FOLDER = "/home/albie/anxious_popcorn/data/UTD-MHAD/"

IDX_PERSON = 0  # Only use the skeleton of the 0th person in each image
IDX_ACTION_LABEL = 3  # [1, 7, 54, "jump", "jump_03-02-12-34-01-795/00240.jpg"]

# -- Helper function


def read_skeletons_from_ith_txt(i):
    '''
    Arguments:
        i {int}: the ith skeleton txt. Zero-based index.
            If there are mutliple people, then there are multiple skeletons' data in this txt.
    Return:
        skeletons_in_ith_txt {list of list}:
            Length of each skeleton data is supposed to be 41 = 5 image info + 36 xy positions.
    '''
    filename = SRC_DETECTED_SKELETONS_FOLDER + \
        SKELETON_FILENAME_FORMAT.format(i)
    skeletons_in_ith_txt = lib_commons.read_listlist(filename)
    return skeletons_in_ith_txt


def get_length_of_one_skeleton_data(filepaths):
    ''' Find a non-empty txt file, and then get the length of one skeleton data.
    The data length should be 41, where:
    41 = 5 + 36.
        5: [cnt_action, cnt_clip, cnt_image, action_label, filepath]
            See utils.lib_io.get_training_imgs_info for more details
        36: 18 joints * 2 xy positions
    '''
    for i in range(len(filepaths)):
        skeletons = read_skeletons_from_ith_txt(i)
        if len(skeletons):
            skeleton = skeletons[IDX_PERSON]
            data_size = len(skeleton)
            assert(data_size == 40)
            return data_size
    raise RuntimeError("No valid txt under: {}".format(SRC_DETECTED_SKELETONS_FOLDER))

files = folders = 0

for _, dirnames, filenames in os.walk(SRC_IMAGES_FOLDER):
    files += len(filenames)
    folders += len(dirnames)
print "{:,} files, {:,} folders".format(files, folders)


if __name__ == "__main__":

    ''' Extract the middle 30 of the dataset'''

    ith_img = 1
    arrays = []
    for folder_idx in range(folders):#[:128]:
        folder_idx += 1
        array = []
        #ith_img = 1
        array_size = 30
        toggle = False

        INPUT_DIR = SRC_IMAGES_FOLDER + ("%s" % (folder_idx)) + "/"
        _, _, files = next(os.walk(INPUT_DIR))

        for frame_idx in range(len(files)):
            array.append(ith_img)
            ith_img += 1
        while len(array) > array_size:
            if toggle is True:      array = array[1:]
            else:                   array = array[:-1]
            toggle = not toggle
        arrays += array
        print(folder_idx)
    #print(arrays)

    ''' Read multiple skeletons txts and saved them into a single txt. '''

    # -- Get skeleton filenames
    filepaths = lib_commons.get_filenames(SRC_DETECTED_SKELETONS_FOLDER,
                                          use_sort=True, with_folder_path=True)
    num_skeletons = len(filepaths)

    # -- Check data length of one skeleton
    data_length = get_length_of_one_skeleton_data(filepaths)
    print("Data length of one skeleton is {}".format(data_length))

    # -- Read in skeletons and push to all_skeletons
    all_skeletons = []
    labels_cnt = collections.defaultdict(int)
    for i in range(num_skeletons):#[:6050]:

        # Read skeletons from a txt
        if i in arrays:
            skeletons = read_skeletons_from_ith_txt(i)
        else:
            continue

        if not skeletons:  # If empty, discard this image.
            continue
        skeleton = skeletons[IDX_PERSON]
        label = skeleton[IDX_ACTION_LABEL]
        if label not in CLASSES:  # If invalid label, discard this image.
            continue
        labels_cnt[label] += 1

        #print(str(skeleton[4:]).strip("[]"))

        # Print
        if i == 1 or i % 100 == 0:
            print("{}/{}".format(i, num_skeletons))

        # -- Save to txt
        if os.path.exists(DST_ALL_SKELETONS_TXT):
            with open(DST_ALL_SKELETONS_TXT, 'a') as f:
                f.write(str(skeleton[4:]).strip("[]")+"\n")
        else:
            with open(DST_ALL_SKELETONS_TXT, 'w') as f:
                f.write(str(skeleton[4:]).strip("[]")+"\n")

    print("There are {} skeleton data.".format(len(arrays)))
    print("They are saved to {}".format(DST_ALL_SKELETONS_TXT))
    print("Number of each action: ")
    for label in CLASSES:
        print("    {0}: {1}".format(label,labels_cnt[label]))
