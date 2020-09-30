import cv2
import numpy as np
import os

VIDEO_DIR = '/home/albie/Gesture-Recognition/data/video/videos/'
LABEL_DIR = '/home/albie/Gesture-Recognition/data/video/labels/'
DST_ALL_SKELETONS_TXT = "/home/albie/anxious_popcorn/data_proc/UTD-MHAD/test_labels.txt"
if not os.path.isdir(VIDEO_DIR):
    os.makedirs(VIDEO_DIR)

files = folders = 0

for _, dirnames, filenames in os.walk(VIDEO_DIR):
    files += len(filenames)
    folders += len(dirnames)
#print "{:,} files, {:,} folders".format(files, folders)

"""
        1.  right arm swipe to the left         (swipt_left)
        2.  right arm swipe to the right        (swipt_right)
        3.  right hand wave                     (wave)
        4.  two hand front clap                 (clap)
        5.  right arm throw                     (throw)
        6.  cross arms in the chest             (arm_cross)
        7.  basketball shooting                 (basketball_shoot)
        8.  draw x                              (draw_x)
        9.  draw circle  (clockwise)            (draw_circle_CW)
        10. draw circle  (counter clockwise)    (draw_circle_CCW)
        11. draw triangle                       (draw_triangle)
        12. bowling (right hand)                (bowling)
        13. front boxing                        (boxing)
        14. baseball swing from right           (baseball_swing)
        15. tennis forehand swing               (tennis_swing)
        16. arm curl (two arms)                 (arm_curl)
        17. tennis serve                        (tennis_serve)
        18. two hand push                       (push)
        19. knock on door                       (knock)
        20. hand catch                          (catch)
        21. pick up and throw                   (pickup_throw)
"""
# action is the index array that you want to use for the action associated above

action = [1,2,3,4,5,6,7]

"""
    file_idx        :   referes to the number of files and folders
    last_file_idx   :   number of frames extracted from a single video files (output)
    current_frame   :   number of frames extracted from a single video files (process)

"""
# Change this if you change action
file_idx = 1
for action_idx in range(len(action)):
    subject_idx = 1
    rep_idx = 1

    while subject_idx < 9:
        """
            #Function used to rename

        video_label = ("a{0}_"+"s{1}_"+"t{2}_"+"color.avi") \
                .format(action[action_idx],subject_idx,rep_idx)

        INPUT_DIR = VIDEO_DIR+video_label
        OUTPUT_DIR = VIDEO_DIR + ("%s.avi" % file_idx)

        os.rename(INPUT_DIR,OUTPUT_DIR)


        """
            #Function used to write labels

        """


        with open(LABEL_DIR + ("%s.txt" % file_idx), "w") as label_file:

            if file_idx < (1*32+1):   label_choice = 'swipe left'
            elif file_idx < (2*32+1):   label_choice = 'swipe right'
            elif file_idx < (3*32+1):   label_choice = 'wave'
            elif file_idx < (4*32+1):   label_choice = 'clap'

            label_file.write(label_choice)
        """

        if file_idx < (1*32+1):   label_choice = 1
        elif file_idx < (2*32+1):   label_choice = 2
        elif file_idx < (3*32+1):   label_choice = 3
        elif file_idx < (4*32+1):   label_choice = 4
        elif file_idx < (5*32+1):   label_choice = 5
        elif file_idx < (6*32+1):   label_choice = 6
        elif file_idx < (7*32+1):   label_choice = 7

        if os.path.exists(DST_ALL_SKELETONS_TXT):
            with open(DST_ALL_SKELETONS_TXT, "a") as f:
                f.write(str(label_choice)+"\n")
        else:
            with open(DST_ALL_SKELETONS_TXT, 'w') as f:
                f.write(str(label_choice)+"\n")


        file_idx += 1
        if rep_idx is 4:
            subject_idx += 1
            rep_idx = 0
        if rep_idx < 5:
            rep_idx += 1
