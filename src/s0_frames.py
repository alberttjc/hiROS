import cv2
import numpy as np
import os

ROOT = os.path.dirname(os.path.abspath(__file__))+"/../"
VIDEO_DIR = '/home/albie/UTD-MHAD/'
OUTPUT_DIR = '/home/albie/anxious_popcorn/data/UTD-MHAD/'

if not os.path.isdir(VIDEO_DIR):
    os.makedirs(VIDEO_DIR)
if not os.path.isdir(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

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

action = [8]

"""
    file_idx        :   referes to the number of files and folders
    last_file_idx   :   number of frames extracted from a single video files (output)
    current_frame   :   number of frames extracted from a single video files (process)

"""
# Change this if you change action
file_idx = 224
for action_idx in range(len(action)):
    subject_idx = 1
    rep_idx = 1

    while subject_idx < 9:
        video_label = ("a{0}_"+"s{1}_"+"t{2}_"+"color.avi") \
                .format(action[action_idx],subject_idx,rep_idx)
        print(VIDEO_DIR+video_label)

        video_idx = VIDEO_DIR+video_label
        cap = cv2.VideoCapture(video_idx)

        file_idx += 1
        last_file_idx = 1
        FRAME_DIR = OUTPUT_DIR + ("/%s" % file_idx)

        if not os.path.isdir(FRAME_DIR):
            os.makedirs(FRAME_DIR)

        while (True):
            ret, frame = cap.read()

            # Save frame as a jpg file
            name = 'frame' + str(last_file_idx) + '.jpg'
            #print ('Creating: ' + name)

            if not ret:
                break

            cv2.imwrite(os.path.join(FRAME_DIR, name), frame)
            last_file_idx += 1

        if rep_idx is 4:
            subject_idx += 1
            rep_idx = 0
        if rep_idx < 5:
            rep_idx += 1

#release capture
cap.release()
print('done')
