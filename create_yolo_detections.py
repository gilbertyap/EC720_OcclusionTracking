# Gilbert Yap
# EC720 Digital Video Processing
# Final Project
# Boston University, Fall 2021

# Assumes that this file will be placed into the darknet folder

# TODO - Should Yolo v4 really be used as the detections for Deep SORT?

import os
import random
import numpy as np
import cv2
import argparse

import darknet
import darknet_images

def main(img_dir, min_conf):
    # Make a text file that lists all of the files
    # TODO - Handle relative pathing
    filenames = os.listdir(img_dir)
    filenames.sort()
    with open('files.txt', 'w') as f:
        for name in filenames:
            f.write(img_dir+name+'\n')

    files = darknet_images.load_images('files.txt')

    # Using the pre-trained weights and cfg from GitHub. Trained on MS COCO
    network, class_names, class_colors = darknet.load_network(
        './/cfg//yolov4.cfg',
        './/cfg//coco.data',
        './/yolov4.weights'
    )

    # Get the original image size and the YOLO resized size
    org_image_size = (cv2.imread(files[0]).shape)
    image, detections = darknet_images.image_detection(files[0], network, ['person'],
                                                        class_colors,thresh=min_conf)
    image_size = image.shape

    scaling_factor_x = org_image_size[1]/image_size[1]
    scaling_factor_y = org_image_size[0]/image_size[0]
    print('Scaling factors {}, {}'.format(scaling_factor_x,scaling_factor_y))
    frame_counter = 1
    blank_frame = np.zeros(image_size)

    with open('det.txt', 'w') as det_file:
        for file in files:
            # Only search for 'person' label
            image, detections = darknet_images.image_detection(file, network, ['person'],
                                                                class_colors,thresh=min_conf)

            for label, confidence, bbox in detections:
                bbox_left, bbox_top, bbox_right, bbox_bottom = darknet.bbox2points(bbox)

                scaled_bbox_left = np.floor((scaling_factor_x*bbox_left) - (scaling_factor_x - 1))
                scaled_bbox_top = np.floor((scaling_factor_y*bbox_top) - (scaling_factor_x - 1))
                scaled_bbox_right = np.ceil(scaling_factor_x*(bbox_right-bbox_left))
                scaled_bbox_bottom = np.ceil(scaling_factor_y*(bbox_bottom-bbox_top))

                print('Writing frame {} detections'.format(frame_counter))
                # The -1 is an ID given to the object
                # TODO - Will need to change label id if doing multi-object
                det_file.write('{},1,{},{},{},{},{}\n'.format(frame_counter, scaled_bbox_left, scaled_bbox_top, scaled_bbox_right, scaled_bbox_bottom, float(confidence)/100))

            frame_counter += 1
    print('Finished writing frame detections')

def parse_args():
    parser = argparse.ArgumentParser(description="Deep SORT")
    parser.add_argument(
        "--img_dir", help="Path to images directory",
        default=None, required=True)
    parser.add_argument(
        "--min_conf", help="Minimum confidence score", default=0.5, type=float)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args.img_dir, args.min_conf)
    # main()
