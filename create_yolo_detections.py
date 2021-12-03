# Gilbert Yap
# EC720 Digital Video Processing
# Final Project
# Boston University, Fall 2021

# Assumes that this file will be placed into the darknet folder

import os
import random
import numpy as np
import cv2

import darknet
import darknet_images

def main():
    # Randomizes the bounding box color or something?
    random.seed(3)

    # TODO - Remove this hard coding
    files = darknet_images.load_images('david3.txt')

    # Increasing the confidence threshold makes the person shape more reliable?
    conf_thresh=0.5

    # Using the pre-trained weights and cfg from GitHub. Trained on MS COCO
    network, class_names, class_colors = darknet.load_network(
        './/cfg//yolov4.cfg',
        './/cfg//coco.data',
        './/yolov4.weights'
    )

    # Get the original image size and the YOLO resized size
    org_image_size = (cv2.imread(files[0]).shape)
    image, detections = darknet_images.image_detection(files[0], network, ['person'],
                                                        class_colors,thresh=conf_thresh)
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
                                                                class_colors,thresh=conf_thresh)

            for label, confidence, bbox in detections:
                bbox_left, bbox_top, bbox_right, bbox_bottom = darknet.bbox2points(bbox)
                scaled_bbox_left = np.floor((scaling_factor_x*bbox_left) - (scaling_factor_x - 1))
                scaled_bbox_top = np.floor((scaling_factor_y*bbox_top) - (scaling_factor_x - 1))
                scaled_bbox_right = np.ceil(scaling_factor_x*bbox_right)
                scaled_bbox_bottom = np.ceil(scaling_factor_y*bbox_bottom)
                # cv2.rectangle(blank_frame, (int(scaled_bbox_left), int(scaled_bbox_top)), (int(scaled_bbox_right), int(scaled_bbox_bottom)), [0,0, 255], 1)
                # cv2.imwrite('.//det_gen//'+'{:05d}'.format(frame_counter)+'.jpg',blank_frame)

                print('Writing frame {} detections'.format(frame_counter))
                det_file.write('{},-1,{},{},{},{},{}\n'.format(frame_counter, scaled_bbox_left, scaled_bbox_top, scaled_bbox_right, scaled_bbox_bottom, confidence))

            frame_counter += 1
    print('Finished writing frame detections')

if __name__ == "__main__":
    main()
