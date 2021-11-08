# Gilbert Yap
# EC720 Digital Video Processing
# Final Project
# Boston University, Fall 2021

# Assumes that this file will be placed into the dark_net folder

import os
import random
import numpy as np
import cv2

import darknet
import darknet_images
from kalmanfilter import KalmanFilter

def main():
    # Set up the Kalman filter object
    frame_rate = 1/30

    # Object dynamics
    # Using parameters from Iraei et al.
    # Their method predicts postion and velocity and assumes that acceleration is constant (I think)
    F = np.array([[1, 0, frame_rate, 0, frame_rate/2, 0], [0, 1, 0, frame_rate, 0 , frame_rate/2], [0, 0, 1, 0, frame_rate, 0], [0, 0, 0, 1, 0, frame_rate], [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1]])

    # Observation - only retrieving the position
    H = np.array([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0 , 0]])

    #  Covariance of the process noise
    # TODO - Iraei does an identity matrix, but Heimbach has 1 and 6 in the matrix?
    Q = np.eye(6)

    # Covariance of the observation noise
    # Lowering the covariance value (<1) makes the prediction closer to the true bbox
    # Increasing the covariance makes the prediction more stable but sometimes slower to catch up
    # Tweaked the value from 15 to 0.5 since it seemed track the YOLO box better
    noise_cov = 0.75
    R = np.multiply(noise_cov, np.eye(2))

    kf = KalmanFilter(F = F, H = H, Q = Q, R = R)

    # Randomizes the bounding box color or something?
    random.seed(3)

    # TODO - Remove this hard coding
    files = darknet_images.load_images('david3.txt')

    # Increasing the confidence threshold makes the person shape more reliable?
    conf_thresh=0.6

    # Using the pre-trained weights and cfg from GitHub. Trained on MS COCO
    network, class_names, class_colors = darknet.load_network(
        './/cfg//yolov4.cfg',
        './/cfg//coco.data',
        './/yolov4.weights'
    )

    # Add a prediction window to the classes, yellow bounding box
    class_colors['prediction']= (255,255,0)

    # TODO - This assumes that there is only one person in the image
    # Go through every image and get the bounding box coordinates of the person
    prev_center = -1*np.ones([1,2])
    new_center = -1*np.ones([1,2])
    bbox_left = 0
    bbox_right= 0
    bbox_top= 0
    bbox_bottom = 0
    first_detection_found = False

    # For files with ground truth, get the ground truth bounding box
    gt_file = open('david3_gt.txt', 'r')
    org_image_size = (cv2.imread(files[0]).shape)

    for file in files:
        # Only search for 'person' label
        image, detections = darknet_images.image_detection(file, network, ['person'],
                                                            class_colors,thresh=conf_thresh)

        if not first_detection_found:
            img_size = image.shape
            for label, confidence, bbox in detections:
                bbox_left, bbox_top, bbox_right, bbox_bottom = darknet.bbox2points(bbox)
                new_center = np.round(np.array([bbox_top+(bbox_bottom-bbox_top)/2, bbox_left+(bbox_right-bbox_left)/2]))
                # Best guess of velocity is that the object came from [0,0]
                kf.update(new_center)

                # Makes the assumption that the prev_center has not been set
                prev_center = new_center
                first_detection_found = True

                # Only care about the first detection of person
                break

        else:
            # If the person has already been identified once,
            # Draw both the true bounding box and the predicted bounding box
            prediction = np.round(np.dot(H,  kf.predict())[0])

            person_found = False
            for label, confidence, bbox in detections:
                person_found = True
                bbox_left, bbox_top, bbox_right, bbox_bottom = darknet.bbox2points(bbox)
                new_center = np.round(np.array([bbox_top+(bbox_bottom-bbox_top)/2, bbox_left+(bbox_right-bbox_left)/2]))

                kf.update(new_center)
                prev_center = new_center

                # Only care about the first detection of person
                break

            # Predicted bounding box
            bbox_half_width = bbox_right - prev_center[1]
            bbox_half_height =  bbox_bottom - prev_center[0]

            # Make the predicted bounding box
            pred_bbox_left = int(np.amax([0, prediction[1]-bbox_half_width]))
            pred_bbox_right= int(np.amin([img_size[1]-1, prediction[1]+bbox_half_width]))
            pred_bbox_top= int(np.amax([0, prediction[0]-bbox_half_height]))
            pred_bbox_bottom = int(np.amin([img_size[1]-1, prediction[0]+bbox_half_height]))

            cv2.rectangle(image, (pred_bbox_left, pred_bbox_top), (pred_bbox_right, pred_bbox_bottom), class_colors['prediction'], 1)
            cv2.putText(image, 'prediction',
                        (pred_bbox_left, pred_bbox_top - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        class_colors['prediction'], 2)

        # Resize the image to the original image size
        image = cv2.resize(image, (org_image_size[1], org_image_size[0]))

        # For files with ground truth, show the ground truth bounding
        coordinates = gt_file.readline().split(',')
        gt_bbox_left = int(coordinates[0])
        gt_bbox_top = int(coordinates[1])
        gt_bbox_right = gt_bbox_left+int(coordinates[2])
        gt_bbox_bottom = gt_bbox_top+int(coordinates[3][0:len(coordinates[3])-1]) # this has a new line character at the end
        cv2.rectangle(image, (gt_bbox_left, gt_bbox_top), (gt_bbox_right, gt_bbox_bottom), (0,255,0), 1)
        cv2.putText(image, 'ground-truth',
                    (gt_bbox_left, gt_bbox_top - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0,255,0), 2)

        cv2.imshow('image',image)
        cv2.waitKey(50)


    gt_file.close()

if __name__ == "__main__":
    main()
