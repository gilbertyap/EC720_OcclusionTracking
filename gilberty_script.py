# Gilbert Yap
# EC720 Digital Video Processing
# Final Project
# Boston University, Fall 2021

# Assumes that this file will be placed into the dark_net folder

import os
import random
import csv
import numpy as np
import cv2

import darknet
import darknet_images
from kalmanfilter import KalmanFilter


def main():
    # Set up the Kalman filter object
    # TODO - What is the frame rate of the video?
    frame_rate = 1/30

    # TODO - For now, only interested in object center position, but will move
    # tracking object velocity for identity after

    # Object dynamics
    # Using parameters from Heimbach et al.
    F = np.array([[1, 0, frame_rate, 0], [0, 1, 0, frame_rate], [0, 0, frame_rate, 0], [0, 0, 0, frame_rate]])

    # Observation
    H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])

    #  Covariance of the process noise
    # TODO - better understand this
    Q = np.array([[1, 0, 6, 0], [0, 1, 0, 6], [6, 0, 2, 0], [0, 6, 0, 2]])

    # Covariance of the observation noise
    noise_cov = 0.5
    R = np.multiply(noise_cov, np.eye(2))

    kf = KalmanFilter(F = F, H = H, Q = Q, R = R)

    # Randomizes the bounding box color or something?
    random.seed(3)

    # TODO - Remove this hard coding
    files = darknet_images.load_images('c1_seq1.txt')
    conf_thresh=0.5

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
    for file in files:
        # TODO - Is there way to non-person detections?
        image, detections = darknet_images.image_detection(file, network, class_names,
                                                            class_colors,thresh=conf_thresh)
        if not first_detection_found:
            img_size = image.shape
            # Only want detections that are "person"
            for label, confidence, bbox in detections:
                if label == 'person':
                    bbox_left, bbox_top, bbox_right, bbox_bottom = darknet.bbox2points(bbox)
                    new_center = np.round(np.array([bbox_top+(bbox_bottom-bbox_top)/2, bbox_left+(bbox_right-bbox_left)/2]))
                    kf.update(new_center)

                    # Show the initial frame
                    # cv2.imshow('image',image)
                    # cv2.waitKey(0)
                    # cv2.destroyAllWindows()

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
                if label == 'person':
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
            pred_bbox_right= int(np.amin([img_size[1], prediction[1]+bbox_half_width]))
            pred_bbox_top= int(np.amax([0, prediction[0]-bbox_half_height]))
            pred_bbox_bottom = int(np.amin([img_size[1], prediction[0]+bbox_half_height]))

            cv2.rectangle(image, (pred_bbox_left, pred_bbox_top), (pred_bbox_right, pred_bbox_bottom), class_colors['prediction'], 1)
            cv2.putText(image, 'prediction',
                        (pred_bbox_left, pred_bbox_top - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        class_colors['prediction'], 2)

            cv2.imshow('image',image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            # If the person isn't found, the prediction is fed as a measurement
            # TODO - This doesn't seem to be working as expected?
            if not person_found:
                # new_center = prediction+(prediction-prev_center)
                # kf.update(new_center)
                kf.update(prediction)


if __name__ == "__main__":
    main()
