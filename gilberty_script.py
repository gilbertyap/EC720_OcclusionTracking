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
    frame_rate = 1.0/30

    # TODO - For now, only interested in object center position, but will move
    # tracking object velocity for identity after

    # Object dynamics
    # Using parameters from Heimbach et al.
    F = np.array([[1, 0, frame_rate, 0], [0, 1, 0, frame_rate], [0, 0, frame_rate, 0], [0, 0, 0, frame_rate]])

    # Observation
    H = np.array([[1, 0, 0, 0],[0, 1, 0, 0]])

    #  Covariance of the process noise
    Q = np.array([[1, 0, 6, 0], [0, 1, 0, 6], [6, 0, 2, 0], [0, 6, 0, 2]])

    # Covariance of the observation noise
    noise_cov = 0.5
    R = np.multiply(noise_cov,np.eye(2))

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
        './/yolov4.weights',
        batch_size=batch_size
    )

    # TODO - This assumes that there is only one person in the image
    # Go through every image and get the bounding box coordinates of the person
    prev_center = -1*np.ones([1,2])
    new_center = -1*np.ones([1,2])
    bbox_width = 0
    bbox_height = 0
    first_detection_found = False
    for file in files:
        image = cv2.imread(file)
        image, detections = darknet_images.image_detection(file, network, class_names,
                                               class_colors,thresh=conf_thresh)

        if not first_detection_found:
           # "detections" has three fields: label, confidence, bbox
           # Only want detections that are "person"
           for label, confidence, bbox in detections:
               if label == 'person':
                   # Use the bounding box coordinates center
                   xmin, ymin, xmax, ymax = darknet.bbox2points(bbox)
                   new_center = np.array([(ymax-ymin)/2, (xmax-xmin)/2])
                   bbox_width = xmax - xmin
                   bbox_height = ymax - ymin
                   darknet.draw_boxes(detections,image,class_colors)

                   # cv2.rectangle(image, (xmin, ymin), (xmax, ymax), class_colors['person'], 1)
                   # cv2.putText(image, "{} [{:.2f}]".format('person', float(confidence)),
                #               (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                #               class_colors[label], 2)
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
           # either update the Kalman filter with the measurement or make a
           # prediction on their position if the person is not detected
           person_found = False
           for label, confidence, bbox in detections:
               if label == 'person':
                   person_found = True

                   xmin, ymin, xmax, ymax = darknet.bbox2points(bbox)
                   new_center = np.array([(ymax-ymin)/2, (xmax-xmin)/2])
                   prediction = np.dot(H,  kf.predict())[0]
                   print(new_center)
                   print(prediction)

                   # cv2.rectangle(image, (xmin, ymin), (xmax, ymax), class_colors['person'], 1)
                   # cv2.putText(image, "{} [{:.2f}]".format('person', float(confidence)),
                #               (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                #               class_colors[label], 2)
                   # cv2.imshow('image',image)
                   # cv2.waitKey(0)
                   # cv2.destroyAllWindows()

                   kf.update(new_center)
                   prev_center = new_center

                   # Only care about the first detection of person
                   break

               # If no person is found in the frame
               else:
                   # TODO - I'm not sure that feeding the prediction into the update is correct but *shrug*
                   prediction = np.dot(H,  kf.predict())[0]
                   kf.update(prediction)
                   print('Predicted center below')
                   print(prediction)
                   print('done with prediction')

           # Use prediction
           # TODO - how many number of frames is sustainable to keep using prediction?
           # if not person_found:


if __name__ == "__main__":
    # unconmment next line for an example of batch processing
    # batch_detection_example()
    main()
