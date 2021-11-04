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
import KalmanFilter from kalman-filter


def main():
    # Set up the Kalman filter object
    # TODO - What is the frame rate of the video?
    frame_rate = 1.0/30

    # TODO - Set up the Kalman filter dynamics, etc.
    # Object dynamics
    # Want to track x pos, y pos, x velocity, y velocity
    F = np.array([[1, frame_rate, 0], [0, 1, frame_rate], [0, 0, 1]])
    # Observation model
    # TODO - For now, only interested in object center position, but will move
    # tracking object velocity for identity after
    H = np.array([1, 0, 0]).reshape(1, 3)
    #  Covariance of the process noise
    Q = np.array([[0.05, 0.05, 0.0], [0.05, 0.05, 0.0], [0.0, 0.0, 0.0]])
    # Covariance of the observation noise
    R = np.array([0.5]).reshape(1, 1)

    x = np.linspace(-10, 10, 100)
    measurements = - (x**2 + 2*x - 2)  + np.random.normal(0, 2, 100)

    kf = KalmanFilter(F = F, H = H, Q = Q, R = R)

    # TODO - Currently the prediction is the x-y coordinate, but in may want
    # x and y velocity as well
    predictions = np.array([1,2])

    # Randomizes the bounding box color or something?
    random.seed(3)

    # TODO - Remove this hard coding
    csv_file = open('c1_seq1.csv')
    conf_thresh=0.5

    reader = csv.reader(csv_file, delimiter=',')
    batch_size=len(reader)

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
    first_detection_found = False
    for file in reader:
        image = cv2.imread(file)
        image, detections = darknet_images.image_detection(file, network, class_names,
                                               class_colors, batch_size=batch_size,
                                               thresh=conf_thresh)

       if not first_detection_found:
           # "detections" has three fields: label, confidence, bbox
           # Only want detections that are "person"
           for label, confidence, bbox in detections:
               if label == 'person':
                   # Use the bounding box coordinates center
                   xmin, ymin, xmax, ymax = bbox2points(bbox)
                   new_center = np.array([(ymax-ymin)/2, (xmax-xmin)/2)])

                   if np.array_equal(prev_center, -1*np.ones([1,2])):
                       # On the very first frame, simply update the center to a valid position
                       prev_center = new_center
                       first_detection = True
                       continue;
                   else:
                       # For every other frame, update Kalman filter
                       predictions.append(np.dot(H,  kf.predict())[0])
                       # TODO - Make sure that new center is correct
                       kf.update(new_center)

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

                   xmin, ymin, xmax, ymax = bbox2points(bbox)
                   new_center = np.array([(ymax-ymin)/2, (xmax-xmin)/2)])
                   predictions.append(np.dot(H,  kf.predict())[0])

                   # TODO - Make sure that new center is correct
                   kf.update(new_center)
                   prev_center = new_center

                   # Only care about the first detection of person
                   break

           # Use prediction
           # TODO - how many number of frames is sustainable to keep using prediction?
           if not person_found:


if __name__ == "__main__":
    # unconmment next line for an example of batch processing
    # batch_detection_example()
    main()
