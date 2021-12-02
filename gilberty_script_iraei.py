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
from kalmanfilter import KalmanFilter

# TODO:
# Clean up code with ground truth bounding boxes
# Set up commandline arguement parser to make it easier to run on a specific dataset
# Streamline the Deep SORT process
# Multi-object tracking based on appearance and/or location/velocity
# Add bbox to the Kalman filter to help with z-axis movement

# Since the ground-truth box does not change shape,
# calculating the square error of the distance between centers may be a better metric
def calculate_bbox_center_dist(gt_center, pred_center):
    return np.sqrt((gt_center[0]-pred_center[0])**2 + (gt_center[1]-pred_center[1])**2)

# Assumes the following tuple format
# gt_bbox = (gt_left, gt_top, gt_right, gt_bottom)
# pred_bbox = (bbox_left, bbox_top, bbox_right, bbox_bottom)
def calculate_iou(gt_bbox, pred_bbox):
    # Determine if the ground truth is contained in the prediction or vice versa
    hor_diff = min(gt_bbox[2], pred_bbox[2]) - max(gt_bbox[0], pred_bbox[0]) + 1
    ver_diff = min(gt_bbox[3], pred_bbox[3]) - max(gt_bbox[1], pred_bbox[1]) + 1
    intersection = max(0, hor_diff) * max(0, ver_diff)

    # Sum of the areas minus the intersection, since it gets counted twice
    union = ((gt_bbox[2]-gt_bbox[0] + 1) * (gt_bbox[3]-gt_bbox[1] + 1)) + ((pred_bbox[2]-pred_bbox[0] + 1) * (pred_bbox[3]-pred_bbox[1] + 1)) - intersection

    return float(intersection/union)


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

    # Randomizes the bounding box color or something
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

    # Get the shape of the resized YOLO frame
    # TODO - Need to have a check incase the input image is smaller than the resize
    image, detections = darknet_images.image_detection(files[0], network, ['person'],
                                                        class_colors,thresh=conf_thresh)
    image_size = image.shape
    frame_counter = 0

    for file in files:
        # Only search for 'person' label
        image, detections = darknet_images.image_detection(file, network, ['person'],
                                                            class_colors,thresh=conf_thresh)

        # For files with ground truth, show the ground truth bounding
        coordinates = gt_file.readline().split(',')
        gt_bbox_left = int(coordinates[0])
        gt_bbox_top = int(coordinates[1])
        gt_bbox_right = gt_bbox_left+int(coordinates[2])
        gt_bbox_bottom = gt_bbox_top+int(coordinates[3][0:len(coordinates[3])-1]) # this has a new line character at the end
        gt_bbox = (gt_bbox_left, gt_bbox_top, gt_bbox_right, gt_bbox_bottom)
        gt_center = np.round(np.array([gt_bbox_top+(gt_bbox_bottom-gt_bbox_top)/2, gt_bbox_left+(gt_bbox_right-gt_bbox_left)/2]))

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
                # TODO - For multi-object tracking, a list of bboxes and confidence scores need to be tracked
                break

        else:
            # If the person has already been identified once,
            # Draw both the true bounding box and the predicted bounding box
            prediction = np.round(np.dot(H,  kf.predict())[0])

            for label, confidence, bbox in detections:
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
            pred_bbox_right = int(np.amin([img_size[1]-1, prediction[1]+bbox_half_width]))
            pred_bbox_top = int(np.amax([0, prediction[0]-bbox_half_height]))
            pred_bbox_bottom = int(np.amin([img_size[1]-1, prediction[0]+bbox_half_height]))
            cv2.rectangle(image, (pred_bbox_left, pred_bbox_top), (pred_bbox_right, pred_bbox_bottom), class_colors['prediction'], 1)

            # Resize the prediciton box to the original image size
            # TODO - Is this scaling method correct?
            scaling_factor_x = org_image_size[1]/image_size[1]
            scaling_factor_y = org_image_size[0]/image_size[0]
            scaled_pred_bbox_left = scaling_factor_x*pred_bbox_left
            scaled_pred_bbox_top = scaling_factor_y*pred_bbox_top
            scaled_pred_bbox_right = scaling_factor_x*pred_bbox_right
            scaled_pred_bbox_bottom = scaling_factor_y*pred_bbox_bottom
            scaled_pred_bbox = (scaled_pred_bbox_left, scaled_pred_bbox_top, scaled_pred_bbox_right, scaled_pred_bbox_bottom)

            # Use the non-scaled prediction coordinates to draw the box since frame scaling happens later
            cv2.putText(image, 'Pred - IoU {:.3f}'.format(calculate_iou(gt_bbox, scaled_pred_bbox)),
                        (pred_bbox_left, pred_bbox_top - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        class_colors['prediction'], 2)

            # TODO - Determine if it's better to use IOU or the Euclidean distance between ground-truth and prediction bounding boxes
            # cv2.putText(image, 'Pred - IoU {:.3f}'.format(calculate_bbox_center_dist(gt_center, prev_center)),
            #             (pred_bbox_left, pred_bbox_top - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
            #             class_colors['prediction'], 2)

        # Resize the image to the original image size
        image = cv2.resize(image, (org_image_size[1], org_image_size[0]))

        # For files with ground truth, show the ground truth bounding box
        cv2.rectangle(image, (gt_bbox_left, gt_bbox_top), (gt_bbox_right, gt_bbox_bottom), (0,255,0), 1)
        cv2.putText(image, 'GT',
                    (gt_bbox_left, gt_bbox_top - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0,255,0), 2)

        # Display the image and write it to the 'gen' folder
        cv2.imshow('image',image)
        cv2.waitKey(33)
        cv2.imwrite('.//gen//'+'{:05d}'.format(frame_counter)+'.jpg',image)
        frame_counter+=1


    gt_file.close()

if __name__ == "__main__":
    main()
