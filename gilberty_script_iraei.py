# Gilbert Yap
# EC720 Digital Video Processing
# Final Project
# Boston University, Fall 2021

# Assumes that this file will be placed into the darknet folder

import os
import random
import numpy as np
import cv2
import argparse

import darknet
import darknet_images
from kalmanfilter import KalmanFilter

def calc_angle(x,y):
    # offset to prevent NaN value as x approaches 0
    offset = 0.000001
    abs_x = np.abs(x)+offset
    abs_y = np.abs(y)

    if x >= 0 and y >=0:
        return np.arctan(abs_y/(abs_x))
    elif x < 0 and y>=0:
        return (np.arctan(abs_y/(abs_x)) + (np.pi/2))
    elif x < 0 and y < 0:
        return (np.arctan(abs_y/(abs_x)) + (np.pi))
    elif x>=0 and y < 0:
        return (np.arctan(abs_y/(abs_x)) + (3*np.pi/2))


# Based on image_detection function from darknet_images.py
# Just removes the drawing of the bounding box
def darknet_det(image_path, network, class_names, class_colors, thresh):
    # Darknet doesn't accept numpy images.
    # Create one with image we reuse for each detect
    width = darknet.network_width(network)
    height = darknet.network_height(network)
    darknet_image = darknet.make_image(width, height, 3)

    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (width, height),
                               interpolation=cv2.INTER_LINEAR)

    darknet.copy_image_from_bytes(darknet_image, image_resized.tobytes())
    detections = darknet.detect_image(network, class_names, darknet_image, thresh=thresh)
    darknet.free_image(darknet_image)
    return detections

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


# def main():
def main(img_dir, min_conf, gt_path, sort_path, output_dir, display):
    # Set up various thresholds
    bbox_dist_thresh = 85 #20
    vel_thresh = 10
    angle_thresh = 3*np.pi/4

    # missed_counter will represent the number of frames that the same person was not detected
    missed_counter = 0
    missed_counter_thresh = 0.5*30 # allow the object to be missing for 0.5 seconds at 30 fps

    # Set up the Kalman filter object
    frame_rate = 1/30

    # Object dynamics
    # Using parameters from Iraei et al.
    # Their method predicts postion and velocity and assumes that acceleration is constant but very small
    F = np.array([[1, 0, frame_rate, 0, frame_rate/2, 0],
                    [0, 1, 0, frame_rate, 0 , frame_rate/2],
                    [0, 0, 1, 0, frame_rate, 0],
                    [0, 0, 0, 1, 0, frame_rate],
                    [0, 0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 0, 1]])

    # Observation - only retrieving the position
    H = np.array([[1, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0 , 0]])

    # Covariance of the process noise
    # By using identity matrix, the noise of each feature (true variance of each feature) to track is independent
    # This allows some tolerance in constant velocity/accerlation model
    Q = np.eye(6)

    # Covariance of the observation noise
    # This covariance should change if the we expect the detector to make bad guesses about bbox center
    noise_cov = 10
    R = np.multiply(noise_cov, np.eye(2))

    kf = KalmanFilter(F = F, H = H, Q = Q, R = R)

    # Randomizes the bounding box color or something
    random.seed(3)

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

    # Add a prediction window to the classes, red bounding box
    class_colors['prediction']= (0,0,255)

    # Go through every image and get the bounding box gt_coordinates of the person
    prev_center = -1*np.ones([1,2])
    prev_vel = -1*np.ones([1,2])
    prev_angle = -1
    bbox_left = 0
    bbox_right= 0
    bbox_top= 0
    bbox_bottom = 0
    prev_half_width= 0
    prev_half_height = 0
    first_detection_found = False

    # For files with ground truth, get the ground truth bounding box
    # TODO - specify a ground truth format in the README
    if gt_path is not None:
        gt_file = open(gt_path, 'r')
        # Start a text file that records IoU and BBOX center distance accuracy
        res_file = open('result_proj.txt', 'w')
        res_file.write('Frame Id, IoU, Bbox Center Precision\n')

    if sort_path is not None:
        sort_file = open(sort_path, 'r')
        # Save results for Deep SORT for comparison to current method
        sort_res_file = open('result_sort.txt', 'w')
        sort_res_file.write('Frame Id, IoU, Bbox Center Precision\n')

    org_image_size = (cv2.imread(files[0]).shape)

    # Get the shape of the resized YOLO frame
    image, detections = darknet_images.image_detection(files[0], network, class_names,
                                                        class_colors,thresh=min_conf)
    img_size = image.shape
    scaling_factor_x = org_image_size[1]/img_size[1]
    scaling_factor_y = org_image_size[0]/img_size[0]

    # Initialize the image matrix
    image = np.zeros(img_size)

    frame_counter = 1
    last_kf_reset = 1

    # The frame number of SORT from the file, initilize to -1 to ensure it is below frame counter
    sort_frame_id = -1

    for file in files:
        # Only search for 'person' label
        detections = darknet_det(file, network, class_names, class_colors,thresh=min_conf)

        # print('missed_counter: {}'.format(missed_counter))

        # Boolean to track when to draw Deep SORT bbox
        draw_sort = False

        # Boolean to track when prediction is made without an observation
        use_pred = True

        if gt_path is not None:
            # For files with ground truth, show the ground truth bounding
            gt_coordinates = gt_file.readline().split(',')
            gt_bbox_left = int(gt_coordinates[0])
            gt_bbox_top = int(gt_coordinates[1])
            gt_bbox_right = gt_bbox_left+int(gt_coordinates[2])
            gt_bbox_bottom = gt_bbox_top+int(gt_coordinates[3][0:len(gt_coordinates[3])-1]) # this has a new line character at the end
            gt_bbox = (gt_bbox_left, gt_bbox_top, gt_bbox_right, gt_bbox_bottom)
            gt_center = np.round(np.array([gt_bbox_top+(gt_bbox_bottom-gt_bbox_top)/2, gt_bbox_left+(gt_bbox_right-gt_bbox_left)/2]))

        if sort_path is not None:
            # If Deep SORT detections are available, display them
            # Have to make sure that the displayed bounding box matches the current frame counter
            # If the sort_frame_id is greater than the frame_counter, hold onto the info until the frame_counter matches
            # else keep reading until sort_frame_id is >= the frame_counter
            while(sort_frame_id < frame_counter):
                sort_coordinates = sort_file.readline().split(',')
                sort_frame_id = int(sort_coordinates[0])

                sort_obj_id = int(sort_coordinates[1])

                # For multi person videos
                # These values are specific to Human3
                # if not ((sort_obj_id == 5) or (sort_obj_id == 9) or (sort_obj_id ==1)):
                #     continue

                sort_bbox_left = int(float(sort_coordinates[2]))
                sort_bbox_top = int(float(sort_coordinates[3]))
                sort_bbox_right = int(sort_bbox_left+float(sort_coordinates[4]))
                sort_bbox_bottom = int(sort_bbox_top+float(sort_coordinates[5]))
                sort_bbox = (sort_bbox_left, sort_bbox_top, sort_bbox_right, sort_bbox_bottom)
                sort_center = np.round(np.array([sort_bbox_top+(sort_bbox_bottom-sort_bbox_top)/2, sort_bbox_left+(sort_bbox_right-sort_bbox_left)/2]))

            # First if statement for single person video, second if statement for multi-person video
            if (sort_frame_id == frame_counter):
            # if (sort_frame_id == frame_counter) and ((sort_obj_id == 5) or (sort_obj_id == 9) or (sort_obj_id ==1)):
                draw_sort = True

        # Differentiate between the first detection and subsequent ones
        # No drawing to be done on the first detection so that the kalman filter can be initialized
        if not first_detection_found:# or ((frame_counter - last_kf_reset) < 6):
            if gt_path is None:
                for label, confidence, bbox in detections:
                    if label == 'person':
                        bbox_left, bbox_top, bbox_right, bbox_bottom = darknet.bbox2points(bbox)

                        new_center = np.round(np.array([bbox_top+(bbox_bottom-bbox_top)/2, bbox_left+(bbox_right-bbox_left)/2]))

                        new_vel = new_center # Best guess of velocity is that the object came from [0,0]
                        new_angle = calc_angle(new_vel[1], new_vel[0])

                        kf.update(new_center)

                        prev_half_width = bbox_right - new_center[1]
                        prev_half_height =  bbox_bottom - new_center[0]

                        prev_center = new_center
                        prev_vel = new_vel
                        prev_angle = new_angle
                        first_detection_found = True
                        last_kf_reset = frame_counter

                        # Only care about the first detection of person
                        break
            else:
                # Use the first GT position to initialize the KF
                new_center = gt_center
                # new_center = np.array([gt_center[0]*scaling_factor_x, gt_center[1]*scaling_factor_y])
                new_vel = new_center # Best guess of velocity is that the object came from [0,0]
                new_angle = calc_angle(new_vel[1], new_vel[0])

                kf.update(new_center)

                prev_half_width = bbox_right - new_center[1]
                prev_half_height =  bbox_bottom - new_center[0]

                prev_center = new_center
                prev_vel = new_vel
                prev_angle = new_angle
                first_detection_found = True
                last_kf_reset = frame_counter

        else:
            # If the person has already been identified once,
            # Draw both the true bounding box and the predicted bounding box
            prediction_pos = np.round(np.dot(H,  kf.predict())[0])

            # Get the predicted velocity and angle
            prediction_vel = prediction_pos - prev_center
            pred_vel_angle = calc_angle(prediction_vel[1], prediction_vel[0])

            # Check to see if there is a detection
            for label, confidence, bbox in detections:
                if label == 'person':
                    bbox_left, bbox_top, bbox_right, bbox_bottom = darknet.bbox2points(bbox)

                    # Get the center of the detected bounding box, and new velocity and angle
                    new_center = np.round(np.array([bbox_top+(bbox_bottom-bbox_top)/2, bbox_left+(bbox_right-bbox_left)/2]))
                    new_vel = new_center - prev_center
                    new_angle = calc_angle(new_vel[1], new_vel[0])

                    # The detected bbox has to be:
                    # 1. Be within a certain distance of the last center
                    # 2. Have a similar velocity as the previous detection
                    # 3. Have a similar angle as the previous detection
                    # 4. If an above condition is met, only skip if its been at least 6 frames since the KF initialized
                    # print('new {} prev {}, frame_counter {} last {}'.format(new_center, prev_center, frame_counter, last_kf_reset))
                    # print( '{} > {}?, {} > {}? '.format(( np.sqrt(np.sum((new_center - prev_center)**2))), bbox_dist_thresh, np.sqrt(np.sum((new_vel - prev_vel)**2)), vel_thresh) )
                    if ( (( np.sqrt(np.sum((new_center - prev_center)**2)) ) > bbox_dist_thresh) or \
                    ( (( np.sqrt(np.sum((new_vel - prev_vel)**2)) ) > vel_thresh) and ((frame_counter-last_kf_reset) > 6)) ):# or \
                    # (np.abs(new_angle - prev_angle) > angle_thresh) ):
                        # print('skipped')
                        continue

                    # Use the measurement bbox instead of prediction
                    use_pred = False

                    kf.update(new_center)

                    # Overwrite the previously saved values
                    prev_half_width = bbox_right - new_center[1]
                    prev_half_height =  bbox_bottom - new_center[0]
                    prev_center = new_center
                    prev_vel = new_vel
                    prev_angle = new_angle

                    # Reset the miss counter every time the person is detected
                    missed_counter = 0
                    break

            # Keep track of how many frames the person was not tracked
            if use_pred:
                missed_counter +=1
                # Predicted bounding box
                bbox_half_width = prev_half_width
                bbox_half_height = prev_half_height

                # Make the predicted bounding box
                disp_bbox_left = int(np.amax([0, prediction_pos[1]-bbox_half_width]))
                disp_bbox_right = int(np.amin([img_size[1]-1, prediction_pos[1]+bbox_half_width]))
                disp_bbox_top = int(np.amax([0, prediction_pos[0]-bbox_half_height]))
                disp_bbox_bottom = int(np.amin([img_size[1]-1, prediction_pos[0]+bbox_half_height]))
            else:
                # Predicted bounding box
                bbox_half_width = bbox_right - new_center[1]
                bbox_half_height =  bbox_bottom - new_center[0]

                # Make the predicted bounding box
                disp_bbox_left = int(np.amax([0, new_center[1]-bbox_half_width]))
                disp_bbox_right = int(np.amin([img_size[1]-1, new_center[1]+bbox_half_width]))
                disp_bbox_top = int(np.amax([0, new_center[0]-bbox_half_height]))
                disp_bbox_bottom = int(np.amin([img_size[1]-1, new_center[0]+bbox_half_height]))

            # If there are too many missed frames, reset the Kalman filter
            if missed_counter > missed_counter_thresh:
                kf = KalmanFilter(F = F, H = H, Q = Q, R = R)
                first_detection_found = False
                missed_counter = 0
                last_kf_reset = frame_counter

            # TODO - Hardcoded to the default YOLOv4 size
            image = cv2.resize(cv2.imread(file), (608, 608), interpolation=cv2.INTER_LINEAR)

            # Draw the prediction bounding box
            if display and (missed_counter < missed_counter_thresh):
                cv2.rectangle(image, (disp_bbox_left, disp_bbox_top), (disp_bbox_right, disp_bbox_bottom), class_colors['prediction'], 1)

            # Resize the prediciton box to the original image size
            scaled_disp_bbox_left = np.floor((scaling_factor_x*disp_bbox_left) - (scaling_factor_x - 1))
            scaled_disp_bbox_top = np.floor((scaling_factor_y*disp_bbox_top) - (scaling_factor_y - 1))
            scaled_disp_bbox_right = np.ceil(scaling_factor_x*disp_bbox_right)
            scaled_disp_bbox_bottom = np.ceil(scaling_factor_y*disp_bbox_bottom)
            scaled_disp_bbox = (scaled_disp_bbox_left, scaled_disp_bbox_top, scaled_disp_bbox_right, scaled_disp_bbox_bottom)
            scaled_center = np.round(np.array([scaled_disp_bbox_top+(scaled_disp_bbox_bottom-scaled_disp_bbox_top)/2, scaled_disp_bbox_left+(scaled_disp_bbox_right-scaled_disp_bbox_left)/2]))

            # Use the non-scaled prediction gt_coordinates to draw the box since frame scaling happens later
            if gt_path is not None:
                if display:
                    if not use_pred:
                        cv2.putText(image, 'YOLO v4 + KF (Measured)- IoU {:.3f}'.format(calculate_iou(gt_bbox, scaled_disp_bbox)),
                                (disp_bbox_left, disp_bbox_top - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                class_colors['prediction'], 2)
                    else:
                        cv2.putText(image, 'YOLO v4 + KF (Pred)- IoU {:.3f}'.format(calculate_iou(gt_bbox, scaled_disp_bbox)),
                                (disp_bbox_left, disp_bbox_top - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                class_colors['prediction'], 2)

                # Write IoU and center distance to file
                res_file.write('{},{},{}\n'.format(frame_counter,
                                                    calculate_iou(gt_bbox, scaled_disp_bbox),
                                                    calculate_bbox_center_dist(gt_center, scaled_center)))

            else:
                if display:
                    cv2.putText(image, 'YOLO v4 + KF',
                            (disp_bbox_left, disp_bbox_top - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            class_colors['prediction'], 2)

        # Resize the image to the original image size
        image = cv2.resize(image, (org_image_size[1], org_image_size[0]))

        # Draw GT bbox after the resizing since the gt_bbox is wrt to the original frame resolution
        if display:
            if gt_path is not None:
                cv2.rectangle(image, (gt_bbox_left, gt_bbox_top), (gt_bbox_right, gt_bbox_bottom), (0,255,0), 1)
                cv2.putText(image, 'GT',
                            (gt_bbox_left, gt_bbox_top - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0,255,0), 2)

        # For files with Deep SORT detections, show their bbox
        if draw_sort:
            if display:
                if (gt_path is not None) and (sort_path is not None):
                        cv2.rectangle(image, (sort_bbox_left, sort_bbox_top), (sort_bbox_right, sort_bbox_bottom), (255,0,0), 1)
                        cv2.putText(image, 'YOLO v4 + DS - IoU {:.3f}'.format(calculate_iou(gt_bbox, sort_bbox)),
                                (sort_bbox_right, sort_bbox_bottom - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (255,0,0), 2)


                elif (sort_path is not None):
                    cv2.rectangle(image, (sort_bbox_left, sort_bbox_top), (sort_bbox_right, sort_bbox_bottom), (255,0,0), 1)
                    cv2.putText(image, 'YOLO v4 + DS',
                                (sort_bbox_right, sort_bbox_bottom - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (255,0,0), 2)

            # Write Deep SORT IoU and bbox center to file
            if (gt_path is not None) :
                sort_res_file.write('{},{},{}\n'.format(frame_counter,
                                                    calculate_iou(gt_bbox, sort_bbox),
                                                    calculate_bbox_center_dist(gt_center, sort_center)))

        # Dispaly the bounding boxes and such
        if display:
            cv2.imshow('image',image)
            cv2.waitKey(1)
        else:
            print('Processed frame {}'.format(frame_counter))

        cv2.imwrite(output_dir+'{:05d}'.format(frame_counter)+'.jpg',image)
        frame_counter+=1

    if (gt_path is not None):
        gt_file.close()
        res_file.close()

    if (sort_path is not None):
        sort_file.close()
        sort_res_file.close()

def parse_args():
    parser = argparse.ArgumentParser(description="EC720 Occlusion Tracking")
    parser.add_argument(
        "--img_dir", help="Path to images directory",
        default=None, required=True)
    parser.add_argument(
        "--min_conf", help="Minimum confidence score", default=0.5, type=float)
    parser.add_argument(
        "--gt_path", help="Ground truth file path", default=None)
    parser.add_argument(
        "--sort_path", help="Deep SORT detections file path", default=None)
    parser.add_argument(
        "--output_dir", help="Path to output directory", default="./gen/")
    parser.add_argument('--display', dest='display', help='Display the output', action='store_true')
    parser.add_argument('--no_display', dest='display', help='Do not display the output', action='store_false')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args.img_dir, args.min_conf, args.gt_path, args.sort_path, args.output_dir, args.display)
