import os
import random
import darknet
import darknet_images

# Assumes that this file will be placed into the dark_net folder

def main():

    random.seed(3)

    batch_size=3
    # TODO - batch_size needs to be configurable? it's the number of images "to be processed in one batch"
    network, class_names, class_colors = darknet.load_network(
        './/cfg//yolov4.cfg',
        './/cfg//coco.data',
        './/yolov4.weights',
        batch_size=batch_size
    )

    # TODO - Change paths
    conf_thresh=0.5
    image_names = ['data/horses.jpg', 'data/horses.jpg', 'data/eagle.jpg']
    images = [cv2.imread(image) for image in image_names]
    images, detections,  = batch_detection(network, images, class_names,
                                           class_colors, batch_size=batch_size,
                                           thresh=conf_thresh)
    for name, image in zip(image_names, images):
        cv2.imwrite(name.replace("data/", ""), image)
    print(detections)


if __name__ == "__main__":
    # unconmment next line for an example of batch processing
    # batch_detection_example()
    main()
