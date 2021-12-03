# EC720_OcclusionTracking

## Object Tracking Under Occlusion Scenarios

This is the repository for Gilbert Yap's Boston Univeristy EC720 "Digital Video Processing" final project.

### Instructions for Custom Video

1. 
1. `python3 create_yolo_detections.py`
1. `python3 ./tools/generate_detections.py --model '/home/gilbert/Downloads/deep_sort/resources/networks/mars-small128.pb' --mot_dir '/home/gilbert/Downloads/David3/' --detection_dir '/home/gilbert/Downloads/darknet/dets' --output_dir /home/gilbert/Downloads/EC720_OcclusionTracking/dets/`
1. `python deep_sort_app.py --sequence_dir=../David3/ --detection_file=../EC720_OcclusionTracking/dets/david3.npy --min_confidence=0.5 --nn_budget=100 --display=True`


### Requirements

* Uses [YOLOv4](https://github.com/AlexeyAB/darknet) for object detection.
* The files in this repository files should be directly placed into the YOLO v4 folder it has been set up.
* Uses [Deep SORT](https://github.com/nwojke/deep_sort) as a benchmark tool
* For python requirements, use `pip install -r requirements.txt`. Even better, open up a virtual environment and install the requirements there.
