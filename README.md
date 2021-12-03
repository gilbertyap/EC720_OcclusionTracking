# EC720_OcclusionTracking

## Object Tracking Under Occlusion Scenarios

This is the repository for Gilbert Yap's Boston Univeristy EC720 "Digital Video Processing" final project.

### Usage guidelines
1. `chmod +x ./run_deepsort.sh` and `./run_deepsort.sh`
1. `chmod +x ./run_kf_method.sh` and `./run_kf_method.sh`
Look at the above mentioned scripts for how to use on custom videos

### Requirements
* Uses [YOLOv4](https://github.com/AlexeyAB/darknet) for object detection.
* The files in this repository files should be directly placed into the YOLO v4 folder it has been set up.
* Uses [Deep SORT](https://github.com/nwojke/deep_sort) as a benchmark tool
* For python requirements, use `pip install -r requirements.txt`. Even better, open up a virtual environment and install the requirements there.
* Place this repository in a parallel folder to darknet and deep_sort
