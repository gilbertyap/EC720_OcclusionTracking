cp create_yolo_detections.py ../darknet/create_yolo_detections.py

cd ../darknet
python3 create_yolo_detections.py --img_dir '/home/gilbert/Downloads/David3/img/'

# mkdir ../darknet/dets/img/det
mv det.txt ./dets/img/det

cd ../deep_sort
source ./env/bin/activate
python3 ./tools/generate_detections.py --model '/home/gilbert/Downloads/deep_sort/resources/networks/mars-small128.pb' --mot_dir '/home/gilbert/Downloads/David3/' --detection_dir '/home/gilbert/Downloads/darknet/dets' --output_dir '/home/gilbert/Downloads/EC720_OcclusionTracking/dets/'
# python3 deep_sort_app.py --sequence_dir='../David3/' --detection_file='../EC720_OcclusionTracking/dets/img.npy' --min_confidence=0.9 --nn_budget=100 --output_file='../EC720_OcclusionTracking/deep_sort/ds_results.txt' --display=True
python3 deep_sort_app.py --sequence_dir='../David3/' --detection_file='../EC720_OcclusionTracking/dets/img.npy' --output_file='../EC720_OcclusionTracking/deep_sort/ds_results.txt' --display=False

# TODO - Take a look at the other deep_sort parameters to try
