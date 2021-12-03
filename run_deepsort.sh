cp creat_yolo_detections.py ../darknet/creat_yolo_detections.py

cd ../darknet
python3 create_yolo_detections.py

mkdir ../darknet/dets/img/det
mv det.txt ./dets/img/det

cd ../deep_sort
source ./env/bin/activate
python3 ./tools/generate_detections.py --model '/home/gilbert/Downloads/deep_sort/resources/networks/mars-small128.pb' --mot_dir '/home/gilbert/Downloads/David3/' --detection_dir '/home/gilbert/Downloads/darknet/dets' --output_dir /home/gilbert/Downloads/EC720_OcclusionTracking/dets/
python deep_sort_app.py --sequence_dir=../David3/ --detection_file=../EC720_OcclusionTracking/dets/img.npy --min_confidence=90 --nn_budget=100 --display=True
