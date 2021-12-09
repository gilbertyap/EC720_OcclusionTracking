cp create_yolo_detections.py ../darknet/create_yolo_detections.py

cd ../darknet
python3 create_yolo_detections.py --img_dir '../David3/img/'
# python3 create_yolo_detections.py --img_dir '../Subway/img/'
# python3 create_yolo_detections.py --img_dir '../Human3/img/'
# python3 create_yolo_detections.py --img_dir '../Challenge1_SEQ1/person2/'

mkdir ../darknet/dets/img/det
mv det.txt ./dets/img/det
# mkdir ../darknet/dets/person2/det
# mv det.txt ./dets/person2/det

cd ../deep_sort
source ./env/bin/activate
python3 ./tools/generate_detections.py --model './resources/networks/mars-small128.pb' --mot_dir '../David3/' --detection_dir '../darknet/dets/' --output_dir '../EC720_OcclusionTracking/dets/'
# python3 ./tools/generate_detections.py --model './resources/networks/mars-small128.pb' --mot_dir '../Subway/' --detection_dir '../darknet/dets/' --output_dir '../EC720_OcclusionTracking/dets/'
# python3 ./tools/generate_detections.py --model './resources/networks/mars-small128.pb' --mot_dir '../Human3/' --detection_dir '../darknet/dets/' --output_dir '../EC720_OcclusionTracking/dets/'
# python3 ./tools/generate_detections.py --model './resources/networks/mars-small128.pb' --mot_dir '../Challenge1_SEQ1/' --detection_dir '../darknet/dets/' --output_dir '../EC720_OcclusionTracking/dets/'

python3 deep_sort_app.py --sequence_dir='../David3/img/' --detection_file='../EC720_OcclusionTracking/dets/img.npy' --min_confidence=0.5 --nn_budget=100 --output_file='../EC720_OcclusionTracking/deep_sort/ds_results.txt' --display=True
# python3 deep_sort_app.py --sequence_dir='../Subway/img/' --detection_file='../EC720_OcclusionTracking/dets/img.npy' --min_confidence=0.5 --nn_budget=100 --output_file='../EC720_OcclusionTracking/deep_sort/ds_results.txt' --display=True
# python3 deep_sort_app.py --sequence_dir='../Human3/img/' --detection_file='../EC720_OcclusionTracking/dets/img.npy' --min_confidence=0.5 --nn_budget=100 --output_file='../EC720_OcclusionTracking/deep_sort/ds_results.txt' --display=True
# python3 deep_sort_app.py --sequence_dir='../Challenge1_SEQ1/person2' --detection_file='../EC720_OcclusionTracking/dets/person2.npy' --min_confidence=0.5 --output_file='../EC720_OcclusionTracking/deep_sort/ds_results.txt' --display=False
