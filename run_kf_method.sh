cp gilberty_script_iraei.py ../darknet/gilberty_script_iraei.py

cd ../darknet
# python3 gilberty_script_iraei.py
# python3 gilberty_script_iraei.py --img_dir '../David3/img/'
# python3 gilberty_script_iraei.py --img_dir '/home/gilbert/Downloads/David3/img/' --gt_path './david3_gt.txt'
python3 gilberty_script_iraei.py --img_dir '/home/gilbert/Downloads/David3/img/' --gt_path './david3_gt.txt' --sort_path '../EC720_OcclusionTracking/deep_sort/ds_results.txt'
