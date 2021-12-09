cp gilberty_script_iraei.py ../darknet/gilberty_script_iraei.py

cd ../darknet
python3 gilberty_script_iraei.py --img_dir '/home/gilbert/Downloads/David3/img/' --gt_path './david3_gt.txt' --display
# python3 gilberty_script_iraei.py --img_dir '/home/gilbert/Downloads/David3/img/' --gt_path './david3_gt.txt' --display --sort_path '../EC720_OcclusionTracking/deep_sort/ds_results.txt'

# python3 gilberty_script_iraei.py --img_dir '/home/gilbert/Downloads/Human3/img/' --gt_path './human3_gt.txt' --display
# python3 gilberty_script_iraei.py --img_dir '/home/gilbert/Downloads/Human3/img/' --gt_path './human3_gt.txt' --display --sort_path '../EC720_OcclusionTracking/deep_sort/ds_results.txt'


# python3 gilberty_script_iraei.py --img_dir '../Challenge1_SEQ1/person2/' --display
# python3 gilberty_script_iraei.py --img_dir '../Challenge1_SEQ1/person2/' --sort_path '../EC720_OcclusionTracking/deep_sort/ds_results.txt' --display
# python3 gilberty_script_iraei.py --img_dir '../Challenge1_SEQ2/person3/' --display
# python3 gilberty_script_iraei.py --img_dir '../Challenge1_SEQ3/20190604_120145UTC_BOS_copy1/' --display
# python3 gilberty_script_iraei.py --img_dir '../Challenge2_SEQ1/Bulgaria_excel-67_sc1.1_RGB/' --display

# This sequence is pretty bad
# python3 gilberty_script_iraei.py --img_dir '/home/gilbert/Downloads/Subway/img/' --gt_path './subway_gt.txt' --display
# python3 gilberty_script_iraei.py --img_dir '/home/gilbert/Downloads/Subway/img/' --gt_path './subway_gt.txt' --sort_path '../EC720_OcclusionTracking/deep_sort/ds_results.txt'
