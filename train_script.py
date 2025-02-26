import os
import glob

base_path = "/home/hanxiao/Desktop/Research/proj-qqtt/proj-QQTT/data/different_types_SG"
dir_names = glob.glob(f"{base_path}/*")

for dir_name in dir_names:
    case_name = dir_name.split("/")[-1]
    os.system(f"python train_our.py --base_path {base_path} --case_name {case_name}")
    break
