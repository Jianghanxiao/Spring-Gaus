import os
import glob

base_path = "/home/hanxiao/Desktop/Research/proj-qqtt/proj-QQTT/data/different_types_SG"
dir_names = glob.glob(f"{base_path}/*")

# rerun_cases = [
#     # "double_lift_cloth_3",
#     "rope_double_hand",
#     # "single_lift_dinosor",
#     # "single_lift_rope",
#     # "single_push_rope_4",
# ]

for dir_name in dir_names:
    case_name = dir_name.split("/")[-1]
    # if case_name not in rerun_cases:
    #     continue
    print(f"Processing {case_name} !!!!!!!!!!!!!!!!!")

    os.system(f"python train_our.py --base_path {base_path} --case_name {case_name}")