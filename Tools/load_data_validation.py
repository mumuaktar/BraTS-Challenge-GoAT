import pandas as pd
import os
from sklearn.model_selection import train_test_split
import torch

def load_data_validation(input_dir):
    
  
    subjects = sorted(os.listdir(input_dir))
    data_list = []

    for subj in subjects:
        subj_folder = os.path.join(input_dir, subj)
        if not os.path.isdir(subj_folder):
            continue
        
        modality_map = {"t1n": 0, "t1c": 1, "t2w": 2, "t2f": 3}
        modality_paths = [None]*4

        files_in_subj = os.listdir(subj_folder)
        print(f"Subject: {subj}, Files: {files_in_subj}")  # Debug print

        for f in files_in_subj:
            f_lower = f.lower()
            for mod in modality_map:
                if mod in f_lower and f_lower.endswith(".nii.gz"):
                    modality_paths[modality_map[mod]] = os.path.join(subj_folder, f)

        if None in modality_paths:
            print(f"Warning: missing modalities for {subj}. Skipping.")
            continue

        data_list.append({"img": modality_paths, "subject_id": subj})

    # Now build your Dataset and DataLoader as needed using data_list

    # For example:
    from tools.transforms_multitask import test_transforms
    from monai.data import Dataset
    from torch.utils.data import DataLoader

    ds_val = Dataset(data=data_list, transform=test_transforms)
 


   
    val_loader = DataLoader(ds_val, num_workers=0, batch_size=1, shuffle=False,  pin_memory=True,drop_last=False)

    
    return val_loader