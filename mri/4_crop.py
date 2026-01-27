import os
import sys
from glob import glob
import nibabel as nib
import numpy as np
import pandas as pd

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from crop_utils import *

data_root = "./nii/"
save_root = './cropped/'
file_paths = ['P0', 'P2', 'ADC', 'T2', 'breast'] #['P0', 'P2', 'ADC', 'T2', 'tumor', 'breast']
crop_size = (96, 96, 96)

breast_files = glob(os.path.join(data_root, "**/breast.nii.gz"), recursive=True)

for breast_path in breast_files:
    subject_path = os.path.dirname(breast_path)
    subject_id = os.path.basename(subject_path)
    
    qualified = all(os.path.exists(os.path.join(subject_path, f"{f}.nii.gz")) for f in file_paths)
    if not qualified:
        continue

    try:
        breast_img = nib.load(breast_path)
        breast_mask = breast_img.get_fdata() > 0
    except:
        continue

    for side in ['L', 'R']:
        save_dir = os.path.join(save_root, subject_id, side)
        if os.path.exists(save_dir):
            continue
        
        crop_bounds = crop_tumor_from_side(breast_path, side, crop_size)
        x1, x2 = crop_bounds[0]
        y1, y2 = crop_bounds[1]
        z1, z2 = crop_bounds[2]

        os.makedirs(save_dir, exist_ok=True)
        
        for file_name in file_paths:
            img_path = os.path.join(subject_path, file_name + '.nii.gz')
            img = nib.load(img_path)
            data = img.get_fdata()
            affine = img.affine
            
            data = data * breast_mask
            data = normalize_percentile(data)
            
            cropped_data = data[x1:x2, y1:y2, z1:z2]
            cropped_img = nib.Nifti1Image(cropped_data, affine)
            save_path = os.path.join(save_dir, file_name + '.nii.gz')
            nib.save(cropped_img, save_path)