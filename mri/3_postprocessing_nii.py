import os
import re
import shutil
import tempfile
import subprocess
from collections import defaultdict
from glob import glob

import numpy as np
import nibabel as nib
import pydicom
import SimpleITK as sitk
import ants
from tqdm import tqdm

def convert_to_ras_nii(save_dir):
    nii_files = glob(save_dir + "/*.nii.gz")
    for nii_file in nii_files:
        img = nib.load(nii_file)
        img_ras = nib.as_closest_canonical(img)
        nib.save(img_ras, nii_file)
    return

def convert_to_lps_nii(save_dir):
    nii_files = glob(os.path.join(save_dir, "*.nii.gz"))
    for nii_file in nii_files:
        img = nib.load(nii_file)
        lps_ornt = nib.orientations.axcodes2ornt(('L', 'P', 'S'))
        curr_ornt = nib.orientations.io_orientation(img.affine)
        transform = nib.orientations.ornt_transform(curr_ornt, lps_ornt)
        img_lps = img.as_reoriented(transform)
        nib.save(img_lps, nii_file)
    return

def compute_adc_from_dwi(dwi_files, output_path):
    b0_img_path = None
    bx_img_path = None
    b_value = None

    for path in dwi_files:
        filename = os.path.basename(path).lower()
        if 'b0' in filename:
            b0_img_path = path
        elif 'b800' in filename:
            bx_img_path = path
            b_value = 800
        elif 'b900' in filename:
            bx_img_path = path
            b_value = 900
        elif 'b1000' in filename and bx_img_path is None:
            bx_img_path = path
            b_value = 1000
    
    if not b0_img_path or not bx_img_path:
        return

    b0_img = sitk.ReadImage(b0_img_path, sitk.sitkFloat32)
    bx_img = sitk.ReadImage(bx_img_path, sitk.sitkFloat32)
    b0_arr = sitk.GetArrayFromImage(b0_img)
    bx_arr = sitk.GetArrayFromImage(bx_img)

    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.divide(bx_arr, b0_arr, out=np.zeros_like(bx_arr), where=b0_arr > 0)
        adc_arr = -1.0 / b_value * np.log(ratio)
        adc_arr[np.isnan(adc_arr)] = 0
        adc_arr[adc_arr < 0] = 0
        adc_arr[adc_arr > 3.0] = 0  
        b0_thresh = np.percentile(b0_arr, 80)
        bx_thresh = np.percentile(bx_arr, 80)
        mask = (b0_arr > b0_thresh) | (bx_arr > bx_thresh)
        adc_arr[~mask] = 0

    adc_img = sitk.GetImageFromArray(adc_arr)
    adc_img.CopyInformation(b0_img)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    sitk.WriteImage(adc_img, output_path)
    print(f"ADC saved to: {output_path}")

root_dir = "./nii/"
subject_folders = glob(root_dir + '*')
print(len(subject_folders))

for subject_folder in tqdm(subject_folders):
    print(subject_folder)
    dwi_files = [os.path.join(subject_folder, f) for f in os.listdir(subject_folder) if f.startswith('DWI_')]
    adc_path = os.path.join(subject_folder, 'ADC.nii.gz')
    if not os.path.exists(adc_path):
        compute_adc_from_dwi(dwi_files, adc_path)

    # breast_path = os.path.join(subject_folder, "breast.nii.gz")
    # tumor_path = os.path.join(subject_folder, "tumor.nii.gz")

    # if not os.path.exists(breast_path):
    #     print(f"Not found corresponding breast.nii.gz: {breast_path}")
    #     continue
    # if not os.path.exists(tumor_path):
    #     print(f"Not found corresponding tumor.nii.gz: {tumor_path}")
    #     continue
    try:
        # tumor_img = nib.load(tumor_path)
        # breast_img = nib.load(breast_path)
        # tumor_data = tumor_img.get_fdata()
        # breast_data = breast_img.get_fdata()
        # if tumor_data.shape != breast_data.shape:
        #     print(f"[Error] Size mismatch: {tumor_path}")
        #     continue
        # tumor_data = np.logical_and(tumor_data > 0, breast_data > 0).astype(np.uint8)
        # tumor_data = nib.Nifti1Image(tumor_data, affine=breast_img.affine, header=breast_img.header)
        # nib.save(tumor_data, tumor_path)
        convert_to_ras_nii(subject_folder)
        
    except Exception as e:
        print(f"[Error] Processing failed:")
        print(e)