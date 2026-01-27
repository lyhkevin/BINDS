import pydicom
import re
import SimpleITK as sitk 
import os
import fcntl
import numpy as np
from collections import defaultdict
from glob import glob
import ants
import shutil
import tempfile
import subprocess
import nibabel as nib

def extract_number(filename):
    match = re.search(r'\d+', filename)
    if match:
        return match.group()
    return None

def copy_dicom_to_temp(dicom_file_list):
    temp_dir = tempfile.mkdtemp()
    seen_z = set()
    for file_path in dicom_file_list:
        ds = pydicom.dcmread(file_path, stop_before_pixels=True)
        z = round(float(ds.ImagePositionPatient[2]), 3)  
        if z not in seen_z:
            seen_z.add(z)
            shutil.copy(file_path, temp_dir)
        else:
            print('Skipping duplicate z-position:', z)
    return temp_dir

def get_dicom_metadata(path):
    dcm_file = pydicom.dcmread(path, stop_before_pixels=True)
    number_of_frames = getattr(dcm_file, 'NumberOfFrames', 1)
    is_multiframe = int(number_of_frames) > 1
    if is_multiframe:
        print('multi-frame dicom:', path, number_of_frames)
    manufacturer = getattr(dcm_file, 'Manufacturer', None)
    series_description = getattr(dcm_file, 'SeriesDescription', None)
    series_number = getattr(dcm_file, 'SeriesNumber', None)
    series_time = getattr(dcm_file, 'SeriesTime', None)
    instance_number = getattr(dcm_file, 'InstanceNumber', None)
    acq_time = getattr(dcm_file, 'AcquisitionTime', None)
    b_value = None
    if series_description in ['dyn_eTHRIVE', 'dyn_eTHRIVE SENSE', 'Ax VIBRANT-xv+C', 'Ax 3D Vibrant+C']:
        time = acq_time
    else:
        time = series_time
    series_name = None
    if manufacturer == None or series_description == None or series_number == None or instance_number == None or series_time == None or acq_time == None:
        return None
    if 'FAT' in series_description:
        return None
    if ('ADC' in series_description and 'eADC' not in series_description and 'dADC' not in series_description):
        series_name = 'ADC'
    if series_description in ['t2_stir_tra_p3', 't2_tse_TRA_spair', 'T2W_SPAIR', 'WATER: Ax T2 FSE-IDEAL', 'WATER: Ax T2 FSE-IDEAL ASSET', 'WATER: Ax T2 IDEAL', 'stir_fse_tra', 'stir_fse_tra-fast', 'Ax T2 STIR ASSET', 'eT2W_SPAIR SENSE', 'tirm_tra', 'tirm_tra_p2', 'tirm_tra_p3', 't2_tirm_tse_tra', 't2_tirm_tse_tra_p2', 't2_tirm_tse_tra_p3']:
        series_name = 'T2'
    if series_description in ['dyn_eTHRIVE', 'Ax 3D Vibrant+C', 't1_fl3d_tra_dyna_spair_1+5', 't1_fl3d_tra_dyna_spair_1+6', 't1_fl3d_tra_dyna_spair_1+7', 't1_fl3d_tra_dynaVIEWS_spair_1+5', 't1_fl3d_tra_dynaVIEWS_spair_1+6', 't1_fl3d_tra_dynaVIEWS_spair_1+7', 'dyn_eTHRIVE SENSE', 't1_fl3d_tra_dynaVIEWS_1+5', 't1_fl3d_tra_dynaVIEWS_1+7', 't1_fl3d_tra_dynaVIEWS_spair_1+6_p2'] or re.match(r'^t1_quick3d_tra_fs_1\+6steps_R\d$', series_description) or 'AX VIBRANT' in series_description.upper(): #re.fullmatch(r's[0-9]-1', series_description)
        series_name = 'DCE'
    if 'DWI' in series_description and 'dReg' not in series_description: 
        series_name = 'DWI'
        tag = (0x0018, 0x9087)
        if tag in dcm_file:
            b_value = dcm_file[tag].value
        tag = (0x0043, 0x1039)
        if tag in dcm_file:
            values = dcm_file[tag].value
            b_value = values[0]
    if series_name == None:
        return None
    if series_name == 'DWI' and b_value == None:
        return None
    dcm_metadata = {'manufacturer': manufacturer, 'time': time, 'series_description': series_description, 
                    'series_name': series_name, 'series_number': series_number, 'instance_number': instance_number, 'path': path, 'b_value': b_value}
    return dcm_metadata

def is_processed(patient_id, processed_file):
    if not os.path.exists(processed_file):
        return False
    with open(processed_file, 'r') as f:
        return str(patient_id) in f.read()

def mark_as_processed(patient_id, processed_file):
    with open(processed_file, 'a+') as f:
        fcntl.flock(f, fcntl.LOCK_EX)  
        f.write(str(patient_id) + '\n')
        fcntl.flock(f, fcntl.LOCK_UN)

def resample_moving_to_fixed(fixed_image, moving_image):
    resample = sitk.ResampleImageFilter()
    resample.SetReferenceImage(fixed_image)
    resample.SetInterpolator(sitk.sitkLinear)
    resample.SetDefaultPixelValue(0)
    resample.SetTransform(sitk.Transform()) 
    return resample.Execute(moving_image)

def resampleVolume(outspacing, vol):
    outsize = [0, 0, 0]
    inputsize = vol.GetSize()
    inputspacing = vol.GetSpacing()
    transform = sitk.Transform()
    transform.SetIdentity()
    outsize[0] = int(inputsize[0] * inputspacing[0] / outspacing[0] + 0.5)
    outsize[1] = int(inputsize[1] * inputspacing[1] / outspacing[1] + 0.5)
    outsize[2] = int(inputsize[2] * inputspacing[2] / outspacing[2] + 0.5)
    resampler = sitk.ResampleImageFilter()
    resampler.SetTransform(transform)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetOutputOrigin(vol.GetOrigin())
    resampler.SetOutputSpacing(outspacing)
    resampler.SetOutputDirection(vol.GetDirection())
    resampler.SetSize(outsize)
    newvol = resampler.Execute(vol)
    return newvol

def dicom_nii(path):
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(path)
    series_IDs = reader.GetGDCMSeriesIDs(path)
    dicom_names = reader.GetGDCMSeriesFileNames(path, series_IDs[0])
    reader.SetFileNames(dicom_names)
    image_itk = reader.Execute()
    return image_itk

def rm_zeros(ss):
    new = sitk.GetArrayFromImage(ss)
    f = np.where(new<0)
    new[f] = 0
    new = sitk.GetImageFromArray(new)
    new.SetSpacing(ss.GetSpacing())
    new.SetDirection(ss.GetDirection())
    new.SetOrigin(ss.GetOrigin())
    return new

def resize_image_itk(ori_img, target_img, resamplemethod=sitk.sitkNearestNeighbor):
    resampler = sitk.ResampleImageFilter()
    resampler.SetSize(target_img.GetSize())
    resampler.SetOutputSpacing(target_img.GetSpacing())
    resampler.SetOutputOrigin(target_img.GetOrigin())
    resampler.SetOutputDirection(target_img.GetDirection())
    resampler.SetInterpolator(resamplemethod)
    resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
    resampler.SetOutputPixelType(ori_img.GetPixelID())
    resampler.SetReferenceImage(target_img)
    return resampler.Execute(ori_img)

def register_to_P0(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    nii_files = glob(os.path.join(input_dir, '*.nii.gz'))
    target_img_sitk = sitk.ReadImage(os.path.join(input_dir, 'P0.nii.gz'))
    for f in nii_files:
        if 'T2' in f or 'ADC' in f or 'DWI' in f:
            ori_img_sitk = sitk.ReadImage(f)
            resampled_img = resize_image_itk(ori_img_sitk, target_img_sitk)
            resampled_img.SetDirection(target_img_sitk.GetDirection())
            resampled_img.SetOrigin(target_img_sitk.GetOrigin())
            resampled_img.SetSpacing(target_img_sitk.GetSpacing())
            out_path = os.path.join(output_dir, os.path.basename(f))
            sitk.WriteImage(resampled_img, out_path)                           
    print('finish registration')
    return

def compute_adc_from_dwi(dwi_files, output_path, threshold=0):
    
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
        elif 'b1000' in filename and bx_img_path is None:
            bx_img_path = path
            b_value = 1000

    if not b0_img_path or not bx_img_path:
        return

    b0_img = sitk.ReadImage(b0_img_path, sitk.sitkFloat32)
    bx_img = sitk.ReadImage(bx_img_path, sitk.sitkFloat32)
    b0_arr = sitk.GetArrayFromImage(b0_img)
    bx_arr = sitk.GetArrayFromImage(bx_img)
    if threshold is None:
        combined = b0_arr + bx_arr
        threshold = np.percentile(combined, 50) 
    mask = (b0_arr > threshold) & (bx_arr > threshold)
    adc_arr = np.zeros_like(b0_arr, dtype=np.float32)
    adc_arr[mask] = -1.0 * b_value * np.log(bx_arr[mask] / b0_arr[mask])
    adc_arr[adc_arr < 0] = 0
    adc_img = sitk.GetImageFromArray(adc_arr)
    adc_img.CopyInformation(b0_img)
    sitk.WriteImage(adc_img, output_path)
    return

def convert_to_ras_nii(save_dir):
    nii_files = glob(os.path.join(save_dir, "*.nii.gz"))
    for nii_file in nii_files:
        img = nib.load(nii_file)
        img_ras = nib.as_closest_canonical(img)
        nib.save(img_ras, nii_file)

def select_series_and_register(dcm_paths, output_dir, patient_id):
    dcm_metadata_all = []
    for i, dcm_path in enumerate(dcm_paths):
        dcm_metadata = get_dicom_metadata(dcm_path)
        if dcm_metadata == None:
            continue
        dcm_metadata_all.append(dcm_metadata)
    grouped_metadata = defaultdict(list)
    for metadata in dcm_metadata_all:
        series_name = metadata.get('series_name')
        series_number = metadata.get('series_number')
        series_description = metadata.get('series_description')
        time = metadata.get('time') 
        b_value = metadata.get('b_value') 
        key = (series_name, time, series_description, series_number, b_value)
        grouped_metadata[key].append(metadata)
        
    dce_thickness_list = []
    keys_to_delete = set()
    for key, value in grouped_metadata.items():
        series_name = key[0]
        num_elements = len(value)
        if series_name == 'DCE':
            if num_elements < 20:
                keys_to_delete.add(key)
            else:
                dce_thickness_list.append(num_elements)
        if (series_name == 'ADC' or series_name == 'T2') and num_elements < 5:
            keys_to_delete.add(key)
    if len(dce_thickness_list) == 0:
        print('without DCE MRI')
        return
    print('number of dce slices')
    print(dce_thickness_list)
    max_dce_thickness = max(dce_thickness_list)
    for key, value in grouped_metadata.items():
        series_name = key[0]
        num_elements = len(value)
        if series_name == 'DCE' and num_elements < max_dce_thickness:
            keys_to_delete.add(key)
    for key in keys_to_delete:
        del grouped_metadata[key]
        
    counts = {'T2': 0, 'DCE': 0, 'ADC': 0, 'DWI': 0}
    for key, value in grouped_metadata.items():
        series_name = key[0]
        counts[series_name] += 1
    print(counts)
    
    if counts['DCE'] < 6:
        print('not enough DCE slices')
        return
    
    if counts['DCE'] > 9:
        print('too many DCE slices')
        return
    
    grouped_metadata = dict(sorted(grouped_metadata.items(),
                            key=lambda item: (item[0][3], item[0][1])))
    for key, dcm_list in grouped_metadata.items():
        dcm_list.sort(key=lambda x: int(x.get('instance_number', 1e9)))

    if counts['ADC'] > 1:
        adc_keys = [key for key in grouped_metadata if key[0] == 'ADC']
        selected_key = None
        for key in adc_keys:
            if '236_ADC' in str(key):
                selected_key = key
                break
            if 'resolve' in str(key):
                selected_key = key
                break
        if not selected_key:
            selected_key = adc_keys[0]
        for key in adc_keys:
            if key != selected_key:
                del grouped_metadata[key]
    
    if counts['T2'] > 1:
        t2_keys = [key for key in grouped_metadata if key[0] == 'T2']
        selected_key = t2_keys[0]
        for key in t2_keys:
            if key != selected_key:
                del grouped_metadata[key]
                
    if counts['DWI'] > 1:
        dwi_keys = [key for key in grouped_metadata if key[0] == 'DWI']
        b_value_to_key = {}
        for key in dwi_keys:
            b_value = key[4]  
            time = key[1]
            if b_value is None:
                continue
            if b_value not in b_value_to_key:
                b_value_to_key[b_value] = key
            else:
                existing_key = b_value_to_key[b_value]
                existing_time = existing_key[1]
                if time < existing_time:
                    del grouped_metadata[existing_key]
                    b_value_to_key[b_value] = key
                else:
                    del grouped_metadata[key]

    save_dir = os.path.join(output_dir, patient_id)
    os.makedirs(save_dir, exist_ok=True)
    dce_index = 0 
    
    for key, value in grouped_metadata.items():
        series_name, acq_time, series_description, series_number, b_value = key
        dcm_file_list = [item['path'] for item in value]
        
        temp_dir = copy_dicom_to_temp(dcm_file_list)
        img = dicom_nii(temp_dir)
        img = rm_zeros(img)
        img = resampleVolume([1.0, 1.0, 1.0], img)
        shutil.rmtree(temp_dir)
        
        if series_name == 'DCE':
            output_path = os.path.join(save_dir, f'P{dce_index}.nii.gz')
            dce_index += 1
        elif series_name == 'DWI' and counts['ADC'] == 0:
            b_value = int(b_value)
            output_path = os.path.join(save_dir, f'{series_name}_b{b_value}.nii.gz')
        else:
            output_path = os.path.join(save_dir, f'{series_name}.nii.gz')
        sitk.WriteImage(img, output_path)
    
    register_to_P0(save_dir, save_dir)
    convert_to_ras_nii(save_dir)
    adc_path = os.path.join(save_dir, 'ADC.nii.gz')
    if os.path.exists(adc_path):
        return
    # dwi_files = [os.path.join(save_dir, f) for f in os.listdir(save_dir) if f.startswith('DWI_')]
    # compute_adc_from_dwi(dwi_files, adc_path)
    return