import os
from glob import glob
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from utils import *
from tqdm import tqdm
import shutil

root = '../example_data_and_weight/mri/data/'
output_dir = './nii/'

subject_paths = glob(root + '/*')
subject_paths.sort(key=lambda x: int(re.search(r'\d+', os.path.basename(x)).group())
                   if re.search(r'\d+', os.path.basename(x)) else float('inf'))

for subject_path in tqdm(subject_paths):
    print('start processing: '+ subject_path)
    try:
        dcm_metadata_all = []
        patient_id = extract_number(subject_path.split('/')[-1])
        dcm_paths = glob(subject_path + '/*.dcm', recursive=True)
        select_series_and_register(dcm_paths, output_dir, patient_id)
    except Exception as e:
        subject_dir = os.path.join(output_dir, patient_id)
        if os.path.exists(subject_dir):
            shutil.rmtree(subject_dir)
            print('delete patient directory')
        print(f"Error processing {subject_path}: {e}")
        
    