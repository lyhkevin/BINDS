import os
from glob import glob
import sys
sys.path.append('../example_data_and_weight/mri/nnUnetv1/')
from nnunet.inference.predict import predict_from_folder
import shutil
import tempfile

os.environ["RESULTS_FOLDER"] = "../example_data_and_weight/mri/nnUnetv1/nnUNet_trained_models"
MODEL_PATH = "../example_data_and_weight/mri/nnUnetv1/nnUNet_trained_models/nnUNet/3d_fullres/Task555_breast/nnUNetTrainerV2__nnUNetPlansv2.1"
nii_list = glob('./nii/**/P2.nii.gz', recursive=True)
print('number of samples:', len(nii_list))

for nii_path in nii_list:
    input_dir = os.path.dirname(nii_path)
    filename = os.path.basename(nii_path)
    case_id = os.path.splitext(os.path.splitext(filename)[0])[0]  
    print(f"Processing: {nii_path}")
  
    with tempfile.TemporaryDirectory() as tmpdir:
        input_tmp = os.path.join(tmpdir, "imagesTs")
        output_tmp = os.path.join(tmpdir, "output")
        os.makedirs(input_tmp, exist_ok=True)
        os.makedirs(output_tmp, exist_ok=True)

        temp_input_file = os.path.join(input_tmp, f"{case_id}_0000.nii.gz")
        shutil.copy(nii_path, temp_input_file)
        try:
            predict_from_folder(
                model=MODEL_PATH,                         
                input_folder=input_tmp,
                output_folder=output_tmp,           
                folds=(0,),                                 
                save_npz=False,                           
                num_threads_preprocessing=1,
                num_threads_nifti_save=1,
                lowres_segmentations=None,                  
                part_id=0,                                   
                num_parts=1,
                tta=False,
                overwrite_existing=True,
                mode="normal"
            )
        except Exception as e:
            print(f"Error processing {case_id}: {e}")
        predicted_file = os.path.join(output_tmp, f"{case_id}.nii.gz")
        output_file_path = os.path.join(input_dir, f"breast.nii.gz")
        shutil.copy(predicted_file, output_file_path)
        print(f"Saved segmentation to: {output_file_path}")