import pandas as pd
import os
from glob import glob
import random
import shutil
from tqdm import tqdm

import sys
import torchvision
sys.path.append('../example_data_and_weight/mammogram/yolov5')
from models.common import DetectMultiBackend
from inference import *
import json
import cv2
import re
from util import *

def lesion_detection(orginal_img, model, conf_thres, max_det=5, iou_thres=0):
    h, w = orginal_img.shape[:2]
    input_np = letterbox(orginal_img, 1280, stride=32, auto=True)[0]
    input = input_np.transpose((2, 0, 1))[::-1] 
    input = np.ascontiguousarray(input)
    input = torch.from_numpy(input).unsqueeze(0).to(device)
    input = input.float()
    input /= 255
    all_detections = []
    with torch.no_grad():
        pred = model(input, augment=False, visualize=False)
        pred = non_max_suppression(pred, conf_thres=conf_thres, iou_thres=iou_thres, classes=None, max_det=max_det)
        x_original, y_original = orginal_img.shape[1], orginal_img.shape[0]
        x_resized, y_resized = input_np.shape[0], input_np.shape[1]
        for i, det in enumerate(pred):
            for *xyxy, conf, cls in reversed(det):
                x0, y0, x1, y1 = xyxy[0].item(), xyxy[1].item(), xyxy[2].item(), xyxy[3].item()
                x0, y0, x1, y1 = adjust_coordinate(x0, y0, x1, y1, y_resized, x_resized)
                image = resize_image(input_np, [[x0, y0, x1, y1, 0]], y_original, x_original)
                x0, y0, x1, y1 = image['bboxes'][0][0], image['bboxes'][0][1], image['bboxes'][0][2], image['bboxes'][0][3]
                x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
                x_center, y_center, box_width, box_height = get_yolo_coordinates(x0, y0, x1, y1, x_original, y_original)
                all_detections.append({
                    'bbox': [x0, y0, x1, y1],  
                    'yolo': [x_center, y_center, box_width, box_height],
                    'conf': conf.item(),
                    'cls': int(cls.item())})
    all_detections = sorted(all_detections, key=lambda d: (d['bbox'][2] - d['bbox'][0]) * (d['bbox'][3] - d['bbox'][1]), reverse=True)
    return all_detections

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
path = '../example_data_and_weight/mammogram/yolov5/weight/detection.pt'
model_mass = Get_Model(path)

img_root = '../example_data_and_weight/mammogram/data/'
save_base_path = './yolo_output/'
save_img_path = save_base_path + 'images/'
save_label_path = save_base_path + 'labels/'
os.makedirs(save_img_path, exist_ok=True)
os.makedirs(save_label_path, exist_ok=True)
with open(os.path.join(save_label_path, 'classes.txt'), 'w') as f:
    f.write('0\n')

max_mass = 1

subject_paths = glob(img_root + '*')
for subject_path in tqdm(subject_paths):
    subject_name = os.path.basename(subject_path)
    dcm_paths = glob(subject_path + '/**/*.dcm', recursive=True)
    for dcm_path in dcm_paths:
       
        dcm_name = os.path.basename(dcm_path)
        img = load_dicom_image(dcm_path)
        img, _, _ = np_ExtractBreast(img)
        img = np.stack([img] * 3, axis=-1)
        img_with_annotation = img.copy()
        detected_mass = lesion_detection(img, model_mass, conf_thres=0.05, max_det=max_mass, iou_thres=0.3)
        
        img_name = subject_name + '_' + dcm_name
        label_name = subject_name + '_' + dcm_name
        img_name = label_name.replace(".dcm", ".png")
        label_name = label_name.replace(".dcm", ".txt")
        
        cv2.imwrite(os.path.join(save_img_path, img_name), img)
        with open(os.path.join(save_label_path, label_name), 'w') as f:
            f.write('')       
            
        with open(os.path.join(save_label_path, label_name), 'w') as f:
            if len(detected_mass) != 0:
                for lesion in detected_mass:
                    x_center, y_center, width, height = lesion['yolo']
                    cls = lesion['cls'] 
                    f.write(f"{cls} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
            else:
                pass
            
            
        