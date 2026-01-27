from glob import glob
import re
import os
import numpy as np
import torch
from tqdm import tqdm
from PIL import Image
import shutil
import torch.nn.functional as F
import sys
import torchvision
import cv2
sys.path.append('../example_data_and_weight/ultrasound/yolov5')
from models.common import DetectMultiBackend
from inference import *
from torchvision import transforms, models
import torch.nn as nn
from sklearn.metrics.pairwise import cosine_similarity

def yolo_detection(orginal_img, model, conf_thres, max_det=1, iou_thres=0):
    h, w = orginal_img.shape[:2]
    input_np = letterbox(orginal_img, 640, stride=32, auto=True)[0]
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
    return all_detections

def create_color_mask(hsv_img):
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([180, 255, 255])
    lower_blue = np.array([100, 100, 100])
    upper_blue = np.array([130, 255, 255])
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])
    mask_red = cv2.inRange(hsv_img, lower_red1, upper_red1) | cv2.inRange(hsv_img, lower_red2, upper_red2)
    mask_blue = cv2.inRange(hsv_img, lower_blue, upper_blue)
    mask_yellow = cv2.inRange(hsv_img, lower_yellow, upper_yellow)
    color_mask = mask_red | mask_blue | mask_yellow
    return color_mask

def expand_bbox_fixed(bbox, top=75, bottom=75, left=75, right=75, image_shape=None):
    x0, y0, x1, y1 = bbox
    x0 = x0 - left
    y0 = y0 - top
    x1 = x1 + right
    y1 = y1 + bottom
    if image_shape is not None:
        img_h, img_w, _ = image_shape
        x0 = max(x0, 0)
        y0 = max(y0, 0)
        x1 = min(x1, img_w)
        y1 = min(y1, img_h)
    return x0, y0, x1, y1

def color_doppler(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = create_color_mask(hsv)
    color_pixel_count = cv2.countNonZero(mask)
    color_ratio = color_pixel_count / img.size
    return color_ratio

def remove_similar_images(samples, threshold=0.99):
    filtered_samples = []
    used_indices = set()
    for i in range(len(samples)):
        if i in used_indices:
            continue
        filtered_samples.append(samples[i])
        for j in range(i + 1, len(samples)):
            if j in used_indices:
                continue
            sim = cosine_similarity(
                samples[i]['features'].reshape(1, -1),
                samples[j]['features'].reshape(1, -1))[0][0]
            if sim > threshold:
                used_indices.add(j)
    return filtered_samples

def pad_image_to_square(image_path, output_path):
    try:
        with Image.open(image_path) as img:
            width, height = img.size
            if width == height:
                img.save(output_path)
                return
            max_side = max(width, height)
            new_img = Image.new('RGB', (max_side, max_side), (0, 0, 0))
            left = (max_side - width) // 2
            top = (max_side - height) // 2
            new_img.paste(img, (left, top))
            new_img.save(output_path)
    except Exception as e:
        print(f"Error padding image {image_path}: {e}")

if __name__ == '__main__':
    
    transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor()])
    subject_paths = glob('../example_data_and_weight/ultrasound/data/*')
    base_save_root = './preprocessing/'
    final_save_root = './processed_data/'
    device = torch.device('cuda:0')
    subject_paths.sort(key=lambda x: [int(c) if c.isdigit() else c for c in re.split(r'(\d+)', x)])
    
    model_selection = models.resnet18(pretrained=True)
    model_selection = torch.nn.Sequential(*list(model_selection.children())[:-1])
    model_selection = model_selection.to(device)
    model_selection.eval()
    
    model_risk = models.resnet18(pretrained=False)
    num_ftrs = model_risk.fc.in_features
    model_risk.fc = nn.Linear(num_ftrs, 2)
    model_risk.load_state_dict(torch.load('../example_data_and_weight/ultrasound/yolov5/weight/risk_model.pth', map_location=device))
    model_risk = model_risk.to(device)
    model_risk.eval()
    
    model_crop_box = Get_Model("../example_data_and_weight/ultrasound/yolov5/weight/box.pt").to(device)
    model_crop_box.eval()
    model_crop_icon = Get_Model("../example_data_and_weight/ultrasound/yolov5/weight/icon.pt").to(device)
    model_crop_icon.eval()
    model_crop_lesion = Get_Model("../example_data_and_weight/ultrasound/yolov5/weight/lesion.pt").to(device)
    model_crop_lesion.eval()
    
    color_threshold = 1e-4
    
    for subject_path in tqdm(subject_paths):
        subject_path = subject_path.replace("\\", "/")
        id = subject_path.split('/')[-1]
        raw_paths = glob(subject_path + '/**/*.png', recursive=True)
        num_L = 0
        num_R = 0
        samples = []
        
        for img_path in raw_paths:
            img_original = cv2.imread(img_path)
            detection = yolo_detection(img_original, model_crop_icon, conf_thres=0.1, max_det=1, iou_thres=0.3)
            if detection == []:
                continue
            side = detection[0]['cls']
            if side == 0:
                side = 'L'
                num_L += 1
                save_name = side + '_' + str(num_L) + '.png'
            else:
                side = 'R'
                num_R += 1
                save_name = side + '_' + str(num_R) + '.png'
            
            save_path = os.path.join(base_save_root, id, 'raw', save_name)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            shutil.copy(img_path, save_path)
            
            detection = yolo_detection(img_original, model_crop_box, conf_thres=0.2, max_det=1, iou_thres=0.3)
            if detection == []:
                continue
            x0, y0, x1, y1 = detection[0]['bbox']
            img_cropped = img_original[y0:y1, x0:x1]
            save_path = os.path.join(base_save_root, id, 'window', save_name)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            cv2.imwrite(save_path, img_cropped)
            
            detection = yolo_detection(img_cropped, model_crop_lesion, conf_thres=0.1, max_det=1, iou_thres=0.3)
            if detection == []:
                continue
            x0, y0, x1, y1 = detection[0]['bbox']
            bbox_width = x1 - x0
            bbox_height = y1 - y0
            x0, y0, x1, y1 = expand_bbox_fixed(detection[0]['bbox'], image_shape=img_cropped.shape)
            img_lesion = img_cropped[y0:y1, x0:x1]
            
            save_path = os.path.join(base_save_root, id, 'lesion', save_name)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            cv2.imwrite(save_path, img_lesion)
            
            color_ratio = color_doppler(img_lesion) 
            if color_ratio > color_threshold or (bbox_width < 50 and bbox_height < 50) or bbox_height / bbox_width > 1.5:
                continue
            
            lesion_tensor = transform(img_lesion).unsqueeze(0).to(device)  
            features = model_selection(lesion_tensor).squeeze().detach().cpu().numpy()
            with torch.no_grad():
                output = model_risk(lesion_tensor)
                risk = F.softmax(output, dim=1)[0][1].item()
                samples.append({'features': features, 'risk': risk, 'path': save_path})
        
        if samples == []:
            print(f"Skip {id}")
            continue
            
        samples = remove_similar_images(samples)
        left_samples = [s for s in samples if os.path.basename(s['path']).startswith('L_')]
        right_samples = [s for s in samples if os.path.basename(s['path']).startswith('R_')]
        left_top2 = sorted(left_samples, key=lambda x: x['risk'], reverse=True)[:2]
        right_top2 = sorted(right_samples, key=lambda x: x['risk'], reverse=True)[:2]
        left_risk_sum = sum(s['risk'] for s in left_top2)
        right_risk_sum = sum(s['risk'] for s in right_top2)
        final_selected = left_top2 if left_risk_sum >= right_risk_sum else right_top2
        
        os.makedirs(os.path.join(base_save_root, id, 'selected'), exist_ok=True)
        os.makedirs(os.path.join(final_save_root, id), exist_ok=True)
        
        for selected in final_selected:
            shutil.copy(selected['path'], os.path.join(base_save_root, id, 'selected', os.path.basename(selected['path'])))
            final_output_path = os.path.join(final_save_root, id, os.path.basename(selected['path']))
            pad_image_to_square(selected['path'], final_output_path)