import sys, os
from util import *
from glob import glob
import numpy as np
from PIL import Image
from tqdm import tqdm

if __name__ == '__main__':
    root = './yolo_output/'
    img_root = os.path.join(root, 'images/')
    label_root = os.path.join(root, 'labels/')
    save_root = './cropped_ROI/'
    
    labels = glob(os.path.join(label_root, '*.txt'))
    ids = set()
    valid_views = ['L CC', 'L MLO', 'R CC', 'R MLO']
    
    for label in labels:
        filename = os.path.basename(label)
        id = filename.split('_')[0]
        if any(v in filename for v in valid_views):
            ids.add(id)
            
    ids = sorted(list(ids))
    
    for id in tqdm(ids):
        label_paths = glob(os.path.join(label_root, f"{id}_*.txt"))
        
        for label_path in label_paths:
            view = None
            for v in valid_views:
                if v in label_path:
                    view = v
                    break
            
            if view is None:
                continue
                
            with open(label_path, 'r') as file:
                line = file.readline().strip()
                if not line:
                    continue
                
                data = line.split()
                _, x_norm, y_norm, w_norm, h_norm = map(float, data)
                
                img_path = label_path.replace('labels', 'images').replace('.txt', '.png')
                
                if not os.path.exists(img_path):
                    continue
                
                img = Image.open(img_path).convert('RGB')
                w, h = img.size
                
                x_px, y_px = yolo_to_center_coordinates(x_norm, y_norm, w, h)
                cropped = crop_centered_box(img, x_px, y_px, 768, 768, view[0])
                
                tumor_dir = os.path.join(save_root, str(id), 'tumor')
                orig_dir = os.path.join(save_root, str(id), 'original')
                os.makedirs(tumor_dir, exist_ok=True)
                os.makedirs(orig_dir, exist_ok=True)
                
                cropped.save(os.path.join(tumor_dir, f"{view}.png"))
                img.save(os.path.join(orig_dir, f"{view}.png"))