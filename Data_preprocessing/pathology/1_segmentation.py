from xml.etree.ElementTree import parse
from glob import glob
from torchvision.utils import draw_segmentation_masks as draw_mask
import torchvision.transforms.functional as F
import torch
from einops import rearrange, reduce, repeat
import tqdm
import os
from preprocessing.filter import *
from torchvision import transforms
import openslide
import numpy as np
from tqdm import tqdm
from PIL import Image, ImageOps
from model.vision_transformer import *
import math
from torch.utils.data import Dataset, DataLoader
from utils.plot import *
from functools import partial
import torch.nn as nn

class dataset_info:
    def __init__(self):
        self.base_dir = './seg/'
        self.wsi_base_dir = '../example_data_and_weight/pathology/data'
        os.makedirs(self.base_dir, exist_ok=True)
        self.slide_dir = None
        self.svs = None
        self.slide = None
        self.mask = None
        self.svs_name = None
        self.path = None
        
        self.w_1x = None 
        self.h_1x = None
        self.w_40x = None
        self.h_40x = None
        
        self.threshold = 80 
        self.model_threshold = 0.2
        self.img_size = (224, 224) 
        self.max_patch_num = 500 
        
    def makedir(self):
        if not os.path.exists(self.path):
            print('no svs file:', self.name)
            return False
            
        self.slide_dir = self.base_dir + '/' + self.name + '/'
        
        if os.path.exists(self.slide_dir):
            print('dir already exist:', self.name, self.slide_dir)
            return False

        self.patch_w_1x = 64
        self.patch_h_1x = 64
        self.patch_w_40x = self.patch_w_1x * 16 
        self.patch_h_40x = self.patch_w_1x * 16
        os.makedirs(self.slide_dir, exist_ok=True)
        return True

class ImageDataset(Dataset):
    def __init__(self, image_list):
        self.image_list = image_list
        self.transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    def __len__(self):
        return len(self.image_list)
    def __getitem__(self, idx):
        image = self.image_list[idx].convert('RGB')
        image = self.transform(image)
        return image

def read_wsi(info):
    info.slide = openslide.OpenSlide(info.path)
    info.w_1x, info.h_1x = info.slide.level_dimensions[2]
    info.num_row_patch = math.floor(info.h_1x / info.patch_h_1x)
    info.num_col_patch = math.floor(info.w_1x / info.patch_w_1x)
    try:
        thumbnail = info.slide.read_region((0, 0), 2, info.slide.level_dimensions[2]).convert("RGB")
    except:
        thumbnail = info.slide.get_thumbnail((info.w_1x, info.h_1x)).convert("RGB")
    data = np.array(thumbnail)
    black_pixels = np.all(data == [0, 0, 0], axis=-1)
    data[black_pixels] = [255, 255, 255]
    thumbnail = Image.fromarray(data)
    thumbnail.save(f"{info.slide_dir}thumbnail_{info.num_row_patch}_{info.num_col_patch}.png")
    info.slide_np = np.clip(data, 0, 255).astype(np.uint8)

def get_patch(row_id, col_id, info):
    x = row_id * info.patch_w_1x
    y = col_id * info.patch_h_1x
    patch = info.slide.read_region((y * 16, x * 16), 0, (info.patch_w_40x, info.patch_h_40x))
    patch = patch.resize(info.img_size)
    return patch

def otsu_threshold(info):
    print('perform otsu threshold..................')
    grayscale = filter_rgb_to_grayscale(info.slide_np)
    complement = filter_complement(grayscale)
    filtered = filter_otsu_threshold(complement)
    info.mask = np.clip(filtered, 0, 1)
    np.save(info.slide_dir + 'otsu_mask.npy', info.mask)
    return

def get_tissue_rate(row_id, col_id, info):
    x = row_id * info.patch_w_1x
    y = col_id * info.patch_h_1x
    if x + info.patch_w_1x > info.mask.shape[0] or y + info.patch_h_1x > info.mask.shape[1]:
        return 0
    mask = info.mask[x:x+info.patch_w_1x, y:y+info.patch_h_1x]
    tissue_rate = int(np.sum(mask) * 100.0 / (mask.shape[0] * mask.shape[1]))
    return tissue_rate

def grid_wsi(info):
    print("grid and select patch from whole slide image................")
    patch_prediction = np.zeros((info.num_row_patch, info.num_col_patch, 2))
    pix_prediction = np.zeros((info.slide_np.shape[0], info.slide_np.shape[1]))
    print('num row:', info.num_row_patch, 'num col:', info.num_col_patch)
    count = 0
    
    if info.num_row_patch < info.num_col_patch:
        for col_id in range(4, info.num_col_patch - 4, 1):
            patches = []
            row_ids = []
            for row_id in range(4, info.num_row_patch - 4, 1):
                tissue_rate = get_tissue_rate(row_id, col_id, info)
                if tissue_rate > info.threshold:
                    patch = get_patch(row_id, col_id, info)
                    patches.append(patch)
                    row_ids.append(row_id)
            
            num = 0
            if len(patches) > 0:
                dataset = ImageDataset(patches)
                dataloader = DataLoader(dataset, batch_size=100, shuffle=False, num_workers=4)
                
                for i, img in enumerate(dataloader):
                    img = img.to(device)
                    with torch.no_grad():
                        outputs = info.vit(img)
                        outputs = torch.nn.functional.softmax(outputs, dim=1)
                    
                    for j in range(len(outputs)):
                        patch_prediction[row_ids[num], col_id, :] = outputs[j].cpu().numpy()
                        if outputs[j][1] >= info.model_threshold:
                            count = count + 1
                        num += 1
            if count > info.max_patch_num:
                break               
    else:
        for row_id in range(4, info.num_row_patch - 4, 1):
            patches = []
            col_ids = []
            for col_id in range(4, info.num_col_patch - 4, 1):
                tissue_rate = get_tissue_rate(row_id, col_id, info)
                if tissue_rate > info.threshold:
                    patch = get_patch(row_id, col_id, info)
                    patches.append(patch)
                    col_ids.append(col_id)
            
            num = 0
            if len(patches) > 0:
                dataset = ImageDataset(patches)
                dataloader = DataLoader(dataset, batch_size=100, num_workers=4, shuffle=False)
                for i, img in enumerate(dataloader):
                    img = img.to(device)
                    with torch.no_grad():
                        outputs = info.vit(img)
                        outputs = torch.nn.functional.softmax(outputs, dim=1)
                    
                    for j in range(len(outputs)):
                        patch_prediction[row_id, col_ids[num], :] = outputs[j].cpu().numpy()
                        if outputs[j][1] >= info.model_threshold:
                            count = count + 1
                        num += 1
            if count > info.max_patch_num:
                break
            
    patch_prediction_tensor = torch.from_numpy(patch_prediction)
    patch_prediction_tensor = rearrange(patch_prediction_tensor, 'h w c -> c h w')
    
    target_h = 8 * info.num_row_patch
    target_w = 8 * info.num_col_patch
    
    patch_prediction_tensor = torch.nn.functional.interpolate(
        patch_prediction_tensor.unsqueeze(0), 
        size=(target_h, target_w), 
        mode='bilinear', 
        align_corners=False
    ).squeeze(0)
    
    prob_class_1 = patch_prediction_tensor[1, :, :].numpy()
    patch_prediction_np = (prob_class_1 >= info.model_threshold).astype(np.uint8)
    
    scale_factor = int(info.patch_w_1x / 8)
    patch_prediction_np = repeat(patch_prediction_np, 'h w -> (h x) (w y)', x=scale_factor, y=scale_factor)
    
    h_cut = min(pix_prediction.shape[0], patch_prediction_np.shape[0])
    w_cut = min(pix_prediction.shape[1], patch_prediction_np.shape[1])
    
    pix_prediction[0:h_cut, 0:w_cut] = patch_prediction_np[0:h_cut, 0:w_cut]
    
    prediction_img = drawing_mask(pix_prediction, info.slide_np)
    prediction_mask = drawing_annotation_mask(pix_prediction, info.slide_np)
    prediction_img.save(info.slide_dir + 'prediction_overlayed.png')
    prediction_mask.save(info.slide_dir + 'prediction_mask.png')
    np.save(info.slide_dir + 'segmentation_mask.npy', pix_prediction)

if __name__ == '__main__':
    info = dataset_info()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    vit = VisionTransformer(patch_size=16, num_classes=2, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6)).to(device)
    state_dict = torch.load('../example_data_and_weight/pathology/weight/classification.pth', map_location=torch.device(device))
    vit.load_state_dict(state_dict, strict=False)
    info.vit = vit 
    info.vit.eval()
    
    svs_files = glob(os.path.join(info.wsi_base_dir, '**', '*.svs'), recursive=True)
    
    for idx, svs_path in enumerate(tqdm(svs_files)):
        info.id = idx
        info.path = svs_path
        info.name = os.path.splitext(os.path.basename(svs_path))[0]
        
        if info.makedir() == True:
            print('preprocessing whole slide image:', svs_path)
            read_wsi(info)
            otsu_threshold(info)
            grid_wsi(info)